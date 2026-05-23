import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]
    # Optional chunk1/source observation used by beta latent models. The normal
    # images/state remain the chunk2/current execution observation.
    condition_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    condition_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    condition_state: at.Float[ArrayT, "*b cs"] | None = None
    condition_tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    condition_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Optional structured meta-area inputs. The last dimension is 6 for the old
    # pose6d path or 12 for pose12d=[pos3, shape6, approach3].
    meta_area_poses: at.Float[ArrayT, "*b m d"] | None = None
    meta_area_dim_masks: at.Bool[ArrayT, "*b m d"] | None = None
    meta_area_types: at.Int[ArrayT, "*b m"] | None = None
    meta_area_masks: at.Bool[ArrayT, "*b m"] | None = None
    # Beta latent path: meta area in the chunk2 execution frame. This is used by
    # the meta-action head only. When missing, beta models fall back to
    # meta_area_* so runtime inference can keep supplying a single meta area.
    execution_meta_area_poses: at.Float[ArrayT, "*b m d"] | None = None
    execution_meta_area_dim_masks: at.Bool[ArrayT, "*b m d"] | None = None
    execution_meta_area_types: at.Int[ArrayT, "*b m"] | None = None
    execution_meta_area_masks: at.Bool[ArrayT, "*b m"] | None = None
    meta_action_target_poses: at.Float[ArrayT, "*b ah m d"] | None = None
    meta_action_target_dim_masks: at.Bool[ArrayT, "*b ah m d"] | None = None
    meta_action_target_masks: at.Bool[ArrayT, "*b ah m"] | None = None
    # Continuous runtime/training control for how strongly the policy should
    # trust structured meta inputs. Missing means "fully trust" for compatibility.
    meta_control_alpha: at.Float[ArrayT, "*b"] | None = None
    # Beta latent path: optional demonstration/reference action chunk used only
    # to infer latent tool-operation tokens. It is not a supervised target.
    reference_actions: at.Float[ArrayT, "*b ah ad"] | None = None
    reference_action_mask: at.Bool[ArrayT, "*b"] | None = None
    # Optional paired condition views used only by beta contrastive training.
    # They share the chunk1/source observation with reference_actions/meta_areas,
    # but are not used to decide which latent path is injected into chunk2.
    contrastive_meta_area_poses: at.Float[ArrayT, "*b m d"] | None = None
    contrastive_meta_area_dim_masks: at.Bool[ArrayT, "*b m d"] | None = None
    contrastive_meta_area_types: at.Int[ArrayT, "*b m"] | None = None
    contrastive_meta_area_masks: at.Bool[ArrayT, "*b m"] | None = None
    contrastive_reference_actions: at.Float[ArrayT, "*b ah ad"] | None = None
    contrastive_reference_action_mask: at.Bool[ArrayT, "*b"] | None = None
    # 0 means the supplied meta area is real/origin; 1 means it is imagined or
    # retargeted. This is only meaningful when meta-area tokens are present.
    meta_imagination_alpha: at.Float[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        if ("condition_tokenized_prompt" in data) != ("condition_tokenized_prompt_mask" in data):
            raise ValueError("condition_tokenized_prompt and condition_tokenized_prompt_mask must be provided together.")
        if "meta_areas" in data:
            meta_areas = data["meta_areas"]
            has_pose = "pose6d" in meta_areas or "pose12d" in meta_areas
            if not (has_pose and {"type", "mask"}.issubset(meta_areas)):
                raise ValueError("meta_areas must contain pose6d or pose12d, plus type and mask, when provided.")
        if "execution_meta_areas" in data:
            execution_meta_areas = data["execution_meta_areas"]
            has_pose = "pose6d" in execution_meta_areas or "pose12d" in execution_meta_areas
            if not (has_pose and {"type", "mask"}.issubset(execution_meta_areas)):
                raise ValueError(
                    "execution_meta_areas must contain pose6d or pose12d, plus type and mask, when provided."
                )
        if "meta_action_targets" in data:
            meta_targets = data["meta_action_targets"]
            has_pose = "pose6d" in meta_targets or "pose12d" in meta_targets
            if not (has_pose and "mask" in meta_targets):
                raise ValueError("meta_action_targets must contain pose6d or pose12d, plus mask, when provided.")
        # If images are uint8, convert them to [-1, 1] float32.
        for image_key in ("image", "condition_image"):
            if image_key not in data or data[image_key] is None:
                continue
            for key in data[image_key]:
                if data[image_key][key].dtype == np.uint8:
                    data[image_key][key] = data[image_key][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(data[image_key][key], "dtype") and data[image_key][key].dtype == torch.uint8:
                    data[image_key][key] = (
                        data[image_key][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                    )
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            condition_images=data.get("condition_image"),
            condition_image_masks=data.get("condition_image_mask"),
            condition_state=data.get("condition_state"),
            condition_tokenized_prompt=data.get("condition_tokenized_prompt"),
            condition_tokenized_prompt_mask=data.get("condition_tokenized_prompt_mask"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            meta_area_poses=(
                data.get("meta_areas", {}).get("pose12d")
                if "pose12d" in data.get("meta_areas", {})
                else data.get("meta_areas", {}).get("pose6d")
            ),
            meta_area_dim_masks=data.get("meta_areas", {}).get("dim_mask12"),
            meta_area_types=data.get("meta_areas", {}).get("type"),
            meta_area_masks=data.get("meta_areas", {}).get("mask"),
            execution_meta_area_poses=(
                data.get("execution_meta_areas", {}).get("pose12d")
                if "pose12d" in data.get("execution_meta_areas", {})
                else data.get("execution_meta_areas", {}).get("pose6d")
            ),
            execution_meta_area_dim_masks=data.get("execution_meta_areas", {}).get("dim_mask12"),
            execution_meta_area_types=data.get("execution_meta_areas", {}).get("type"),
            execution_meta_area_masks=data.get("execution_meta_areas", {}).get("mask"),
            meta_action_target_poses=(
                data.get("meta_action_targets", {}).get("pose12d")
                if "pose12d" in data.get("meta_action_targets", {})
                else data.get("meta_action_targets", {}).get("pose6d")
            ),
            meta_action_target_dim_masks=data.get("meta_action_targets", {}).get("dim_mask12"),
            meta_action_target_masks=data.get("meta_action_targets", {}).get("mask"),
            meta_control_alpha=data.get("meta_control", {}).get("alpha", data.get("meta_control_alpha")),
            reference_actions=data.get("reference_actions"),
            reference_action_mask=data.get("reference_action_mask"),
            contrastive_meta_area_poses=(
                data.get("contrastive_meta_areas", {}).get("pose12d")
                if "pose12d" in data.get("contrastive_meta_areas", {})
                else data.get("contrastive_meta_areas", {}).get("pose6d")
            ),
            contrastive_meta_area_dim_masks=data.get("contrastive_meta_areas", {}).get("dim_mask12"),
            contrastive_meta_area_types=data.get("contrastive_meta_areas", {}).get("type"),
            contrastive_meta_area_masks=data.get("contrastive_meta_areas", {}).get("mask"),
            contrastive_reference_actions=data.get("contrastive_reference_actions"),
            contrastive_reference_action_mask=data.get("contrastive_reference_action_mask"),
            meta_imagination_alpha=data.get("meta_control", {}).get(
                "imagination_alpha", data.get("meta_imagination_alpha")
            ),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        condition_images = result.pop("condition_images")
        condition_image_masks = result.pop("condition_image_masks")
        condition_state = result.pop("condition_state")
        condition_tokenized_prompt = result.pop("condition_tokenized_prompt")
        condition_tokenized_prompt_mask = result.pop("condition_tokenized_prompt_mask")
        meta_area_poses = result.pop("meta_area_poses")
        meta_area_dim_masks = result.pop("meta_area_dim_masks")
        meta_area_types = result.pop("meta_area_types")
        meta_area_masks = result.pop("meta_area_masks")
        execution_meta_area_poses = result.pop("execution_meta_area_poses")
        execution_meta_area_dim_masks = result.pop("execution_meta_area_dim_masks")
        execution_meta_area_types = result.pop("execution_meta_area_types")
        execution_meta_area_masks = result.pop("execution_meta_area_masks")
        meta_action_target_poses = result.pop("meta_action_target_poses")
        meta_action_target_dim_masks = result.pop("meta_action_target_dim_masks")
        meta_action_target_masks = result.pop("meta_action_target_masks")
        meta_control_alpha = result.pop("meta_control_alpha")
        reference_actions = result.pop("reference_actions")
        reference_action_mask = result.pop("reference_action_mask")
        contrastive_meta_area_poses = result.pop("contrastive_meta_area_poses")
        contrastive_meta_area_dim_masks = result.pop("contrastive_meta_area_dim_masks")
        contrastive_meta_area_types = result.pop("contrastive_meta_area_types")
        contrastive_meta_area_masks = result.pop("contrastive_meta_area_masks")
        contrastive_reference_actions = result.pop("contrastive_reference_actions")
        contrastive_reference_action_mask = result.pop("contrastive_reference_action_mask")
        meta_imagination_alpha = result.pop("meta_imagination_alpha")
        if meta_area_poses is not None or meta_area_types is not None or meta_area_masks is not None:
            pose_key = "pose12d" if getattr(meta_area_poses, "shape", ()) and meta_area_poses.shape[-1] == 12 else "pose6d"
            result["meta_areas"] = {
                pose_key: meta_area_poses,
                "type": meta_area_types,
                "mask": meta_area_masks,
            }
            if meta_area_dim_masks is not None:
                result["meta_areas"]["dim_mask12"] = meta_area_dim_masks
        if (
            execution_meta_area_poses is not None
            or execution_meta_area_types is not None
            or execution_meta_area_masks is not None
        ):
            pose_key = (
                "pose12d"
                if getattr(execution_meta_area_poses, "shape", ()) and execution_meta_area_poses.shape[-1] == 12
                else "pose6d"
            )
            result["execution_meta_areas"] = {
                pose_key: execution_meta_area_poses,
                "type": execution_meta_area_types,
                "mask": execution_meta_area_masks,
            }
            if execution_meta_area_dim_masks is not None:
                result["execution_meta_areas"]["dim_mask12"] = execution_meta_area_dim_masks
        if meta_action_target_poses is not None or meta_action_target_masks is not None:
            pose_key = (
                "pose12d"
                if getattr(meta_action_target_poses, "shape", ()) and meta_action_target_poses.shape[-1] == 12
                else "pose6d"
            )
            result["meta_action_targets"] = {
                pose_key: meta_action_target_poses,
                "mask": meta_action_target_masks,
            }
            if meta_action_target_dim_masks is not None:
                result["meta_action_targets"]["dim_mask12"] = meta_action_target_dim_masks
        if meta_control_alpha is not None:
            result["meta_control"] = {"alpha": meta_control_alpha}
        if meta_imagination_alpha is not None:
            result.setdefault("meta_control", {})["imagination_alpha"] = meta_imagination_alpha
        if condition_images is not None:
            result["condition_image"] = condition_images
        if condition_image_masks is not None:
            result["condition_image_mask"] = condition_image_masks
        if condition_state is not None:
            result["condition_state"] = condition_state
        if condition_tokenized_prompt is not None:
            result["condition_tokenized_prompt"] = condition_tokenized_prompt
        if condition_tokenized_prompt_mask is not None:
            result["condition_tokenized_prompt_mask"] = condition_tokenized_prompt_mask
        if reference_actions is not None:
            result["reference_actions"] = reference_actions
        if reference_action_mask is not None:
            result["reference_action_mask"] = reference_action_mask
        if (
            contrastive_meta_area_poses is not None
            or contrastive_meta_area_types is not None
            or contrastive_meta_area_masks is not None
        ):
            pose_key = (
                "pose12d"
                if getattr(contrastive_meta_area_poses, "shape", ())
                and contrastive_meta_area_poses.shape[-1] == 12
                else "pose6d"
            )
            result["contrastive_meta_areas"] = {
                pose_key: contrastive_meta_area_poses,
                "type": contrastive_meta_area_types,
                "mask": contrastive_meta_area_masks,
            }
            if contrastive_meta_area_dim_masks is not None:
                result["contrastive_meta_areas"]["dim_mask12"] = contrastive_meta_area_dim_masks
        if contrastive_reference_actions is not None:
            result["contrastive_reference_actions"] = contrastive_reference_actions
        if contrastive_reference_action_mask is not None:
            result["contrastive_reference_action_mask"] = contrastive_reference_action_mask
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def _preprocess_image_dict(
    images: dict[str, at.Float[ArrayT, "*b h w c"]],
    image_masks: dict[str, at.Bool[ArrayT, "*b"]],
    rng: at.KeyArrayLike | None,
    *,
    train: bool,
    image_keys: Sequence[str],
    image_resolution: tuple[int, int],
    batch_shape: tuple[int, ...],
    log_prefix: str,
) -> tuple[dict[str, at.Float[ArrayT, "*b h w c"]], dict[str, at.Bool[ArrayT, "*b"]]]:
    if not set(image_keys).issubset(images):
        raise ValueError(f"{log_prefix} dict missing keys: expected {image_keys}, got {list(images)}")

    out_images = {}
    for image_index, key in enumerate(image_keys):
        image = images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing {log_prefix} {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            image_rng = None if rng is None else jax.random.fold_in(rng, image_index)
            sub_rngs = jax.random.split(image_rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    out_masks = {}
    for key in out_images:
        if key not in image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(image_masks[key])
    return out_images, out_masks


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    batch_shape = observation.state.shape[:-1]
    out_images, out_masks = _preprocess_image_dict(
        observation.images,
        observation.image_masks,
        rng,
        train=train,
        image_keys=image_keys,
        image_resolution=image_resolution,
        batch_shape=batch_shape,
        log_prefix="image",
    )
    condition_images = None
    condition_image_masks = None
    if observation.condition_images is not None:
        condition_masks_in = observation.condition_image_masks or {}
        condition_images, condition_image_masks = _preprocess_image_dict(
            observation.condition_images,
            condition_masks_in,
            None if rng is None else jax.random.fold_in(rng, 1729),
            train=train,
            image_keys=image_keys,
            image_resolution=image_resolution,
            batch_shape=batch_shape,
            log_prefix="condition_image",
        )

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        condition_images=condition_images,
        condition_image_masks=condition_image_masks,
        condition_state=observation.condition_state,
        condition_tokenized_prompt=observation.condition_tokenized_prompt,
        condition_tokenized_prompt_mask=observation.condition_tokenized_prompt_mask,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        meta_area_poses=observation.meta_area_poses,
        meta_area_dim_masks=observation.meta_area_dim_masks,
        meta_area_types=observation.meta_area_types,
        meta_area_masks=observation.meta_area_masks,
        execution_meta_area_poses=observation.execution_meta_area_poses,
        execution_meta_area_dim_masks=observation.execution_meta_area_dim_masks,
        execution_meta_area_types=observation.execution_meta_area_types,
        execution_meta_area_masks=observation.execution_meta_area_masks,
        meta_action_target_poses=observation.meta_action_target_poses,
        meta_action_target_dim_masks=observation.meta_action_target_dim_masks,
        meta_action_target_masks=observation.meta_action_target_masks,
        meta_control_alpha=observation.meta_control_alpha,
        reference_actions=observation.reference_actions,
        reference_action_mask=observation.reference_action_mask,
        contrastive_meta_area_poses=observation.contrastive_meta_area_poses,
        contrastive_meta_area_dim_masks=observation.contrastive_meta_area_dim_masks,
        contrastive_meta_area_types=observation.contrastive_meta_area_types,
        contrastive_meta_area_masks=observation.contrastive_meta_area_masks,
        contrastive_reference_actions=observation.contrastive_reference_actions,
        contrastive_reference_action_mask=observation.contrastive_reference_action_mask,
        meta_imagination_alpha=observation.meta_imagination_alpha,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
