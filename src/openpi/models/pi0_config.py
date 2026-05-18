import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    meta_model: bool = False
    max_meta_areas: int = 3
    meta_area_type_vocab_size: int = 3
    meta_area_pose_dim: int = 6
    num_meta_special_tokens: int = 4
    # Weight applied to the backbone flow-matching loss term.
    action_loss_weight: float = 1.0
    # Weight applied to the meta MSE loss term. Keep small (0.1) while the meta head
    # is randomly initialised; set to 1.0 only after the head has warmed up.
    meta_loss_weight: float = 0.1
    meta_action_start_dim: int = 14
    meta_action_dim: int = 6
    # Legacy 6D meta targets are mirrored into action[14:20]. Structured 12D
    # targets must stay outside the action vector because action[20:26] is camera.
    meta_actions_in_action_slice: bool = True
    meta_dropout_prob: float = 0.0
    # Optional continuous control token for structured meta. When enabled, the
    # model receives observation.meta_control.alpha and meta loss can be scaled
    # per sample by alpha ** meta_loss_alpha_power.
    use_meta_control_alpha: bool = False
    meta_loss_alpha_power: float = 0.0
    # Beta latent path. This keeps the old structured meta model intact and
    # switches creation to Pi0MetaBeta only for explicit beta configs.
    meta_beta_model: bool = False
    num_meta_latent_tokens: int = 4
    reference_action_group_size: int = 5
    # When True, stop gradients from the meta loss from flowing back into the shared
    # backbone (prefix_out / suffix_out). The meta head still receives full gradients
    # through its own new parameters. Recommended while loading from a pre-trained
    # backbone so the pre-trained weights are not corrupted by the random meta head.
    meta_stop_backbone_grad: bool = True

    # When True, freeze the VLM backbone (PaliGemma image encoder + first LLM expert)
    # during training. The action expert and meta head still receive gradients.
    freeze_vlm_backbone: bool = False
    # Fine-grained freezing of the action flow-matching head. The VLM backbone and
    # meta head still receive gradients regardless of these flags.
    # - freeze_action_expert_llm: freezes the action expert LLM (and time MLP, which
    #   conditions the expert via adaRMSNorm).
    # - freeze_action_in_proj: freezes the noisy-action input projection.
    # - freeze_action_out_proj: freezes the velocity output projection.
    freeze_action_expert_llm: bool = False
    freeze_action_in_proj: bool = False
    freeze_action_out_proj: bool = False

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        if self.meta_model:
            if not self.pi05:
                raise ValueError("The meta model path is only implemented for PI0.5.")
            if self.meta_beta_model:
                from openpi.models.pi0_meta_beta import Pi0MetaBeta

                return Pi0MetaBeta(self, rngs=nnx.Rngs(rng))
            from openpi.models.pi0_meta import Pi0Meta

            return Pi0Meta(self, rngs=nnx.Rngs(rng))
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                condition_images=(
                    {
                        "base_0_rgb": image_spec,
                        "left_wrist_0_rgb": image_spec,
                        "right_wrist_0_rgb": image_spec,
                    }
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                condition_image_masks=(
                    {
                        "base_0_rgb": image_mask_spec,
                        "left_wrist_0_rgb": image_mask_spec,
                        "right_wrist_0_rgb": image_mask_spec,
                    }
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                condition_state=(
                    jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                meta_area_poses=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas, self.meta_area_pose_dim], jnp.float32)
                    if self.meta_model
                    else None
                ),
                meta_area_dim_masks=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas, self.meta_area_pose_dim], jnp.bool_)
                    if self.meta_model and self.meta_area_pose_dim == 12
                    else None
                ),
                meta_area_types=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas], jnp.int32) if self.meta_model else None
                ),
                meta_area_masks=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas], jnp.bool_) if self.meta_model else None
                ),
                execution_meta_area_poses=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas, self.meta_area_pose_dim], jnp.float32)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                execution_meta_area_dim_masks=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas, self.meta_area_pose_dim], jnp.bool_)
                    if self.meta_model and self.meta_beta_model and self.meta_area_pose_dim == 12
                    else None
                ),
                execution_meta_area_types=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas], jnp.int32)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                execution_meta_area_masks=(
                    jax.ShapeDtypeStruct([batch_size, self.max_meta_areas], jnp.bool_)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                meta_action_target_poses=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.action_horizon, self.max_meta_areas, self.meta_action_dim],
                        jnp.float32,
                    )
                    if self.meta_model
                    else None
                ),
                meta_action_target_dim_masks=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.action_horizon, self.max_meta_areas, self.meta_action_dim],
                        jnp.bool_,
                    )
                    if self.meta_model and self.meta_action_dim == 12
                    else None
                ),
                meta_action_target_masks=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.max_meta_areas], jnp.bool_)
                    if self.meta_model
                    else None
                ),
                meta_control_alpha=(
                    jax.ShapeDtypeStruct([batch_size], jnp.float32)
                    if self.meta_model and self.use_meta_control_alpha
                    else None
                ),
                reference_actions=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                reference_action_mask=(
                    jax.ShapeDtypeStruct([batch_size], jnp.bool_)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
                meta_imagination_alpha=(
                    jax.ShapeDtypeStruct([batch_size], jnp.float32)
                    if self.meta_model and self.meta_beta_model
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        lora_filter = nnx.All(*filters) if filters else None

        extra_filters = []
        if self.freeze_vlm_backbone:
            # PaliGemma image encoder + first LLM expert (the action expert lives in the
            # second LLM module whose params include `_1` in their path).
            extra_filters.append(
                nnx.Any(
                    nnx_utils.PathRegex(".*PaliGemma/img.*"),
                    nnx.All(
                        nnx_utils.PathRegex(".*PaliGemma/llm.*"),
                        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
                    ),
                )
            )
        if self.freeze_action_expert_llm:
            extra_filters.append(
                nnx.Any(
                    nnx_utils.PathRegex(".*llm.*_1.*"),
                    nnx_utils.PathRegex(".*time_mlp.*"),
                )
            )
        if self.freeze_action_in_proj:
            extra_filters.append(nnx_utils.PathRegex(".*action_in_proj.*"))
        if self.freeze_action_out_proj:
            extra_filters.append(nnx_utils.PathRegex(".*action_out_proj.*"))

        if extra_filters:
            combined = nnx.Any(*extra_filters)
            if lora_filter is not None:
                return nnx.Any(lora_filter, combined)
            return combined

        if lora_filter is None:
            return nnx.Nothing
        return lora_filter
