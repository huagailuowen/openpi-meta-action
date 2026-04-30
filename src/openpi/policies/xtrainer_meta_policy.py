import dataclasses

import einops
import numpy as np

from openpi import transforms

_DEFAULT_MAX_META_AREAS = 3
_DEFAULT_META_TYPE_LINE = 0


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_first(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Expected one of {keys}, got {tuple(data)}")


@dataclasses.dataclass(frozen=True)
class XTrainerMetaInputs(transforms.DataTransformFn):
    """Map xtrainer fields into the generic OpenPI observation format with optional meta-area inputs."""

    max_meta_areas: int = _DEFAULT_MAX_META_AREAS
    derive_meta_from_state_if_missing: bool = True
    zero_meta_state_slice: bool = True

    def __call__(self, data: dict) -> dict:
        images = _get_first(data, "images", "image")
        top_image = _parse_image(_get_first(images, "cam_high", "top", "base_0_rgb"))
        left_wrist_image = _parse_image(_get_first(images, "cam_left_wrist", "left_wrist", "left_wrist_0_rgb"))
        right_wrist_image = _parse_image(_get_first(images, "cam_right_wrist", "right_wrist", "right_wrist_0_rgb"))
        raw_state = np.asarray(_get_first(data, "state", "observation.state"), dtype=np.float32)
        state = raw_state.copy()

        meta_area_poses = np.zeros((self.max_meta_areas, 6), dtype=np.float32)
        meta_area_types = np.full((self.max_meta_areas,), _DEFAULT_META_TYPE_LINE, dtype=np.int32)
        meta_area_masks = np.zeros((self.max_meta_areas,), dtype=bool)

        meta_areas = data.get("meta_areas")
        if meta_areas is not None:
            poses = np.asarray(meta_areas["pose6d"], dtype=np.float32)
            types = np.asarray(meta_areas["type"], dtype=np.int32)
            masks = np.asarray(meta_areas["mask"], dtype=bool)
            count = min(self.max_meta_areas, poses.shape[0])
            meta_area_poses[:count] = poses[:count]
            meta_area_types[:count] = types[:count]
            meta_area_masks[:count] = masks[:count]
        elif self.derive_meta_from_state_if_missing and raw_state.shape[-1] >= 20:
            meta_area_poses[0] = raw_state[14:20]
            meta_area_types[0] = _DEFAULT_META_TYPE_LINE
            meta_area_masks[0] = True

        if self.zero_meta_state_slice and state.shape[-1] >= 20:
            state[14:20] = 0.0

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "meta_areas": {
                "pose6d": meta_area_poses,
                "type": meta_area_types,
                "mask": meta_area_masks,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        elif "action" in data:
            inputs["actions"] = np.asarray(data["action"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XTrainerRaw14Inputs(transforms.DataTransformFn):
    """Joints-only baseline: take the first 14 dims of state and actions and zero-pad
    them up to the model's full ``action_dim`` so the Pi0 architecture stays unchanged.

    No meta information is read or injected. Use with ``meta_model=False``.
    """

    model_action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        images = _get_first(data, "images", "image")
        top_image = _parse_image(_get_first(images, "cam_high", "top", "base_0_rgb"))
        left_wrist_image = _parse_image(_get_first(images, "cam_left_wrist", "left_wrist", "left_wrist_0_rgb"))
        right_wrist_image = _parse_image(_get_first(images, "cam_right_wrist", "right_wrist", "right_wrist_0_rgb"))

        raw_state = np.asarray(_get_first(data, "state", "observation.state"), dtype=np.float32)
        state = np.zeros((self.model_action_dim,), dtype=np.float32)
        state[:14] = raw_state[:14]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data or "action" in data:
            raw_actions = np.asarray(_get_first(data, "actions", "action"), dtype=np.float32)
            actions = np.zeros((raw_actions.shape[0], self.model_action_dim), dtype=np.float32)
            actions[:, :14] = raw_actions[:, :14]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XTrainerMetaOutputs(transforms.DataTransformFn):
    """Keep the packed action output and preserve structured meta side outputs when present."""

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        outputs = {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}
        if "meta_actions" in data:
            outputs["meta_actions"] = np.asarray(data["meta_actions"], dtype=np.float32)
        return outputs
