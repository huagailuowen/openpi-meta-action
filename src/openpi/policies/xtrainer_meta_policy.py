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


def _structured_image_inputs(images: dict) -> dict[str, np.ndarray]:
    return {
        "base_0_rgb": _parse_image(_get_first(images, "cam_high", "top", "base_0_rgb")),
        "left_wrist_0_rgb": _parse_image(_get_first(images, "cam_left_wrist", "left_wrist", "left_wrist_0_rgb")),
        "right_wrist_0_rgb": _parse_image(_get_first(images, "cam_right_wrist", "right_wrist", "right_wrist_0_rgb")),
    }


@dataclasses.dataclass(frozen=True)
class XTrainerMetaInputs(transforms.DataTransformFn):
    """Map xtrainer fields into the generic OpenPI observation format with optional meta-area inputs."""

    max_meta_areas: int = _DEFAULT_MAX_META_AREAS
    derive_meta_from_state_if_missing: bool = True
    zero_meta_state_slice: bool = True

    def __call__(self, data: dict) -> dict:
        image_inputs = _structured_image_inputs(_get_first(data, "images", "image"))
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
            "image": image_inputs,
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
        for key in ("execution_meta_areas", "meta_control", "reference_actions", "reference_action_mask"):
            if key in data:
                inputs[key] = data[key]
        for key in ("tool_instance_hash", "source_type_id", "episode_index"):
            if key in data:
                inputs[key] = data[key]
        inputs.setdefault("tool_instance_hash", np.asarray([0], dtype=np.int32))
        inputs.setdefault("source_type_id", np.asarray([0], dtype=np.int32))

        return inputs


@dataclasses.dataclass(frozen=True)
class XTrainerStructuredMetaInputs(transforms.DataTransformFn):
    """Structured-meta input path.

    This class is intentionally separate from ``XTrainerMetaInputs``. It consumes
    explicit LeRobot fields under ``meta_areas`` and ``meta_action_targets`` and
    forwards target fields explicitly. It supports both the old 6D path and the
    new 12D path. Only the old 6D path may mirror targets into action[14:20];
    12D targets never occupy action/state slices because action[20:26] is camera.
    """

    max_meta_areas: int = _DEFAULT_MAX_META_AREAS
    meta_area_pose_dim: int = 6
    require_structured_meta: bool = True
    zero_meta_state_slice: bool = True
    fill_action_meta_slice_from_targets: bool = True

    def __call__(self, data: dict) -> dict:
        image_inputs = _structured_image_inputs(_get_first(data, "images", "image"))
        raw_state = np.asarray(_get_first(data, "state", "observation.state"), dtype=np.float32)
        state = raw_state.copy()

        meta_dim = int(self.meta_area_pose_dim)
        meta_key = "pose12d" if meta_dim == 12 else "pose6d"
        meta_area_poses = np.zeros((self.max_meta_areas, meta_dim), dtype=np.float32)
        meta_area_dim_masks = np.zeros((self.max_meta_areas, meta_dim), dtype=bool)
        meta_area_types = np.full((self.max_meta_areas,), _DEFAULT_META_TYPE_LINE, dtype=np.int32)
        meta_area_masks = np.zeros((self.max_meta_areas,), dtype=bool)

        meta_areas = data.get("meta_areas")
        if meta_areas is None:
            if self.require_structured_meta:
                raise KeyError("Structured meta input requires meta_areas.pose6d or pose12d, plus type/mask.")
            if meta_dim == 6 and raw_state.shape[-1] >= 20:
                meta_area_poses[0] = raw_state[14 : 14 + meta_dim]
                meta_area_dim_masks[0] = True
                meta_area_masks[0] = True
        else:
            if "pose12d" in meta_areas:
                meta_key = "pose12d"
            elif "pose6d" in meta_areas:
                meta_key = "pose6d"
            else:
                raise KeyError("Structured meta input requires meta_areas.pose6d or meta_areas.pose12d.")
            poses = np.asarray(meta_areas[meta_key], dtype=np.float32)
            if poses.shape[-1] != meta_dim:
                raise ValueError(f"Expected {meta_key} last dim {meta_dim}, got {poses.shape}")
            types = np.asarray(meta_areas["type"], dtype=np.int32)
            masks = np.asarray(meta_areas["mask"], dtype=bool)
            if meta_key == "pose12d":
                raw_dim_masks = meta_areas.get("dim_mask12")
                dim_masks = (
                    np.asarray(raw_dim_masks, dtype=bool)
                    if raw_dim_masks is not None
                    else np.ones(poses.shape, dtype=bool)
                )
            else:
                dim_masks = np.ones(poses.shape, dtype=bool)
            if types.ndim > 1 and types.shape[-1] == 1:
                types = np.squeeze(types, axis=-1)
            if masks.ndim > 1 and masks.shape[-1] == 1:
                masks = np.squeeze(masks, axis=-1)
            count = min(self.max_meta_areas, poses.shape[0])
            meta_area_poses[:count] = poses[:count]
            meta_area_dim_masks[:count] = dim_masks[:count]
            meta_area_types[:count] = types[:count]
            meta_area_masks[:count] = masks[:count]

        if self.zero_meta_state_slice and state.shape[-1] >= 20:
            state[14:20] = 0.0

        meta_areas_out = {
            meta_key: meta_area_poses,
            "type": meta_area_types,
            "mask": meta_area_masks,
        }
        if meta_key == "pose12d":
            meta_areas_out["dim_mask12"] = meta_area_dim_masks
        inputs = {
            "state": state,
            "image": image_inputs,
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "meta_areas": meta_areas_out,
        }

        meta_targets = data.get("meta_action_targets")
        if meta_targets is not None:
            target_key = "pose12d" if meta_dim == 12 and "pose12d" in meta_targets else "pose6d"
            target_poses = np.asarray(meta_targets[target_key], dtype=np.float32)
            if target_poses.shape[-1] != meta_dim:
                raise ValueError(f"Expected meta_action_targets.{target_key} last dim {meta_dim}, got {target_poses.shape}")

            if target_poses.ndim == 2:
                target_poses = target_poses[:, None, :] if target_poses.shape[0] != self.max_meta_areas else target_poses[None, :, :]
            elif target_poses.ndim != 3:
                raise ValueError(f"Expected meta_action_targets.{target_key} with 2 or 3 dims, got {target_poses.shape}")

            horizon = target_poses.shape[0]
            padded_target_poses = np.zeros((horizon, self.max_meta_areas, meta_dim), dtype=np.float32)
            count = min(self.max_meta_areas, target_poses.shape[1])
            padded_target_poses[:, :count, :] = target_poses[:, :count, :]

            raw_target_masks = meta_targets.get("mask")
            if raw_target_masks is None:
                target_masks = np.ones(target_poses.shape[:2], dtype=bool)
            else:
                target_masks = np.asarray(raw_target_masks, dtype=bool)
                if target_masks.ndim == 3 and target_masks.shape[-1] == 1:
                    target_masks = np.squeeze(target_masks, axis=-1)
                if target_masks.ndim == 1:
                    target_masks = target_masks[:, None] if target_masks.shape[0] == horizon and count == 1 else target_masks[None, :]
                if target_masks.ndim != 2:
                    raise ValueError(f"Expected meta_action_targets.mask with 1, 2 or [H,M,1] dims, got {target_masks.shape}")
                if target_masks.shape[0] == 1 and horizon > 1:
                    target_masks = np.broadcast_to(target_masks, (horizon, target_masks.shape[1]))
                if target_masks.shape[0] != horizon:
                    raise ValueError(f"meta_action_targets.mask horizon mismatch: {target_masks.shape[0]} vs {horizon}")
            padded_target_masks = np.zeros((horizon, self.max_meta_areas), dtype=bool)
            padded_target_masks[:, : min(self.max_meta_areas, target_masks.shape[1])] = target_masks[:, : self.max_meta_areas]

            targets_out = {
                target_key: padded_target_poses,
                "mask": padded_target_masks,
            }
            if target_key == "pose12d":
                raw_dim_masks = meta_targets.get("dim_mask12")
                target_dim_masks = (
                    np.asarray(raw_dim_masks, dtype=bool)
                    if raw_dim_masks is not None
                    else np.ones(target_poses.shape, dtype=bool)
                )
                if target_dim_masks.ndim == 2:
                    target_dim_masks = (
                        target_dim_masks[:, None, :]
                        if target_dim_masks.shape[0] == horizon and count == 1
                        else target_dim_masks[None, :, :]
                    )
                if target_dim_masks.shape[0] == 1 and horizon > 1:
                    target_dim_masks = np.broadcast_to(target_dim_masks, (horizon, target_dim_masks.shape[1], meta_dim))
                if target_dim_masks.shape[0] != horizon:
                    raise ValueError(f"meta_action_targets.dim_mask12 horizon mismatch: {target_dim_masks.shape[0]} vs {horizon}")
                padded_dim_masks = np.zeros((horizon, self.max_meta_areas, meta_dim), dtype=bool)
                padded_dim_masks[:, : min(self.max_meta_areas, target_dim_masks.shape[1]), :] = target_dim_masks[
                    :, : self.max_meta_areas, :
                ]
                targets_out["dim_mask12"] = padded_dim_masks
            inputs["meta_action_targets"] = targets_out

        if "actions" in data or "action" in data:
            actions = np.asarray(_get_first(data, "actions", "action"), dtype=np.float32).copy()
            if meta_dim == 12 and actions.shape[-1] >= 20:
                actions[..., 14:20] = 0.0
            if self.fill_action_meta_slice_from_targets and meta_targets is not None and meta_dim == 6:
                target_poses = np.asarray(meta_targets["pose6d"], dtype=np.float32)
                if target_poses.ndim == 3:
                    slot0_targets = target_poses[:, 0, :]
                elif target_poses.ndim == 2:
                    slot0_targets = target_poses
                else:
                    raise ValueError(f"Expected meta_action_targets.pose6d with 2 or 3 dims, got {target_poses.shape}")
                target_dim = int(slot0_targets.shape[-1])
                if actions.shape[-1] >= 14 + target_dim:
                    actions[..., 14 : 14 + target_dim] = slot0_targets
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "condition_images" in data:
            inputs["condition_image"] = _structured_image_inputs(data["condition_images"])
            inputs["condition_image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }
        if "condition_state" in data:
            condition_state = np.asarray(data["condition_state"], dtype=np.float32).copy()
            if self.zero_meta_state_slice and condition_state.shape[-1] >= 20:
                condition_state[14:20] = 0.0
            inputs["condition_state"] = condition_state

        for key in ("execution_meta_areas", "meta_control", "reference_actions", "reference_action_mask"):
            if key in data:
                inputs[key] = data[key]
        for key in ("tool_instance_hash", "source_type_id", "episode_index"):
            if key in data:
                inputs[key] = data[key]
        inputs.setdefault("tool_instance_hash", np.asarray([0], dtype=np.int32))
        inputs.setdefault("source_type_id", np.asarray([0], dtype=np.int32))

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
    zero_unused_meta_slice: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32).copy()
        if self.zero_unused_meta_slice and actions.shape[-1] >= 20:
            actions[..., 14:20] = 0.0
        outputs = {"actions": actions}
        if "meta_actions" in data:
            outputs["meta_actions"] = np.asarray(data["meta_actions"], dtype=np.float32)
        return outputs
