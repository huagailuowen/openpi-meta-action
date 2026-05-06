"""Offline retarget augmentation for x-trainer geometry meta chunks.

This module operates on the canonical chunk format produced after the x-trainer
meta input adapters:

  state:                 [32]
  meta_areas.pose6d:     [M, 6]
  meta_areas.type:       [M]
  meta_areas.mask:       [M]
  actions:               [H, 32]

The retarget operation perturbs the wrist-local geometry meta area, then solves
right-arm qpos so the perturbed meta area follows the original meta-action
trajectory. It is designed for offline cache generation. Training should sample
from the cache rather than running IK in every dataloader worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
from pathlib import Path
import sys
from typing import Any

import numpy as np


META_AREA_TYPE_TO_ID = {
    "line": 0,
    "point": 1,
    "surface": 2,
}
META_AREA_ID_TO_TYPE = {value: key for key, value in META_AREA_TYPE_TO_ID.items()}

STATE_META_SLICE = slice(14, 20)
STATE_CAMERA_INPUT_SLICE = slice(20, 26)
ACTION_META_SLICE = slice(14, 20)
ACTION_CAMERA_INPUT_SLICE = slice(20, 26)
RIGHT_ARM_QPOS_SLICE = slice(7, 13)


@dataclass(frozen=True)
class MetaRetargetGeneratorConfig:
    """Configuration for one canonical chunk retarget attempt."""

    position_noise_max_m: float = 0.04
    direction_noise_max_deg: float = 25.0
    approach_joint_step_rad: float = 0.04
    max_approach_steps: int | None = None
    ik_max_iters: int = 80
    ik_tolerance: float = 1e-3
    ik_damping: float = 1e-3
    ik_step_scale: float = 1.0
    ik_position_weight: float = 1.0
    ik_direction_weight: float = 0.35
    ik_finite_diff_eps: float = 1e-4
    ik_max_joint_step_rad: float = 0.08
    ik_backend: str = "numpy"
    accept_max_position_error_m: float = 0.015
    accept_max_direction_error_rad: float = 0.25
    accept_max_step_joint_delta_rad: float = 0.35
    recompute_action_camera_pose: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaRetargetDiagnostics:
    accepted: bool
    area_type: str
    horizon: int
    approach_steps: int
    ik_nonconverged: int
    max_position_error_m: float
    max_direction_error_rad: float
    max_step_joint_delta_rad: float
    reason: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaRetargetResult:
    state: np.ndarray
    actions: np.ndarray
    meta_area_pose6d: np.ndarray
    meta_area_type: np.ndarray
    meta_area_mask: np.ndarray
    diagnostics: MetaRetargetDiagnostics


@dataclass(frozen=True)
class _DynamicApproachPlan:
    steps: int
    aligned_qpos_real: np.ndarray
    converged: bool
    final_error: float


def canonicalize_repacked_xtrainer_chunk(
    data: dict[str, Any],
    *,
    max_meta_areas: int,
    derive_meta_from_state_if_missing: bool = True,
    zero_meta_state_slice: bool = True,
    fill_action_meta_slice_from_targets: bool = True,
) -> dict[str, Any]:
    """Extract canonical meta fields without touching images.

    This mirrors the field semantics of ``XTrainerMetaInputs`` and
    ``XTrainerStructuredMetaInputs`` but intentionally avoids image parsing so
    offline cache building does not spend time resizing/decoding data that the IK
    generator does not need.
    """

    raw_state = np.asarray(_get_first(data, "state", "observation.state"), dtype=np.float32)
    state = raw_state.copy()
    actions = np.asarray(_get_first(data, "actions", "action"), dtype=np.float32).copy()

    meta_area_poses = np.zeros((max_meta_areas, 6), dtype=np.float32)
    meta_area_types = np.full((max_meta_areas,), META_AREA_TYPE_TO_ID["line"], dtype=np.int32)
    meta_area_masks = np.zeros((max_meta_areas,), dtype=bool)

    meta_areas = data.get("meta_areas")
    if meta_areas is not None:
        poses = np.asarray(meta_areas["pose6d"], dtype=np.float32)
        types = np.asarray(meta_areas["type"], dtype=np.int32)
        masks = np.asarray(meta_areas["mask"], dtype=bool)
        if types.ndim > 1 and types.shape[-1] == 1:
            types = np.squeeze(types, axis=-1)
        if masks.ndim > 1 and masks.shape[-1] == 1:
            masks = np.squeeze(masks, axis=-1)
        count = min(max_meta_areas, poses.shape[0])
        meta_area_poses[:count] = poses[:count]
        meta_area_types[:count] = types[:count]
        meta_area_masks[:count] = masks[:count]
    elif derive_meta_from_state_if_missing and raw_state.shape[-1] >= STATE_META_SLICE.stop:
        meta_area_poses[0] = raw_state[STATE_META_SLICE]
        meta_area_types[0] = META_AREA_TYPE_TO_ID["line"]
        meta_area_masks[0] = True

    meta_targets = data.get("meta_action_targets")
    if fill_action_meta_slice_from_targets and meta_targets is not None and actions.shape[-1] >= ACTION_META_SLICE.stop:
        target_pose6d = np.asarray(meta_targets["pose6d"], dtype=np.float32)
        if target_pose6d.ndim == 3:
            actions[..., ACTION_META_SLICE] = target_pose6d[:, 0, :]
        elif target_pose6d.ndim == 2:
            actions[..., ACTION_META_SLICE] = target_pose6d
        else:
            raise ValueError(f"Expected meta_action_targets.pose6d with 2 or 3 dims, got {target_pose6d.shape}")

    if zero_meta_state_slice and state.shape[-1] >= STATE_META_SLICE.stop:
        state[STATE_META_SLICE] = 0.0

    return {
        "state": state,
        "actions": actions,
        "meta_areas": {
            "pose6d": meta_area_poses,
            "type": meta_area_types,
            "mask": meta_area_masks,
        },
    }


def generate_retargeted_chunk(
    data: dict[str, Any],
    *,
    rng: np.random.Generator,
    config: MetaRetargetGeneratorConfig,
) -> MetaRetargetResult | None:
    """Generate one retargeted chunk, returning ``None`` if the sample is not usable."""

    state = np.asarray(data["state"], dtype=np.float32).copy()
    actions = np.asarray(data["actions"], dtype=np.float32).copy()
    meta_areas = data["meta_areas"]
    meta_area_pose6d = np.asarray(meta_areas["pose6d"], dtype=np.float32).copy()
    meta_area_type = np.asarray(meta_areas["type"], dtype=np.int32).copy()
    meta_area_mask = np.asarray(meta_areas["mask"], dtype=bool).copy()

    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[-1] < ACTION_META_SLICE.stop:
        return None
    if state.shape[-1] < 14 or actions.shape[-1] < 14:
        return None
    if meta_area_pose6d.shape[0] == 0 or not bool(meta_area_mask[0]):
        return None

    horizon = int(actions.shape[0])
    area_type = META_AREA_ID_TO_TYPE.get(int(meta_area_type[0]), "line")
    helpers = _load_xtrainer_helpers()

    state_qpos = state[:14].copy()
    T_state_wrist = np.asarray(helpers["fk"](state_qpos, "right_wrist"), dtype=np.float32)
    old_input_pose = meta_area_pose6d[0].copy()
    T_wrist_meta_old = _base_meta_pose_to_wrist_transform(area_type, old_input_pose, T_state_wrist)
    T_wrist_meta_new = _perturb_wrist_meta_transform(area_type, T_wrist_meta_old, rng, config)

    new_input_pose = _encode_meta_from_transform(area_type, T_state_wrist @ T_wrist_meta_new)
    old_target_meta = actions[:, ACTION_META_SLICE].copy()
    seed_sim = helpers["real_to_sim_arm"](state_qpos[RIGHT_ARM_QPOS_SLICE], "right")
    approach_plan = _compute_dynamic_approach_plan(
        area_type=area_type,
        state_qpos=state_qpos,
        first_target_meta=old_target_meta[0],
        T_wrist_meta=T_wrist_meta_new,
        initial_qpos_sim=seed_sim,
        config=config,
        helpers=helpers,
        horizon=horizon,
    )
    approach_steps = approach_plan.steps

    out_actions = actions.copy()
    previous_real_qpos = state_qpos.copy()

    ik_nonconverged = 0 if approach_plan.converged else 1
    position_errors: list[float] = []
    direction_errors: list[float] = []
    joint_step_deltas: list[float] = []
    T_wrist_cam = _camera_pose_to_transform(state[STATE_CAMERA_INPUT_SLICE]) if state.shape[-1] >= 26 else None
    align_pos_err, align_dir_err = _meta_tracking_errors(
        area_type=area_type,
        qpos_real=approach_plan.aligned_qpos_real,
        T_wrist_meta=T_wrist_meta_new,
        target_meta_pose=old_target_meta[0],
        fk_fn=helpers["fk"],
    )
    position_errors.append(align_pos_err)
    direction_errors.append(align_dir_err)

    jax_follow: dict[str, Any] | None = None
    if config.ik_backend == "jax" and approach_steps < horizon:
        follow_seed_sim = (
            helpers["real_to_sim_arm"](approach_plan.aligned_qpos_real[RIGHT_ARM_QPOS_SLICE], "right")
            if approach_steps > 0
            else seed_sim
        )
        jax_follow = _solve_right_arm_meta_ik_sequence_jax(
            target_meta_poses=old_target_meta,
            T_wrist_meta=T_wrist_meta_new,
            area_type=area_type,
            initial_qpos_sim=follow_seed_sim,
            config=config,
        )

    for step in range(horizon):
        if step < approach_steps:
            alpha = float(step + 1) / float(max(approach_steps, 1))
            target_meta = _interpolate_meta_pose(area_type, new_input_pose, old_target_meta[0], alpha)
            out_action = actions[0].copy()
            out_action[:14] = state_qpos
            out_action[RIGHT_ARM_QPOS_SLICE] = (
                (1.0 - alpha) * state_qpos[RIGHT_ARM_QPOS_SLICE]
                + alpha * approach_plan.aligned_qpos_real[RIGHT_ARM_QPOS_SLICE]
            ).astype(np.float32)
            out_action[ACTION_META_SLICE] = target_meta
            seed_sim = helpers["real_to_sim_arm"](out_action[RIGHT_ARM_QPOS_SLICE], "right")
        else:
            source_idx = min(step - approach_steps, horizon - 1)
            out_action = actions[source_idx].copy()
            target_meta = old_target_meta[source_idx]

            if jax_follow is None:
                ik_result = _solve_right_arm_meta_ik(
                    template_qpos_real=out_action[:14],
                    target_meta_pose=target_meta,
                    T_wrist_meta=T_wrist_meta_new,
                    area_type=area_type,
                    initial_qpos_sim=seed_sim,
                    config=config,
                    helpers=helpers,
                )
                solved_sim = _wrap_near_seed(ik_result["qpos_sim"], seed_sim)
                converged = bool(ik_result["converged"])
            else:
                solved_sim = np.asarray(jax_follow["qpos_sim"][source_idx], dtype=np.float32)
                converged = bool(jax_follow["converged"][source_idx])

            solved_real = _sim_to_real_right_arm_qpos(solved_sim, helpers)

            out_action[RIGHT_ARM_QPOS_SLICE] = solved_real
            out_action[ACTION_META_SLICE] = target_meta
            seed_sim = solved_sim
            if not converged:
                ik_nonconverged += 1

        if config.recompute_action_camera_pose and T_wrist_cam is not None and out_action.shape[-1] >= 26:
            T_action_wrist = np.asarray(helpers["fk"](out_action[:14], "right_wrist"), dtype=np.float32)
            out_action[ACTION_CAMERA_INPUT_SLICE] = _transform_to_camera_pose(T_action_wrist @ T_wrist_cam)

        out_actions[step] = out_action

        pos_err, dir_err = _meta_tracking_errors(
            area_type=area_type,
            qpos_real=out_action[:14],
            T_wrist_meta=T_wrist_meta_new,
            target_meta_pose=out_action[ACTION_META_SLICE],
            fk_fn=helpers["fk"],
        )
        position_errors.append(pos_err)
        direction_errors.append(dir_err)
        joint_step_deltas.append(float(np.max(np.abs(out_action[:14] - previous_real_qpos[:14]))))
        previous_real_qpos = out_action[:14].copy()

    meta_area_pose6d[0] = new_input_pose
    max_position_error = float(max(position_errors, default=np.inf))
    max_direction_error = float(max(direction_errors, default=0.0))
    max_joint_delta = float(max(joint_step_deltas, default=0.0))

    accepted = (
        max_position_error <= config.accept_max_position_error_m
        and max_direction_error <= config.accept_max_direction_error_rad
        and max_joint_delta <= config.accept_max_step_joint_delta_rad
    )
    reason = ""
    if not accepted:
        reason = (
            f"ik_nonconverged={ik_nonconverged}, "
            f"max_position_error_m={max_position_error:.6f}, "
            f"max_direction_error_rad={max_direction_error:.6f}, "
            f"max_step_joint_delta_rad={max_joint_delta:.6f}"
        )

    diagnostics = MetaRetargetDiagnostics(
        accepted=accepted,
        area_type=area_type,
        horizon=horizon,
        approach_steps=approach_steps,
        ik_nonconverged=ik_nonconverged,
        max_position_error_m=max_position_error,
        max_direction_error_rad=max_direction_error,
        max_step_joint_delta_rad=max_joint_delta,
        reason=reason,
    )
    return MetaRetargetResult(
        state=state.astype(np.float32),
        actions=out_actions.astype(np.float32),
        meta_area_pose6d=meta_area_pose6d.astype(np.float32),
        meta_area_type=meta_area_type.astype(np.int32),
        meta_area_mask=meta_area_mask.astype(bool),
        diagnostics=diagnostics,
    )


def generate_retargeted_chunks_batch(
    data_batch: list[dict[str, Any]],
    *,
    rngs: list[np.random.Generator],
    config: MetaRetargetGeneratorConfig,
) -> list[MetaRetargetResult | None]:
    """Generate a batch of retargeted chunks, using batched JAX IK when requested."""

    if len(data_batch) != len(rngs):
        raise ValueError(f"data_batch and rngs length mismatch: {len(data_batch)} != {len(rngs)}")
    if not data_batch:
        return []
    if config.ik_backend != "jax":
        return [generate_retargeted_chunk(data, rng=rng, config=config) for data, rng in zip(data_batch, rngs)]
    if config.approach_joint_step_rad <= 0:
        raise ValueError(f"approach_joint_step_rad must be positive, got {config.approach_joint_step_rad}")

    helpers = _load_xtrainer_helpers()
    prepared: list[dict[str, Any] | None] = [
        _prepare_retarget_chunk(data, rng=rng, config=config, helpers=helpers) for data, rng in zip(data_batch, rngs)
    ]
    results: list[MetaRetargetResult | None] = [None] * len(data_batch)

    for area_type in META_AREA_TYPE_TO_ID:
        group_indices = [
            idx
            for idx, item in enumerate(prepared)
            if item is not None and item["area_type"] == area_type
        ]
        if not group_indices:
            continue

        first_targets = np.stack([prepared[idx]["old_target_meta"][0] for idx in group_indices], axis=0)[:, None, :]
        wrist_meta = np.stack([prepared[idx]["T_wrist_meta_new"] for idx in group_indices], axis=0)
        initial_qpos = np.stack([prepared[idx]["initial_seed_sim"] for idx in group_indices], axis=0)
        alignment = _solve_right_arm_meta_ik_sequence_batch_jax(
            target_meta_poses=first_targets,
            T_wrist_meta=wrist_meta,
            area_type=area_type,
            initial_qpos_sim=initial_qpos,
            config=config,
        )

        aligned_real_right = np.stack(
            [_sim_to_real_right_arm_qpos(qpos, helpers) for qpos in alignment["qpos_sim"][:, 0, :]],
            axis=0,
        )
        approach_plans: list[_DynamicApproachPlan] = []
        follow_initial_qpos = []
        for group_offset, idx in enumerate(group_indices):
            item = prepared[idx]
            assert item is not None
            aligned_qpos_real = item["state_qpos"].copy()
            aligned_qpos_real[RIGHT_ARM_QPOS_SLICE] = aligned_real_right[group_offset]
            right_delta = aligned_real_right[group_offset] - item["state_qpos"][RIGHT_ARM_QPOS_SLICE]
            approach_steps = int(np.ceil(float(np.max(np.abs(right_delta))) / float(config.approach_joint_step_rad)))
            if config.max_approach_steps is not None:
                approach_steps = min(approach_steps, int(config.max_approach_steps))
            approach_steps = min(max(approach_steps, 0), item["horizon"])
            approach_plans.append(
                _DynamicApproachPlan(
                    steps=approach_steps,
                    aligned_qpos_real=aligned_qpos_real.astype(np.float32),
                    converged=bool(alignment["converged"][group_offset, 0]),
                    final_error=float(alignment["final_error"][group_offset, 0]),
                )
            )
            follow_initial_qpos.append(
                alignment["qpos_sim"][group_offset, 0, :]
                if approach_steps > 0
                else item["initial_seed_sim"]
            )

        follow_initial_qpos = np.stack(follow_initial_qpos, axis=0).astype(np.float32)
        follow_targets = np.stack([prepared[idx]["old_target_meta"] for idx in group_indices], axis=0)
        follow = _solve_right_arm_meta_ik_sequence_batch_jax(
            target_meta_poses=follow_targets,
            T_wrist_meta=wrist_meta,
            area_type=area_type,
            initial_qpos_sim=follow_initial_qpos,
            config=config,
        )

        for group_offset, idx in enumerate(group_indices):
            item = prepared[idx]
            assert item is not None
            results[idx] = _assemble_retarget_result(
                item,
                approach_plan=approach_plans[group_offset],
                follow_qpos_sim=np.asarray(follow["qpos_sim"][group_offset], dtype=np.float32),
                follow_converged=np.asarray(follow["converged"][group_offset], dtype=bool),
                config=config,
                helpers=helpers,
            )

    return results


def save_retarget_result(path: Path, result: MetaRetargetResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            state=result.state,
            actions=result.actions,
            meta_area_pose6d=result.meta_area_pose6d,
            meta_area_type=result.meta_area_type,
            meta_area_mask=result.meta_area_mask,
        )
    tmp_path.replace(path)


def load_retarget_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def _get_first(data: dict[str, Any], *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Expected one of {keys}, got {tuple(data)}")


def _load_xtrainer_helpers() -> dict[str, Any]:
    funcpoint_root = Path(__file__).resolve().parents[4]
    if str(funcpoint_root) not in sys.path:
        sys.path.insert(0, str(funcpoint_root))

    from src.calibration.xtrainer_fk import (  # type: ignore
        RIGHT_REAL_TO_SIM_SIGN,
        RIGHT_REAL_ZERO_ARM,
        real_to_sim_xtrainer_arm_qpos,
        xtrainer_forward_kinematics,
    )

    return {
        "right_real_to_sim_sign": RIGHT_REAL_TO_SIM_SIGN,
        "right_real_zero_arm": RIGHT_REAL_ZERO_ARM,
        "real_to_sim_arm": real_to_sim_xtrainer_arm_qpos,
        "fk": xtrainer_forward_kinematics,
    }


def _prepare_retarget_chunk(
    data: dict[str, Any],
    *,
    rng: np.random.Generator,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> dict[str, Any] | None:
    state = np.asarray(data["state"], dtype=np.float32).copy()
    actions = np.asarray(data["actions"], dtype=np.float32).copy()
    meta_areas = data["meta_areas"]
    meta_area_pose6d = np.asarray(meta_areas["pose6d"], dtype=np.float32).copy()
    meta_area_type = np.asarray(meta_areas["type"], dtype=np.int32).copy()
    meta_area_mask = np.asarray(meta_areas["mask"], dtype=bool).copy()

    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[-1] < ACTION_META_SLICE.stop:
        return None
    if state.shape[-1] < 14 or actions.shape[-1] < 14:
        return None
    if meta_area_pose6d.shape[0] == 0 or not bool(meta_area_mask[0]):
        return None

    horizon = int(actions.shape[0])
    area_type = META_AREA_ID_TO_TYPE.get(int(meta_area_type[0]), "line")
    state_qpos = state[:14].copy()
    T_state_wrist = np.asarray(helpers["fk"](state_qpos, "right_wrist"), dtype=np.float32)
    old_input_pose = meta_area_pose6d[0].copy()
    T_wrist_meta_old = _base_meta_pose_to_wrist_transform(area_type, old_input_pose, T_state_wrist)
    T_wrist_meta_new = _perturb_wrist_meta_transform(area_type, T_wrist_meta_old, rng, config)
    new_input_pose = _encode_meta_from_transform(area_type, T_state_wrist @ T_wrist_meta_new)

    return {
        "state": state,
        "actions": actions,
        "meta_area_pose6d": meta_area_pose6d,
        "meta_area_type": meta_area_type,
        "meta_area_mask": meta_area_mask,
        "horizon": horizon,
        "area_type": area_type,
        "state_qpos": state_qpos,
        "T_wrist_meta_new": T_wrist_meta_new,
        "new_input_pose": new_input_pose,
        "old_target_meta": actions[:, ACTION_META_SLICE].copy(),
        "initial_seed_sim": helpers["real_to_sim_arm"](state_qpos[RIGHT_ARM_QPOS_SLICE], "right"),
    }


def _assemble_retarget_result(
    item: dict[str, Any],
    *,
    approach_plan: _DynamicApproachPlan,
    follow_qpos_sim: np.ndarray,
    follow_converged: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> MetaRetargetResult:
    state = item["state"].copy()
    actions = item["actions"]
    meta_area_pose6d = item["meta_area_pose6d"].copy()
    meta_area_type = item["meta_area_type"].copy()
    meta_area_mask = item["meta_area_mask"].copy()
    horizon = int(item["horizon"])
    area_type = item["area_type"]
    state_qpos = item["state_qpos"]
    T_wrist_meta_new = item["T_wrist_meta_new"]
    new_input_pose = item["new_input_pose"]
    old_target_meta = item["old_target_meta"]

    approach_steps = approach_plan.steps
    out_actions = actions.copy()
    previous_real_qpos = state_qpos.copy()
    ik_nonconverged = 0 if approach_plan.converged else 1
    position_errors: list[float] = []
    direction_errors: list[float] = []
    joint_step_deltas: list[float] = []
    T_wrist_cam = _camera_pose_to_transform(state[STATE_CAMERA_INPUT_SLICE]) if state.shape[-1] >= 26 else None
    align_pos_err, align_dir_err = _meta_tracking_errors(
        area_type=area_type,
        qpos_real=approach_plan.aligned_qpos_real,
        T_wrist_meta=T_wrist_meta_new,
        target_meta_pose=old_target_meta[0],
        fk_fn=helpers["fk"],
    )
    position_errors.append(align_pos_err)
    direction_errors.append(align_dir_err)

    for step in range(horizon):
        if step < approach_steps:
            alpha = float(step + 1) / float(max(approach_steps, 1))
            out_action = actions[0].copy()
            out_action[:14] = state_qpos
            out_action[RIGHT_ARM_QPOS_SLICE] = (
                (1.0 - alpha) * state_qpos[RIGHT_ARM_QPOS_SLICE]
                + alpha * approach_plan.aligned_qpos_real[RIGHT_ARM_QPOS_SLICE]
            ).astype(np.float32)
            out_action[ACTION_META_SLICE] = _interpolate_meta_pose(
                area_type,
                new_input_pose,
                old_target_meta[0],
                alpha,
            )
        else:
            source_idx = min(step - approach_steps, horizon - 1)
            out_action = actions[source_idx].copy()
            target_meta = old_target_meta[source_idx]
            solved_real = _sim_to_real_right_arm_qpos(follow_qpos_sim[source_idx], helpers)

            out_action[RIGHT_ARM_QPOS_SLICE] = solved_real
            out_action[ACTION_META_SLICE] = target_meta
            if not bool(follow_converged[source_idx]):
                ik_nonconverged += 1

        if config.recompute_action_camera_pose and T_wrist_cam is not None and out_action.shape[-1] >= 26:
            T_action_wrist = np.asarray(helpers["fk"](out_action[:14], "right_wrist"), dtype=np.float32)
            out_action[ACTION_CAMERA_INPUT_SLICE] = _transform_to_camera_pose(T_action_wrist @ T_wrist_cam)

        out_actions[step] = out_action

        pos_err, dir_err = _meta_tracking_errors(
            area_type=area_type,
            qpos_real=out_action[:14],
            T_wrist_meta=T_wrist_meta_new,
            target_meta_pose=out_action[ACTION_META_SLICE],
            fk_fn=helpers["fk"],
        )
        position_errors.append(pos_err)
        direction_errors.append(dir_err)
        joint_step_deltas.append(float(np.max(np.abs(out_action[:14] - previous_real_qpos[:14]))))
        previous_real_qpos = out_action[:14].copy()

    meta_area_pose6d[0] = new_input_pose
    max_position_error = float(max(position_errors, default=np.inf))
    max_direction_error = float(max(direction_errors, default=0.0))
    max_joint_delta = float(max(joint_step_deltas, default=0.0))
    accepted = (
        max_position_error <= config.accept_max_position_error_m
        and max_direction_error <= config.accept_max_direction_error_rad
        and max_joint_delta <= config.accept_max_step_joint_delta_rad
    )
    reason = ""
    if not accepted:
        reason = (
            f"ik_nonconverged={ik_nonconverged}, "
            f"max_position_error_m={max_position_error:.6f}, "
            f"max_direction_error_rad={max_direction_error:.6f}, "
            f"max_step_joint_delta_rad={max_joint_delta:.6f}"
        )

    diagnostics = MetaRetargetDiagnostics(
        accepted=accepted,
        area_type=area_type,
        horizon=horizon,
        approach_steps=approach_steps,
        ik_nonconverged=ik_nonconverged,
        max_position_error_m=max_position_error,
        max_direction_error_rad=max_direction_error,
        max_step_joint_delta_rad=max_joint_delta,
        reason=reason,
    )
    return MetaRetargetResult(
        state=state.astype(np.float32),
        actions=out_actions.astype(np.float32),
        meta_area_pose6d=meta_area_pose6d.astype(np.float32),
        meta_area_type=meta_area_type.astype(np.int32),
        meta_area_mask=meta_area_mask.astype(bool),
        diagnostics=diagnostics,
    )


def _sim_to_real_right_arm_qpos(sim_qpos_6: np.ndarray, helpers: dict[str, Any]) -> np.ndarray:
    sim_qpos_6 = np.asarray(sim_qpos_6, dtype=np.float32).reshape(-1)[:6]
    return sim_qpos_6 * helpers["right_real_to_sim_sign"] + helpers["right_real_zero_arm"]


def _wrap_near_seed(qpos: np.ndarray, seed: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float32)
    seed = np.asarray(seed, dtype=np.float32)
    return (seed + np.arctan2(np.sin(q - seed), np.cos(q - seed))).astype(np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = r / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R.astype(np.float32)


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    ) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32)


def _random_small_rotation(rng: np.random.Generator, max_abs_deg: float) -> np.ndarray:
    if max_abs_deg <= 0:
        return np.eye(3, dtype=np.float32)
    axis = rng.normal(size=3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis = axis / axis_norm
    angle = float(rng.uniform(-np.deg2rad(max_abs_deg), np.deg2rad(max_abs_deg)))
    return _rotvec_to_matrix(axis * angle)


def _make_meta_transform(area_type: str, position_3d: np.ndarray, direction_3d: np.ndarray) -> np.ndarray:
    origin = np.asarray(position_3d, dtype=np.float32).reshape(3)
    direction = _normalize(direction_3d)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(helper, direction))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if area_type == "line":
        x_axis = direction
        z_axis = _normalize(np.cross(x_axis, helper))
        y_axis = _normalize(np.cross(z_axis, x_axis))
    else:
        z_axis = direction
        x_axis = _normalize(np.cross(helper, z_axis))
        y_axis = _normalize(np.cross(z_axis, x_axis))

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin
    return T


def _base_meta_pose_to_wrist_transform(area_type: str, pose6d: np.ndarray, T_base_wrist: np.ndarray) -> np.ndarray:
    pose6d = np.asarray(pose6d, dtype=np.float32).reshape(6)
    T_base_wrist = np.asarray(T_base_wrist, dtype=np.float32).reshape(4, 4)
    R_base_wrist = T_base_wrist[:3, :3]
    p_wrist = R_base_wrist.T @ (pose6d[:3] - T_base_wrist[:3, 3])
    if area_type == "point":
        direction_wrist = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        direction_wrist = _normalize(R_base_wrist.T @ _normalize(pose6d[3:6]))
    return _make_meta_transform(area_type, p_wrist, direction_wrist)


def _perturb_wrist_meta_transform(
    area_type: str,
    T_wrist_meta: np.ndarray,
    rng: np.random.Generator,
    config: MetaRetargetGeneratorConfig,
) -> np.ndarray:
    T_new = np.asarray(T_wrist_meta, dtype=np.float32).copy()
    if config.position_noise_max_m > 0:
        T_new[:3, 3] += rng.uniform(
            -config.position_noise_max_m,
            config.position_noise_max_m,
            size=3,
        ).astype(np.float32)
    if area_type != "point" and config.direction_noise_max_deg > 0:
        axis_col = 0 if area_type == "line" else 2
        new_direction = _normalize(_random_small_rotation(rng, config.direction_noise_max_deg) @ T_new[:3, axis_col])
        T_new = _make_meta_transform(area_type, T_new[:3, 3], new_direction)
    return T_new.astype(np.float32)


def _encode_meta_from_transform(area_type: str, T_base_meta: np.ndarray) -> np.ndarray:
    T_base_meta = np.asarray(T_base_meta, dtype=np.float32).reshape(4, 4)
    if area_type == "point":
        tail = np.zeros(3, dtype=np.float32)
    elif area_type == "line":
        tail = _normalize(T_base_meta[:3, 0])
    else:
        tail = _normalize(T_base_meta[:3, 2])
    return np.concatenate([T_base_meta[:3, 3], tail], axis=0).astype(np.float32)


def _build_target_meta_trajectory(
    area_type: str,
    *,
    start_pose: np.ndarray,
    old_target_meta: np.ndarray,
    approach_steps: int,
) -> np.ndarray:
    old_target_meta = np.asarray(old_target_meta, dtype=np.float32)
    horizon = old_target_meta.shape[0]
    approach = min(max(int(approach_steps), 0), horizon)
    if approach == 0:
        return old_target_meta.copy()

    target = np.zeros_like(old_target_meta)
    first_old = old_target_meta[0]
    for step in range(approach):
        alpha = float(step + 1) / float(approach)
        target[step] = _interpolate_meta_pose(area_type, start_pose, first_old, alpha)
    for step in range(approach, horizon):
        target[step] = old_target_meta[min(step - approach, horizon - 1)]
    return target.astype(np.float32)


def _compute_dynamic_approach_plan(
    *,
    area_type: str,
    state_qpos: np.ndarray,
    first_target_meta: np.ndarray,
    T_wrist_meta: np.ndarray,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
    horizon: int,
) -> _DynamicApproachPlan:
    """Choose approach length from raw right-arm qpos delta.

    We first solve the qpos needed to make the perturbed meta area coincide with
    the first original meta target. The approach length is then
    ceil(max(|delta right qpos|) / approach_joint_step_rad), clipped to the
    current chunk horizon.
    """

    if horizon <= 0:
        aligned = np.asarray(state_qpos, dtype=np.float32).reshape(-1)[:14].copy()
        return _DynamicApproachPlan(steps=0, aligned_qpos_real=aligned, converged=True, final_error=0.0)
    if config.approach_joint_step_rad <= 0:
        raise ValueError(f"approach_joint_step_rad must be positive, got {config.approach_joint_step_rad}")

    alignment = _solve_right_arm_meta_ik(
        template_qpos_real=np.asarray(state_qpos, dtype=np.float32).reshape(-1)[:14],
        target_meta_pose=first_target_meta,
        T_wrist_meta=T_wrist_meta,
        area_type=area_type,
        initial_qpos_sim=initial_qpos_sim,
        config=config,
        helpers=helpers,
    )
    solved_sim = _wrap_near_seed(alignment["qpos_sim"], initial_qpos_sim)
    solved_real = _sim_to_real_right_arm_qpos(solved_sim, helpers)
    aligned_qpos_real = np.asarray(state_qpos, dtype=np.float32).reshape(-1)[:14].copy()
    aligned_qpos_real[RIGHT_ARM_QPOS_SLICE] = solved_real
    right_delta = np.asarray(solved_real, dtype=np.float32) - np.asarray(state_qpos[RIGHT_ARM_QPOS_SLICE], dtype=np.float32)
    max_abs_delta = float(np.max(np.abs(right_delta)))
    steps = int(np.ceil(max_abs_delta / float(config.approach_joint_step_rad)))
    if config.max_approach_steps is not None:
        steps = min(steps, int(config.max_approach_steps))
    steps = min(max(steps, 0), int(horizon))
    return _DynamicApproachPlan(
        steps=steps,
        aligned_qpos_real=aligned_qpos_real.astype(np.float32),
        converged=bool(alignment["converged"]),
        final_error=float(alignment["final_error"]),
    )


def _interpolate_meta_pose(area_type: str, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(6)
    b = np.asarray(b, dtype=np.float32).reshape(6)
    out = np.zeros(6, dtype=np.float32)
    out[:3] = (1.0 - alpha) * a[:3] + alpha * b[:3]
    if area_type != "point":
        out[3:6] = _slerp_unit_vectors(_normalize(a[3:6]), _normalize(b[3:6]), alpha)
    return out


def _slerp_unit_vectors(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate directed unit vectors along the great-circle arc."""

    start = _normalize(a)
    end = _normalize(b)
    t = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    theta = float(np.arccos(dot))

    if theta < 1e-6:
        return start.astype(np.float32)
    if np.pi - theta < 1e-5:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(helper, start))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = _normalize(np.cross(start, helper))
        return _normalize(_rotvec_to_matrix(axis * (theta * t)) @ start)

    sin_theta = float(np.sin(theta))
    return _normalize(
        (np.sin((1.0 - t) * theta) / sin_theta) * start
        + (np.sin(t * theta) / sin_theta) * end
    )


def _solve_right_arm_meta_ik(
    *,
    template_qpos_real: np.ndarray,
    target_meta_pose: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> dict[str, Any]:
    if config.ik_backend == "jax":
        return _solve_right_arm_meta_ik_jax(
            target_meta_pose=target_meta_pose,
            T_wrist_meta=T_wrist_meta,
            area_type=area_type,
            initial_qpos_sim=initial_qpos_sim,
            config=config,
        )
    if config.ik_backend != "numpy":
        raise ValueError(f"Unsupported ik_backend={config.ik_backend!r}; expected 'numpy' or 'jax'")
    return _solve_right_arm_meta_ik_numpy(
        template_qpos_real=template_qpos_real,
        target_meta_pose=target_meta_pose,
        T_wrist_meta=T_wrist_meta,
        area_type=area_type,
        initial_qpos_sim=initial_qpos_sim,
        config=config,
        helpers=helpers,
    )


def _solve_right_arm_meta_ik_numpy(
    *,
    template_qpos_real: np.ndarray,
    target_meta_pose: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> dict[str, Any]:
    q = np.asarray(initial_qpos_sim, dtype=np.float64).reshape(6).copy()
    target_feature = _weighted_meta_feature_from_pose(area_type, target_meta_pose, config)
    converged = False
    final_error = np.inf
    iterations = 0

    for iterations in range(1, config.ik_max_iters + 1):
        feature = _weighted_meta_feature_from_sim_qpos(
            q,
            template_qpos_real=template_qpos_real,
            T_wrist_meta=T_wrist_meta,
            area_type=area_type,
            config=config,
            helpers=helpers,
        )
        error = target_feature - feature
        final_error = float(np.linalg.norm(error))
        if final_error < config.ik_tolerance:
            converged = True
            break

        J = _numerical_meta_feature_jacobian(
            q,
            template_qpos_real=template_qpos_real,
            T_wrist_meta=T_wrist_meta,
            area_type=area_type,
            config=config,
            helpers=helpers,
        )
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + (config.ik_damping**2) * np.eye(J.shape[0]), error)
        max_step = float(np.max(np.abs(dq)))
        if max_step > config.ik_max_joint_step_rad > 0:
            dq *= config.ik_max_joint_step_rad / max_step
        q += config.ik_step_scale * dq

    return {
        "qpos_sim": q.astype(np.float32),
        "converged": converged,
        "iterations": iterations,
        "final_error": final_error,
    }


def _solve_right_arm_meta_ik_jax(
    *,
    target_meta_pose: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
) -> dict[str, Any]:
    result = _solve_right_arm_meta_ik_sequence_jax(
        target_meta_poses=np.asarray(target_meta_pose, dtype=np.float32).reshape(1, 6),
        T_wrist_meta=T_wrist_meta,
        area_type=area_type,
        initial_qpos_sim=initial_qpos_sim,
        config=config,
    )
    return {
        "qpos_sim": result["qpos_sim"][0],
        "converged": bool(result["converged"][0]),
        "iterations": int(config.ik_max_iters),
        "final_error": float(result["final_error"][0]),
    }


def _solve_right_arm_meta_ik_sequence_jax(
    *,
    target_meta_poses: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
) -> dict[str, np.ndarray]:
    solver = _get_jax_ik_sequence_solver(
        area_type=area_type,
        ik_max_iters=int(config.ik_max_iters),
        ik_tolerance=float(config.ik_tolerance),
        ik_damping=float(config.ik_damping),
        ik_step_scale=float(config.ik_step_scale),
        ik_position_weight=float(config.ik_position_weight),
        ik_direction_weight=float(config.ik_direction_weight),
        ik_max_joint_step_rad=float(config.ik_max_joint_step_rad),
    )
    jax = _import_jax()
    qpos_sim, converged, final_error = solver(
        np.asarray(target_meta_poses, dtype=np.float32).reshape(-1, 6),
        np.asarray(T_wrist_meta, dtype=np.float32).reshape(4, 4),
        np.asarray(initial_qpos_sim, dtype=np.float32).reshape(6),
    )
    qpos_sim, converged, final_error = jax.device_get((qpos_sim, converged, final_error))
    return {
        "qpos_sim": np.asarray(qpos_sim, dtype=np.float32),
        "converged": np.asarray(converged, dtype=bool),
        "final_error": np.asarray(final_error, dtype=np.float32),
    }


def _solve_right_arm_meta_ik_sequence_batch_jax(
    *,
    target_meta_poses: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
) -> dict[str, np.ndarray]:
    solver = _get_jax_ik_batch_sequence_solver(
        area_type=area_type,
        ik_max_iters=int(config.ik_max_iters),
        ik_tolerance=float(config.ik_tolerance),
        ik_damping=float(config.ik_damping),
        ik_step_scale=float(config.ik_step_scale),
        ik_position_weight=float(config.ik_position_weight),
        ik_direction_weight=float(config.ik_direction_weight),
        ik_max_joint_step_rad=float(config.ik_max_joint_step_rad),
    )
    jax = _import_jax()
    qpos_sim, converged, final_error = solver(
        np.asarray(target_meta_poses, dtype=np.float32),
        np.asarray(T_wrist_meta, dtype=np.float32),
        np.asarray(initial_qpos_sim, dtype=np.float32),
    )
    qpos_sim, converged, final_error = jax.device_get((qpos_sim, converged, final_error))
    return {
        "qpos_sim": np.asarray(qpos_sim, dtype=np.float32),
        "converged": np.asarray(converged, dtype=bool),
        "final_error": np.asarray(final_error, dtype=np.float32),
    }


def _import_jax():
    try:
        import jax  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ik_backend='jax' requires JAX to be installed.") from exc
    return jax


@functools.lru_cache(maxsize=64)
def _get_jax_ik_batch_sequence_solver(
    *,
    area_type: str,
    ik_max_iters: int,
    ik_tolerance: float,
    ik_damping: float,
    ik_step_scale: float,
    ik_position_weight: float,
    ik_direction_weight: float,
    ik_max_joint_step_rad: float,
):
    jax = _import_jax()
    single_solver = _get_jax_ik_sequence_solver(
        area_type=area_type,
        ik_max_iters=ik_max_iters,
        ik_tolerance=ik_tolerance,
        ik_damping=ik_damping,
        ik_step_scale=ik_step_scale,
        ik_position_weight=ik_position_weight,
        ik_direction_weight=ik_direction_weight,
        ik_max_joint_step_rad=ik_max_joint_step_rad,
    )
    return jax.jit(jax.vmap(single_solver, in_axes=(0, 0, 0)))


@functools.lru_cache(maxsize=64)
def _get_jax_ik_sequence_solver(
    *,
    area_type: str,
    ik_max_iters: int,
    ik_tolerance: float,
    ik_damping: float,
    ik_step_scale: float,
    ik_position_weight: float,
    ik_direction_weight: float,
    ik_max_joint_step_rad: float,
):
    if area_type not in META_AREA_TYPE_TO_ID:
        raise ValueError(f"Unsupported area_type={area_type!r}")

    jax = _import_jax()
    import jax.numpy as jnp  # type: ignore

    def make_transform(R, t):
        T = jnp.eye(4, dtype=jnp.float32)
        T = T.at[:3, :3].set(jnp.asarray(R, dtype=jnp.float32))
        T = T.at[:3, 3].set(jnp.asarray(t, dtype=jnp.float32))
        return T

    def invert_transform(T):
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = jnp.eye(4, dtype=jnp.float32)
        T_inv = T_inv.at[:3, :3].set(R.T)
        T_inv = T_inv.at[:3, 3].set(-(R.T @ t))
        return T_inv

    def rot_y(theta_rad: float):
        c = jnp.cos(jnp.asarray(theta_rad, dtype=jnp.float32))
        s = jnp.sin(jnp.asarray(theta_rad, dtype=jnp.float32))
        return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=jnp.float32)

    def rot_z(theta):
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        z = jnp.zeros_like(theta)
        o = jnp.ones_like(theta)
        return jnp.stack(
            [
                jnp.stack([c, -s, z]),
                jnp.stack([s, c, z]),
                jnp.stack([z, z, o]),
            ],
            axis=0,
        ).astype(jnp.float32)

    r_flip_y = rot_y(np.pi)
    r_left_j2 = jnp.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=jnp.float32,
    )
    r_right_j5 = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float32,
    )
    r_right_j6 = jnp.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=jnp.float32,
    )

    parents = jnp.stack(
        [
            make_transform(jnp.eye(3, dtype=jnp.float32), jnp.array([1.06, 0.0, 0.2234], dtype=jnp.float32)),
            make_transform(r_left_j2, jnp.zeros(3, dtype=jnp.float32)),
            make_transform(jnp.eye(3, dtype=jnp.float32), jnp.array([0.0, -0.28, 0.0], dtype=jnp.float32)),
            make_transform(jnp.diag(jnp.array([-1.0, 1.0, -1.0], dtype=jnp.float32)), jnp.array([0.225, 0.0, 0.1175], dtype=jnp.float32)),
            make_transform(r_right_j5, jnp.array([-0.12, 0.0, 0.0], dtype=jnp.float32)),
            make_transform(r_right_j6, jnp.array([-0.087743, 0.0, 0.0], dtype=jnp.float32)),
        ],
        axis=0,
    )
    child_frames = jnp.stack(
        [
            jnp.eye(4, dtype=jnp.float32),
            jnp.eye(4, dtype=jnp.float32),
            jnp.eye(4, dtype=jnp.float32),
            jnp.eye(4, dtype=jnp.float32),
            make_transform(r_flip_y, jnp.zeros(3, dtype=jnp.float32)),
            jnp.eye(4, dtype=jnp.float32),
        ],
        axis=0,
    )
    child_inverses = jax.vmap(invert_transform)(child_frames)

    feature_dim = 3 if area_type == "point" else 6
    eye_feature = jnp.eye(feature_dim, dtype=jnp.float32)

    def normalize(v):
        norm = jnp.linalg.norm(v)
        fallback = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        return jnp.where(norm < 1e-8, fallback, v / norm)

    def right_fk(qpos_sim):
        def scan_step(T, xs):
            parent, child_inv, joint_value = xs
            T_next = T @ parent @ make_transform(rot_z(joint_value), jnp.zeros(3, dtype=jnp.float32)) @ child_inv
            return T_next, None

        T_final, _ = jax.lax.scan(
            scan_step,
            jnp.eye(4, dtype=jnp.float32),
            (parents, child_inverses, qpos_sim),
        )
        return T_final

    def encode_meta_from_transform(T_base_meta):
        if area_type == "point":
            tail = jnp.zeros(3, dtype=jnp.float32)
        elif area_type == "line":
            tail = normalize(T_base_meta[:3, 0])
        else:
            tail = normalize(T_base_meta[:3, 2])
        return jnp.concatenate([T_base_meta[:3, 3], tail], axis=0).astype(jnp.float32)

    def weighted_feature_from_pose(pose6d):
        parts = [pose6d[:3] * ik_position_weight]
        if area_type != "point":
            parts.append(normalize(pose6d[3:6]) * ik_direction_weight)
        return jnp.concatenate(parts, axis=0).astype(jnp.float32)

    def weighted_feature_from_qpos(qpos_sim, T_wrist_meta):
        T_base_wrist = right_fk(qpos_sim)
        pose6d = encode_meta_from_transform(T_base_wrist @ T_wrist_meta)
        return weighted_feature_from_pose(pose6d)

    def wrap_near_seed(qpos, seed):
        return (seed + jnp.arctan2(jnp.sin(qpos - seed), jnp.cos(qpos - seed))).astype(jnp.float32)

    def solve_one(seed_qpos_sim, target_meta_pose, T_wrist_meta):
        seed_qpos_sim = seed_qpos_sim.astype(jnp.float32)
        target_feature = weighted_feature_from_pose(target_meta_pose)
        jacobian_fn = jax.jacfwd(lambda q: weighted_feature_from_qpos(q, T_wrist_meta))

        def cond_fn(carry):
            _, iteration, err_norm = carry
            return (iteration < ik_max_iters) & (err_norm >= ik_tolerance)

        def body_fn(carry):
            q, iteration, _ = carry
            feature = weighted_feature_from_qpos(q, T_wrist_meta)
            error = target_feature - feature
            err_norm = jnp.linalg.norm(error)
            J = jacobian_fn(q)
            JT = J.T
            dq = JT @ jnp.linalg.solve(J @ JT + (ik_damping**2) * eye_feature, error)
            if ik_max_joint_step_rad > 0:
                max_step = jnp.max(jnp.abs(dq))
                dq = dq * jnp.minimum(1.0, ik_max_joint_step_rad / jnp.maximum(max_step, 1e-8))
            q_next = (q + ik_step_scale * dq).astype(jnp.float32)
            next_error = jnp.linalg.norm(target_feature - weighted_feature_from_qpos(q_next, T_wrist_meta))
            return q_next, iteration + 1, next_error.astype(jnp.float32)

        initial_error = jnp.linalg.norm(target_feature - weighted_feature_from_qpos(seed_qpos_sim, T_wrist_meta))
        q_final, _, final_error = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (seed_qpos_sim, jnp.asarray(0, dtype=jnp.int32), initial_error.astype(jnp.float32)),
        )
        q_final = wrap_near_seed(q_final, seed_qpos_sim)
        final_error = jnp.linalg.norm(target_feature - weighted_feature_from_qpos(q_final, T_wrist_meta))
        return q_final, final_error < ik_tolerance, final_error.astype(jnp.float32)

    @jax.jit
    def solve_sequence(target_meta_poses, T_wrist_meta, initial_qpos_sim):
        def scan_step(seed, target_meta_pose):
            qpos_sim, converged, final_error = solve_one(seed, target_meta_pose, T_wrist_meta)
            return qpos_sim, (qpos_sim, converged, final_error)

        _, outputs = jax.lax.scan(scan_step, initial_qpos_sim.astype(jnp.float32), target_meta_poses)
        return outputs

    return solve_sequence


def _weighted_meta_feature_from_pose(
    area_type: str,
    pose6d: np.ndarray,
    config: MetaRetargetGeneratorConfig,
) -> np.ndarray:
    pose6d = np.asarray(pose6d, dtype=np.float32).reshape(6)
    parts = [pose6d[:3] * config.ik_position_weight]
    if area_type != "point":
        parts.append(_normalize(pose6d[3:6]) * config.ik_direction_weight)
    return np.concatenate(parts, axis=0).astype(np.float64)


def _weighted_meta_feature_from_sim_qpos(
    qpos_sim: np.ndarray,
    *,
    template_qpos_real: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> np.ndarray:
    full_real = np.asarray(template_qpos_real, dtype=np.float32).reshape(-1)[:14].copy()
    full_real[RIGHT_ARM_QPOS_SLICE] = _sim_to_real_right_arm_qpos(np.asarray(qpos_sim, dtype=np.float32), helpers)
    T_base_wrist = np.asarray(helpers["fk"](full_real, "right_wrist"), dtype=np.float32)
    pose6d = _encode_meta_from_transform(area_type, T_base_wrist @ T_wrist_meta)
    return _weighted_meta_feature_from_pose(area_type, pose6d, config)


def _numerical_meta_feature_jacobian(
    qpos_sim: np.ndarray,
    *,
    template_qpos_real: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> np.ndarray:
    q = np.asarray(qpos_sim, dtype=np.float64).reshape(6)
    base_feature = _weighted_meta_feature_from_sim_qpos(
        q,
        template_qpos_real=template_qpos_real,
        T_wrist_meta=T_wrist_meta,
        area_type=area_type,
        config=config,
        helpers=helpers,
    )
    J = np.zeros((base_feature.shape[0], q.shape[0]), dtype=np.float64)
    for dim in range(q.shape[0]):
        q_perturbed = q.copy()
        q_perturbed[dim] += config.ik_finite_diff_eps
        feature = _weighted_meta_feature_from_sim_qpos(
            q_perturbed,
            template_qpos_real=template_qpos_real,
            T_wrist_meta=T_wrist_meta,
            area_type=area_type,
            config=config,
            helpers=helpers,
        )
        J[:, dim] = (feature - base_feature) / config.ik_finite_diff_eps
    return J


def _meta_tracking_errors(
    *,
    area_type: str,
    qpos_real: np.ndarray,
    T_wrist_meta: np.ndarray,
    target_meta_pose: np.ndarray,
    fk_fn,
) -> tuple[float, float]:
    T_base_wrist = np.asarray(fk_fn(qpos_real, "right_wrist"), dtype=np.float32)
    actual_pose = _encode_meta_from_transform(area_type, T_base_wrist @ T_wrist_meta)
    target_meta_pose = np.asarray(target_meta_pose, dtype=np.float32).reshape(6)
    position_error = float(np.linalg.norm(actual_pose[:3] - target_meta_pose[:3]))
    if area_type == "point":
        direction_error = 0.0
    else:
        dot = float(np.clip(np.dot(_normalize(actual_pose[3:6]), _normalize(target_meta_pose[3:6])), -1.0, 1.0))
        direction_error = float(np.arccos(dot))
    return position_error, direction_error


def _camera_pose_to_transform(pose6d: np.ndarray) -> np.ndarray:
    pose6d = np.asarray(pose6d, dtype=np.float32).reshape(6)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose6d[:3]
    T[:3, :3] = _rotvec_to_matrix(pose6d[3:6])
    return T


def _transform_to_camera_pose(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    return np.concatenate([T[:3, 3], _matrix_to_rotvec(T[:3, :3])], axis=0).astype(np.float32)
