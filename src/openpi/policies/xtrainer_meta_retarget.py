"""Offline retarget augmentation for x-trainer geometry meta chunks.

This module operates on the canonical chunk format produced after the x-trainer
meta input adapters:

  state:                 [32]
  meta_areas.pose6d:     [M, 6] or meta_areas.pose12d: [M, 12]
  meta_areas.type:       [M]
  meta_areas.mask:       [M]
  actions:               [H, 32]

The retarget operation perturbs the wrist-local geometry meta area, then solves
right-arm qpos so the perturbed meta area follows the original meta-action
trajectory. It is designed for offline cache generation. Training should sample
from the cache rather than running IK in every dataloader worker.
"""

from __future__ import annotations

import dataclasses
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
META12_DIM = 12
META6_DIM = 6


def _meta_slice_for_dim(meta_dim: int) -> slice:
    return slice(14, 14 + int(meta_dim))


def _camera_slice_for_dim(meta_dim: int) -> slice:
    del meta_dim
    return ACTION_CAMERA_INPUT_SLICE


@dataclass(frozen=True)
class MetaRetargetGeneratorConfig:
    """Configuration for one canonical chunk retarget attempt."""

    position_noise_max_m: float = 0.04
    direction_noise_max_deg: float = 35.0
    future_near_mode_prob: float = 0.5
    future_near_window_frames: int = 6
    future_near_position_noise_max_m: float = 0.008
    future_near_direction_noise_max_deg: float = 7.0
    future_near_transition_steps: int = 8
    correction_forward_exclusion_angle_deg: float = 70.0
    correction_min_position_offset_m: float = 0.02
    correction_min_direction_offset_deg: float = 13.0
    correction_max_sample_attempts: int = 64
    pair_retarget_lookahead_frames: int = 5
    # Correction uses a short imagined path before resuming the original meta
    # trajectory. resume_source_index=2 means the first two origin trajectory
    # steps are consumed by the smooth correction path, then tracking resumes at
    # the third semantic step.
    correction_resume_source_index: int = 2
    approach_joint_step_rad: float = 0.03
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
    wrist_target_shape_weight: float = 1.0
    wrist_target_approach_weight: float = 1.0
    wrist_target_pos_weight: float = 0.1
    wrist_target_rot_weight: float = 0.3
    wrist_target_roll_weight: float = 0.3
    wrist_target_shape_clip_low_deg: float = 3.0
    wrist_target_shape_clip_high_deg: float = 30.0
    wrist_target_approach_clip_low_deg: float = 3.0
    wrist_target_approach_clip_high_deg: float = 30.0
    wrist_target_pos_clip_low_m: float = 0.02
    wrist_target_pos_clip_high_m: float = 0.30
    wrist_target_rot_clip_low_deg: float = 5.0
    wrist_target_rot_clip_high_deg: float = 50.0
    wrist_target_roll_free_angle_deg: float = 45.0
    wrist_target_roll_clip_low_deg: float = 5.0
    wrist_target_roll_clip_high_deg: float = 35.0
    wrist_target_angle_tail_slope_per_deg: float = 0.01
    wrist_target_distance_tail_slope_per_m: float = 1.0
    wrist_target_shape_tail_slope_per_deg: float = 0.01
    wrist_target_approach_tail_slope_per_deg: float = 0.01
    wrist_target_rot_tail_slope_per_deg: float = 0.01
    wrist_target_roll_tail_slope_per_deg: float = 0.01
    wrist_target_roll_samples: int = 37
    follow_wrist_target_shape_weight: float = 0.8
    follow_wrist_target_approach_weight: float = 0.8
    follow_wrist_target_pos_weight: float = 1.0
    follow_wrist_target_rot_weight: float = 1.0
    follow_wrist_target_pos_clip_low_m: float = 0.01
    follow_wrist_target_pos_clip_high_m: float = 0.04
    follow_wrist_target_pos_tail_slope_per_m: float = 50.0
    follow_wrist_target_rot_clip_low_deg: float = 4.0
    follow_wrist_target_rot_clip_high_deg: float = 13.0
    follow_wrist_target_rot_tail_slope_per_deg: float = 0.125
    accept_max_position_error_m: float = 0.015
    accept_max_direction_error_rad: float = 0.4
    accept_max_step_joint_delta_rad: float = 0.35
    accept_max_abs_action_value: float = 1e4
    accept_max_camera_rotvec_norm_rad: float = 3.143
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
    actions_finite: bool = True
    max_abs_action_value: float = 0.0
    max_camera_rotvec_norm_rad: float = 0.0
    retarget_mode: str = "correction"
    trajectory_start_index: int = 0

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
    meta_area_pose12d: np.ndarray | None = None
    meta_area_dim_mask12: np.ndarray | None = None
    meta_action_target_pose12d: np.ndarray | None = None
    meta_action_target_dim_mask12: np.ndarray | None = None
    meta_action_target_mask: np.ndarray | None = None


@dataclass(frozen=True)
class _DynamicApproachPlan:
    steps: int
    aligned_qpos_real: np.ndarray
    converged: bool
    final_error: float
    qpos_real_sequence: np.ndarray | None = None


def _set_retarget_meta_action(
    out_action: np.ndarray,
    out_target_pose12d: np.ndarray | None,
    step: int,
    target_meta: np.ndarray,
    action_meta_slice: slice,
    meta_dim: int,
) -> None:
    if meta_dim == META12_DIM:
        if out_target_pose12d is None:
            raise ValueError("12D retargeting requires meta_action_targets.pose12d output storage.")
        out_target_pose12d[step, 0, :] = np.asarray(target_meta, dtype=np.float32)
    else:
        out_action[action_meta_slice] = target_meta


def _retarget_meta_for_error(
    out_action: np.ndarray,
    out_target_pose12d: np.ndarray | None,
    step: int,
    target_meta: np.ndarray,
    action_meta_slice: slice,
    meta_dim: int,
) -> np.ndarray:
    if meta_dim == META12_DIM:
        if out_target_pose12d is not None:
            return out_target_pose12d[step, 0]
        return np.asarray(target_meta, dtype=np.float32)
    return out_action[action_meta_slice]


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

    meta_dim = META6_DIM
    meta_key = "pose6d"
    meta_target_key = "pose6d"
    meta_areas = data.get("meta_areas")
    if meta_areas is not None and "pose12d" in meta_areas:
        meta_dim = META12_DIM
        meta_key = "pose12d"
        meta_target_key = "pose12d"

    meta_area_poses = np.zeros((max_meta_areas, meta_dim), dtype=np.float32)
    meta_area_dim_masks = np.zeros((max_meta_areas, meta_dim), dtype=bool)
    meta_area_types = np.full((max_meta_areas,), META_AREA_TYPE_TO_ID["line"], dtype=np.int32)
    meta_area_masks = np.zeros((max_meta_areas,), dtype=bool)

    if meta_areas is not None:
        poses = np.asarray(meta_areas[meta_key], dtype=np.float32)
        types = np.asarray(meta_areas["type"], dtype=np.int32)
        masks = np.asarray(meta_areas["mask"], dtype=bool)
        if meta_dim == META12_DIM:
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
        count = min(max_meta_areas, poses.shape[0])
        meta_area_poses[:count] = poses[:count]
        meta_area_dim_masks[:count] = dim_masks[:count]
        meta_area_types[:count] = types[:count]
        meta_area_masks[:count] = masks[:count]
    elif derive_meta_from_state_if_missing and raw_state.shape[-1] >= STATE_META_SLICE.stop:
        meta_area_poses[0] = raw_state[STATE_META_SLICE]
        meta_area_dim_masks[0] = True
        meta_area_types[0] = META_AREA_TYPE_TO_ID["line"]
        meta_area_masks[0] = True

    action_meta_slice = _meta_slice_for_dim(meta_dim)
    meta_targets = data.get("meta_action_targets")
    meta_targets_out = None
    if meta_targets is not None:
        target_pose = np.asarray(meta_targets[meta_target_key], dtype=np.float32)
        if target_pose.ndim not in (2, 3):
            raise ValueError(f"Expected meta_action_targets.{meta_target_key} with 2 or 3 dims, got {target_pose.shape}")
        target_mask = np.asarray(meta_targets.get("mask", np.ones(target_pose.shape[:-1], dtype=bool)), dtype=bool)
        if target_mask.ndim == 3 and target_mask.shape[-1] == 1:
            target_mask = np.squeeze(target_mask, axis=-1)
        meta_targets_out = {meta_target_key: target_pose.copy(), "mask": target_mask.copy()}
        if meta_dim == META12_DIM:
            raw_dim_mask = meta_targets.get("dim_mask12")
            meta_targets_out["dim_mask12"] = (
                np.asarray(raw_dim_mask, dtype=bool).copy()
                if raw_dim_mask is not None
                else np.ones(target_pose.shape, dtype=bool)
            )
        elif fill_action_meta_slice_from_targets and actions.shape[-1] >= action_meta_slice.stop:
            if target_pose.ndim == 3:
                actions[..., action_meta_slice] = target_pose[:, 0, :]
            else:
                actions[..., action_meta_slice] = target_pose

    if meta_dim == META12_DIM and actions.shape[-1] >= ACTION_META_SLICE.stop:
        actions[..., ACTION_META_SLICE] = 0.0

    if zero_meta_state_slice and state.shape[-1] >= STATE_META_SLICE.stop:
        state[STATE_META_SLICE] = 0.0

    meta_areas_out = {
        meta_key: meta_area_poses,
        "type": meta_area_types,
        "mask": meta_area_masks,
    }
    if meta_dim == META12_DIM:
        meta_areas_out["dim_mask12"] = meta_area_dim_masks

    out = {
        "state": state,
        "actions": actions,
        "meta_areas": meta_areas_out,
    }
    if meta_targets_out is not None:
        out["meta_action_targets"] = meta_targets_out
    return out


def generate_retargeted_chunk(
    data: dict[str, Any],
    *,
    rng: np.random.Generator,
    config: MetaRetargetGeneratorConfig,
) -> MetaRetargetResult | None:
    """Generate one retargeted chunk, returning ``None`` if the sample is not usable."""

    helpers = _load_xtrainer_helpers()
    item = _prepare_retarget_chunk(data, rng=rng, config=config, helpers=helpers)
    if item is None:
        return None
    if int(item.get("meta_dim", META6_DIM)) == META12_DIM and config.ik_backend == "jax":
        # The 12D objective includes a shape matrix and optional approach vector.
        # Keep the first implementation semantically correct by using the numpy
        # finite-difference IK path; the cached result can still be consumed by
        # the normal dataloader before normalization.
        config = dataclasses.replace(config, ik_backend="numpy")

    if item["retarget_mode"] == "future_near":
        qpos_sim, converged = _solve_smooth_chase_sequence(
            item,
            initial_qpos_sim=item["initial_seed_sim"],
            config=config,
            helpers=helpers,
        )
        return _assemble_smooth_chase_result(
            item,
            qpos_sim=np.asarray(qpos_sim, dtype=np.float32),
            converged=np.asarray(converged, dtype=bool),
            config=config,
            helpers=helpers,
        )

    return _generate_correction_chunk_from_item(item, config=config, helpers=helpers)


def generate_pair_retargeted_chunk(
    target_data: dict[str, Any],
    source_data: dict[str, Any],
    *,
    rng: np.random.Generator,
    config: MetaRetargetGeneratorConfig,
) -> MetaRetargetResult | None:
    """Retarget ``target_data`` using the wrist-local 12D affordance from ``source_data``.

    The source chunk supplies the tool geometry in its right-wrist frame. The target
    chunk supplies the observation/action trajectory and the original meta target
    path. A short lookahead over the target path decides whether the transferred
    source affordance can be treated as a future-near or correction retarget. If
    no legal anchor is found, the pair is dismissed by returning ``None``.
    """

    helpers = _load_xtrainer_helpers()
    fixed = _pair_source_affordance_as_target_input_pose(source_data, target_data, helpers=helpers)
    if fixed is None:
        return None

    data = dict(target_data)
    data["_retarget_fixed_input_pose12d"] = fixed["input_pose12d"]
    data["_retarget_fixed_area_type_id"] = fixed["area_type_id"]
    data["_retarget_fixed_dim_mask12"] = fixed["dim_mask12"]
    item = _prepare_retarget_chunk(data, rng=rng, config=config, helpers=helpers)
    if item is None:
        return None
    if int(item.get("meta_dim", META6_DIM)) == META12_DIM and config.ik_backend == "jax":
        config = dataclasses.replace(config, ik_backend="numpy")

    if item["retarget_mode"] == "future_near":
        qpos_sim, converged = _solve_smooth_chase_sequence(
            item,
            initial_qpos_sim=item["initial_seed_sim"],
            config=config,
            helpers=helpers,
        )
        return _assemble_smooth_chase_result(
            item,
            qpos_sim=np.asarray(qpos_sim, dtype=np.float32),
            converged=np.asarray(converged, dtype=bool),
            config=config,
            helpers=helpers,
        )

    return _generate_correction_chunk_from_item(item, config=config, helpers=helpers)


def _generate_correction_chunk_from_item(
    item: dict[str, Any],
    *,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> MetaRetargetResult:
    state = item["state"].copy()
    actions = item["actions"]
    meta_area_pose6d = item["meta_area_pose6d"].copy()
    meta_area_pose12d = None if item["meta_area_pose12d"] is None else item["meta_area_pose12d"].copy()
    meta_area_dim_mask12 = None if item["meta_area_dim_mask12"] is None else item["meta_area_dim_mask12"].copy()
    out_target_pose12d = None if item["meta_action_target_pose12d"] is None else item["meta_action_target_pose12d"].copy()
    out_target_dim_mask12 = (
        None if item["meta_action_target_dim_mask12"] is None else item["meta_action_target_dim_mask12"].copy()
    )
    out_target_mask = None if item["meta_action_target_mask"] is None else item["meta_action_target_mask"].copy()
    meta_area_type = item["meta_area_type"].copy()
    meta_area_mask = item["meta_area_mask"].copy()
    meta_dim = int(item.get("meta_dim", META6_DIM))
    action_meta_slice = item.get("action_meta_slice", _meta_slice_for_dim(meta_dim))
    action_camera_slice = item.get("action_camera_slice", _camera_slice_for_dim(meta_dim))
    dim_mask12 = item.get("dim_mask12")
    horizon = int(item["horizon"])
    area_type = item["area_type"]
    state_qpos = item["state_qpos"]
    T_wrist_meta_new = item["T_wrist_meta_new"]
    new_input_pose = item["new_input_pose"]
    old_target_meta = item["old_target_meta"]
    trajectory_start_index = min(max(int(item.get("trajectory_start_index", 0)), 0), horizon - 1)
    seed_sim = item["initial_seed_sim"]
    resume_source_index = min(max(int(config.correction_resume_source_index), 1), horizon - 1)
    join_source_index = min(trajectory_start_index + max(resume_source_index - 1, 0), horizon - 1)
    join_target_meta = old_target_meta[join_source_index]
    approach_plan = _compute_dynamic_approach_plan(
        area_type=area_type,
        state_qpos=state_qpos,
        first_target_meta=join_target_meta,
        T_wrist_meta=T_wrist_meta_new,
        initial_qpos_sim=seed_sim,
        config=config,
        helpers=helpers,
        horizon=horizon,
        dim_mask12=dim_mask12,
    )
    approach_steps = approach_plan.steps

    out_actions = actions.copy()
    previous_real_qpos = state_qpos.copy()

    ik_nonconverged = 0 if approach_plan.converged else 1
    position_errors: list[float] = []
    direction_errors: list[float] = []
    joint_step_deltas: list[float] = []
    T_wrist_cam = _camera_pose_to_transform(state[action_camera_slice]) if state.shape[-1] >= action_camera_slice.stop else None
    align_pos_err, align_dir_err = _meta_tracking_errors(
        area_type=area_type,
        qpos_real=approach_plan.aligned_qpos_real,
        T_wrist_meta=T_wrist_meta_new,
        target_meta_pose=join_target_meta,
        dim_mask12=dim_mask12,
        fk_fn=helpers["fk"],
    )
    position_errors.append(align_pos_err)
    direction_errors.append(align_dir_err)

    follow_config = _follow_wrist_target_config(config)

    for step in range(horizon):
        if step < approach_steps:
            alpha = float(step + 1) / float(max(approach_steps, 1))
            target_meta = _correction_path_pose(
                area_type,
                new_input_pose,
                join_target_meta,
                alpha,
                dim_mask12=dim_mask12,
            )
            out_action = actions[0].copy()
            out_action[:14] = state_qpos
            if approach_plan.qpos_real_sequence is not None and step < approach_plan.qpos_real_sequence.shape[0]:
                out_action[RIGHT_ARM_QPOS_SLICE] = approach_plan.qpos_real_sequence[step, RIGHT_ARM_QPOS_SLICE]
            else:
                out_action[RIGHT_ARM_QPOS_SLICE] = (
                    (1.0 - alpha) * state_qpos[RIGHT_ARM_QPOS_SLICE]
                    + alpha * approach_plan.aligned_qpos_real[RIGHT_ARM_QPOS_SLICE]
                ).astype(np.float32)
            _set_retarget_meta_action(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim)
            seed_sim = helpers["real_to_sim_arm"](out_action[RIGHT_ARM_QPOS_SLICE], "right")
        else:
            source_idx = min(step - approach_steps + trajectory_start_index + resume_source_index, horizon - 1)
            out_action = actions[source_idx].copy()
            target_meta = old_target_meta[source_idx]

            ik_result = _solve_right_arm_meta_ik(
                template_qpos_real=previous_real_qpos,
                target_meta_pose=target_meta,
                T_wrist_meta=T_wrist_meta_new,
                area_type=area_type,
                initial_qpos_sim=seed_sim,
                config=follow_config,
                helpers=helpers,
                dim_mask12=dim_mask12,
            )
            solved_sim = _wrap_near_seed(ik_result["qpos_sim"], seed_sim)
            converged = bool(ik_result["converged"])

            solved_real = _sim_to_real_right_arm_qpos(solved_sim, helpers)

            out_action[RIGHT_ARM_QPOS_SLICE] = solved_real
            _set_retarget_meta_action(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim)
            seed_sim = solved_sim
            if not converged:
                ik_nonconverged += 1

        if config.recompute_action_camera_pose and T_wrist_cam is not None and out_action.shape[-1] >= action_camera_slice.stop:
            T_action_wrist = np.asarray(helpers["fk"](out_action[:14], "right_wrist"), dtype=np.float32)
            out_action[action_camera_slice] = _transform_to_camera_pose(T_action_wrist @ T_wrist_cam)

        out_actions[step] = out_action

        pos_err, dir_err = _meta_tracking_errors(
            area_type=area_type,
            qpos_real=out_action[:14],
            T_wrist_meta=T_wrist_meta_new,
            target_meta_pose=_retarget_meta_for_error(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim),
            dim_mask12=dim_mask12,
            fk_fn=helpers["fk"],
        )
        position_errors.append(pos_err)
        direction_errors.append(dir_err)
        joint_step_deltas.append(float(np.max(np.abs(out_action[:14] - previous_real_qpos[:14]))))
        previous_real_qpos = out_action[:14].copy()

    if meta_area_pose12d is not None:
        meta_area_pose12d[0] = new_input_pose
        meta_area_pose6d[0] = _pose12_to_pose6d(area_type, new_input_pose, dim_mask12)
    else:
        meta_area_pose6d[0] = new_input_pose
    max_position_error = float(max(position_errors, default=np.inf))
    max_direction_error = float(max(direction_errors, default=0.0))
    max_joint_delta = float(max(joint_step_deltas, default=0.0))
    (
        values_ok,
        actions_finite,
        max_abs_action,
        max_camera_rotvec_norm,
        value_reason,
    ) = _action_value_diagnostics(out_actions, config, meta_dim=meta_dim)

    accepted = (
        max_position_error <= config.accept_max_position_error_m
        and max_direction_error <= config.accept_max_direction_error_rad
        and max_joint_delta <= config.accept_max_step_joint_delta_rad
        and values_ok
    )
    reason = ""
    if not accepted:
        reason_parts = [
            f"ik_nonconverged={ik_nonconverged}",
            f"max_position_error_m={max_position_error:.6f}",
            f"max_direction_error_rad={max_direction_error:.6f}",
            f"max_step_joint_delta_rad={max_joint_delta:.6f}",
        ]
        if value_reason:
            reason_parts.append(value_reason)
        reason = ", ".join(reason_parts)

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
        actions_finite=actions_finite,
        max_abs_action_value=max_abs_action,
        max_camera_rotvec_norm_rad=max_camera_rotvec_norm,
        retarget_mode=item["retarget_mode"],
        trajectory_start_index=int(item["trajectory_start_index"]),
    )
    return MetaRetargetResult(
        state=state.astype(np.float32),
        actions=out_actions.astype(np.float32),
        meta_area_pose6d=meta_area_pose6d.astype(np.float32),
        meta_area_type=meta_area_type.astype(np.int32),
        meta_area_mask=meta_area_mask.astype(bool),
        diagnostics=diagnostics,
        meta_area_pose12d=None if meta_area_pose12d is None else meta_area_pose12d.astype(np.float32),
        meta_area_dim_mask12=None if meta_area_dim_mask12 is None else meta_area_dim_mask12.astype(bool),
        meta_action_target_pose12d=None if out_target_pose12d is None else out_target_pose12d.astype(np.float32),
        meta_action_target_dim_mask12=None if out_target_dim_mask12 is None else out_target_dim_mask12.astype(bool),
        meta_action_target_mask=None if out_target_mask is None else out_target_mask.astype(bool),
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
        return [
            generate_retargeted_chunk(data, rng=rng, config=config)
            for data, rng in zip(data_batch, rngs, strict=True)
        ]
    if config.approach_joint_step_rad <= 0:
        raise ValueError(f"approach_joint_step_rad must be positive, got {config.approach_joint_step_rad}")

    helpers = _load_xtrainer_helpers()
    prepared: list[dict[str, Any] | None] = [
        _prepare_retarget_chunk(data, rng=rng, config=config, helpers=helpers)
        for data, rng in zip(data_batch, rngs, strict=True)
    ]
    results: list[MetaRetargetResult | None] = [None] * len(data_batch)
    if any(item is not None and int(item.get("meta_dim", META6_DIM)) == META12_DIM for item in prepared):
        numpy_config = dataclasses.replace(config, ik_backend="numpy")
        return [
            generate_retargeted_chunk(data, rng=rng, config=numpy_config)
            for data, rng in zip(data_batch, rngs, strict=True)
        ]

    for area_type in META_AREA_TYPE_TO_ID:
        correction_indices = [
            idx
            for idx, item in enumerate(prepared)
            if item is not None and item["area_type"] == area_type and item["retarget_mode"] == "correction"
        ]
        if correction_indices:
            first_targets = []
            for idx in correction_indices:
                item = prepared[idx]
                assert item is not None
                trajectory_start_index = min(max(int(item.get("trajectory_start_index", 0)), 0), int(item["horizon"]) - 1)
                resume_source_index = min(max(int(config.correction_resume_source_index), 1), int(item["horizon"]) - 1)
                first_targets.append(
                    item["old_target_meta"][
                        min(trajectory_start_index + max(resume_source_index - 1, 0), int(item["horizon"]) - 1)
                    ]
                )
            first_targets = np.stack(first_targets, axis=0)[:, None, :]
            wrist_meta = np.stack([prepared[idx]["T_wrist_meta_new"] for idx in correction_indices], axis=0)
            initial_qpos = np.stack([prepared[idx]["initial_seed_sim"] for idx in correction_indices], axis=0)
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
            for group_offset, idx in enumerate(correction_indices):
                item = prepared[idx]
                assert item is not None
                aligned_qpos_real = item["state_qpos"].copy()
                aligned_qpos_real[RIGHT_ARM_QPOS_SLICE] = aligned_real_right[group_offset]
                right_delta = aligned_real_right[group_offset] - item["state_qpos"][RIGHT_ARM_QPOS_SLICE]
                approach_steps = int(
                    np.ceil(float(np.max(np.abs(right_delta))) / float(config.approach_joint_step_rad))
                )
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
            follow_targets = np.stack([prepared[idx]["old_target_meta"] for idx in correction_indices], axis=0)
            follow = _solve_right_arm_meta_ik_sequence_batch_jax(
                target_meta_poses=follow_targets,
                T_wrist_meta=wrist_meta,
                area_type=area_type,
                initial_qpos_sim=follow_initial_qpos,
                config=config,
            )

            for group_offset, idx in enumerate(correction_indices):
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

        future_indices = [
            idx
            for idx, item in enumerate(prepared)
            if item is not None and item["area_type"] == area_type and item["retarget_mode"] == "future_near"
        ]
        if not future_indices:
            continue

        future_targets = np.stack([prepared[idx]["target_meta_sequence"] for idx in future_indices], axis=0)
        future_wrist_meta = np.stack([prepared[idx]["T_wrist_meta_new"] for idx in future_indices], axis=0)
        future_initial_qpos = np.stack([prepared[idx]["initial_seed_sim"] for idx in future_indices], axis=0)
        future_follow = _solve_right_arm_meta_ik_sequence_batch_jax(
            target_meta_poses=future_targets,
            T_wrist_meta=future_wrist_meta,
            area_type=area_type,
            initial_qpos_sim=future_initial_qpos,
            config=config,
        )
        for group_offset, idx in enumerate(future_indices):
            item = prepared[idx]
            assert item is not None
            results[idx] = _assemble_smooth_chase_result(
                item,
                qpos_sim=np.asarray(future_follow["qpos_sim"][group_offset], dtype=np.float32),
                converged=np.asarray(future_follow["converged"][group_offset], dtype=bool),
                config=config,
                helpers=helpers,
            )

    return results


def save_retarget_result(path: Path, result: MetaRetargetResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    arrays = {
        "state": result.state,
        "actions": result.actions,
        "meta_area_pose6d": result.meta_area_pose6d,
        "meta_area_type": result.meta_area_type,
        "meta_area_mask": result.meta_area_mask,
    }
    if result.meta_area_pose12d is not None:
        arrays["meta_area_pose12d"] = result.meta_area_pose12d
    if result.meta_area_dim_mask12 is not None:
        arrays["meta_area_dim_mask12"] = result.meta_area_dim_mask12
    if result.meta_action_target_pose12d is not None:
        arrays["meta_action_target_pose12d"] = result.meta_action_target_pose12d
    if result.meta_action_target_dim_mask12 is not None:
        arrays["meta_action_target_dim_mask12"] = result.meta_action_target_dim_mask12
    if result.meta_action_target_mask is not None:
        arrays["meta_action_target_mask"] = result.meta_action_target_mask
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
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


def _sample_random_vector_in_ball(rng: np.random.Generator, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return np.zeros(3, dtype=np.float32)
    direction = rng.normal(size=3)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-8:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        direction = direction / direction_norm
    radius = float(max_norm) * float(rng.random() ** (1.0 / 3.0))
    return (direction * radius).astype(np.float32)


def _trajectory_frame_pose(old_input_pose: np.ndarray, old_target_meta: np.ndarray, frame_index: int) -> np.ndarray:
    if frame_index <= 0:
        return np.asarray(old_input_pose, dtype=np.float32).reshape(-1).copy()
    return np.asarray(old_target_meta[min(frame_index - 1, old_target_meta.shape[0] - 1)], dtype=np.float32).copy()


def _sample_pose_near_trajectory_frame(
    area_type: str,
    base_pose: np.ndarray,
    *,
    rng: np.random.Generator,
    position_noise_max_m: float,
    direction_noise_max_deg: float,
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float32).reshape(-1).copy()
    pose[:3] += _sample_random_vector_in_ball(rng, position_noise_max_m)
    if pose.shape[0] == META12_DIM:
        if area_type != "point" and direction_noise_max_deg > 0:
            R = _random_small_rotation(rng, direction_noise_max_deg)
            pose[3:9] = _matrix_to_shape6(_project_psd_trace1(R @ _shape6_to_matrix(pose[3:9]) @ R.T))
            if _pose12_has_approach(pose, dim_mask12):
                pose[9:12] = _normalize(R @ pose[9:12])
        elif area_type == "point":
            pose[3:12] = 0.0
        return pose.astype(np.float32)

    if area_type != "point" and direction_noise_max_deg > 0:
        pose[3:6] = _normalize(_random_small_rotation(rng, direction_noise_max_deg) @ _normalize(pose[3:6]))
    elif area_type == "point":
        pose[3:6] = 0.0
    return pose.astype(np.float32)


def _initial_motion_direction(old_input_pose: np.ndarray, old_target_meta: np.ndarray, window_frames: int) -> np.ndarray:
    frame_count = min(max(int(window_frames), 2), old_target_meta.shape[0] + 1)
    points = [_trajectory_frame_pose(old_input_pose, old_target_meta, frame_idx)[:3] for frame_idx in range(frame_count)]
    points_arr = np.asarray(points, dtype=np.float32)
    times = np.arange(frame_count, dtype=np.float32)
    times = times - float(np.mean(times))
    direction = np.sum(points_arr * times[:, None], axis=0)
    if float(np.linalg.norm(direction)) < 1e-8:
        direction = points_arr[-1] - points_arr[0]
    return _normalize(direction)


def _direction_angle_rad(
    area_type: str,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    dim_mask12: np.ndarray | None = None,
) -> float:
    if area_type == "point":
        return 0.0
    pose_a = np.asarray(pose_a, dtype=np.float32).reshape(-1)
    pose_b = np.asarray(pose_b, dtype=np.float32).reshape(-1)
    if pose_a.shape[0] == META12_DIM:
        axis_a = _shape_axis_from_pose12(area_type, pose_a, dim_mask12)
        axis_b = _shape_axis_from_pose12(area_type, pose_b, dim_mask12)
        axis_dot = abs(float(np.clip(np.dot(axis_a, axis_b), -1.0, 1.0)))
        errors = [float(np.arccos(axis_dot))]
        if _pose12_has_approach(pose_a, dim_mask12) or _pose12_has_approach(pose_b, dim_mask12):
            app_dot = float(np.clip(np.dot(_normalize(pose_a[9:12]), _normalize(pose_b[9:12])), -1.0, 1.0))
            errors.append(float(np.arccos(app_dot)))
        return float(max(errors))
    dot = float(np.clip(np.dot(_normalize(pose_a[3:6]), _normalize(pose_b[3:6])), -1.0, 1.0))
    return float(np.arccos(dot))


def _sample_correction_pose(
    area_type: str,
    old_input_pose: np.ndarray,
    old_target_meta: np.ndarray,
    *,
    rng: np.random.Generator,
    config: MetaRetargetGeneratorConfig,
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray | None:
    motion_direction = _initial_motion_direction(old_input_pose, old_target_meta, config.future_near_window_frames)
    min_dir_angle = np.deg2rad(config.correction_min_direction_offset_deg)
    excluded_forward_angle = np.deg2rad(config.correction_forward_exclusion_angle_deg)

    for _ in range(max(1, int(config.correction_max_sample_attempts))):
        pose = _sample_pose_near_trajectory_frame(
            area_type,
            old_input_pose,
            rng=rng,
            position_noise_max_m=config.position_noise_max_m,
            direction_noise_max_deg=config.direction_noise_max_deg,
            dim_mask12=dim_mask12,
        )
        offset = pose[:3] - old_input_pose[:3]
        offset_norm = float(np.linalg.norm(offset))
        direction_delta = _direction_angle_rad(area_type, old_input_pose, pose, dim_mask12=dim_mask12)

        if offset_norm > config.position_noise_max_m + 1e-6:
            continue
        if direction_delta > np.deg2rad(config.direction_noise_max_deg) + 1e-6:
            continue
        if not (offset_norm > config.correction_min_position_offset_m or direction_delta > min_dir_angle):
            continue
        if offset_norm > 1e-6:
            forward_dot = float(np.clip(np.dot(offset / offset_norm, motion_direction), -1.0, 1.0))
            forward_angle = float(np.arccos(forward_dot))
            if forward_angle < excluded_forward_angle:
                continue
        return pose.astype(np.float32)
    return None


def _motion_direction_from_anchor(
    old_input_pose: np.ndarray,
    old_target_meta: np.ndarray,
    anchor_index: int,
    window_frames: int,
) -> np.ndarray:
    horizon = int(old_target_meta.shape[0])
    start = min(max(int(anchor_index), 0), horizon)
    frame_count = min(max(int(window_frames), 2), horizon - start + 1)
    points = [
        _trajectory_frame_pose(old_input_pose, old_target_meta, start + frame_idx)[:3]
        for frame_idx in range(frame_count)
    ]
    points_arr = np.asarray(points, dtype=np.float32)
    times = np.arange(frame_count, dtype=np.float32)
    times = times - float(np.mean(times))
    direction = np.sum(points_arr * times[:, None], axis=0)
    if float(np.linalg.norm(direction)) < 1e-8:
        direction = points_arr[-1] - points_arr[0]
    return _normalize(direction)


def _candidate_is_future_near(
    area_type: str,
    candidate_pose: np.ndarray,
    anchor_pose: np.ndarray,
    *,
    config: MetaRetargetGeneratorConfig,
    dim_mask12: np.ndarray | None = None,
) -> bool:
    offset_norm = float(np.linalg.norm(np.asarray(candidate_pose[:3]) - np.asarray(anchor_pose[:3])))
    direction_delta = _direction_angle_rad(area_type, anchor_pose, candidate_pose, dim_mask12=dim_mask12)
    return (
        offset_norm <= config.future_near_position_noise_max_m + 1e-6
        and direction_delta <= np.deg2rad(config.future_near_direction_noise_max_deg) + 1e-6
    )


def _candidate_is_correction(
    area_type: str,
    candidate_pose: np.ndarray,
    anchor_pose: np.ndarray,
    old_input_pose: np.ndarray,
    old_target_meta: np.ndarray,
    anchor_index: int,
    *,
    config: MetaRetargetGeneratorConfig,
    dim_mask12: np.ndarray | None = None,
) -> bool:
    offset = np.asarray(candidate_pose[:3], dtype=np.float32) - np.asarray(anchor_pose[:3], dtype=np.float32)
    offset_norm = float(np.linalg.norm(offset))
    direction_delta = _direction_angle_rad(area_type, anchor_pose, candidate_pose, dim_mask12=dim_mask12)
    if offset_norm > config.position_noise_max_m + 1e-6:
        return False
    if direction_delta > np.deg2rad(config.direction_noise_max_deg) + 1e-6:
        return False
    if not (offset_norm > config.correction_min_position_offset_m or direction_delta > np.deg2rad(config.correction_min_direction_offset_deg)):
        return False
    if offset_norm > 1e-6:
        motion_direction = _motion_direction_from_anchor(
            old_input_pose,
            old_target_meta,
            anchor_index,
            config.future_near_window_frames,
        )
        forward_dot = float(np.clip(np.dot(offset / offset_norm, motion_direction), -1.0, 1.0))
        forward_angle = float(np.arccos(forward_dot))
        if forward_angle < np.deg2rad(config.correction_forward_exclusion_angle_deg):
            return False
    return True


def _select_pair_retarget_anchor(
    area_type: str,
    candidate_pose: np.ndarray,
    old_input_pose: np.ndarray,
    old_target_meta: np.ndarray,
    *,
    config: MetaRetargetGeneratorConfig,
    dim_mask12: np.ndarray | None = None,
) -> dict[str, Any] | None:
    max_anchor = min(max(int(config.pair_retarget_lookahead_frames), 0), int(old_target_meta.shape[0]))
    for anchor_index in range(max_anchor + 1):
        anchor_pose = _trajectory_frame_pose(old_input_pose, old_target_meta, anchor_index)
        if _candidate_is_future_near(
            area_type,
            candidate_pose,
            anchor_pose,
            config=config,
            dim_mask12=dim_mask12,
        ):
            source_indices = _direct_follow_source_indices(
                old_target_meta,
                trajectory_start_index=min(anchor_index, old_target_meta.shape[0] - 1),
            )
            target_meta_sequence = old_target_meta[source_indices].astype(np.float32)
            return {
                "retarget_mode": "future_near",
                "trajectory_start_index": int(anchor_index),
                "target_meta_sequence": target_meta_sequence,
                "source_indices": source_indices,
            }

        if _candidate_is_correction(
            area_type,
            candidate_pose,
            anchor_pose,
            old_input_pose,
            old_target_meta,
            anchor_index,
            config=config,
            dim_mask12=dim_mask12,
        ):
            return {
                "retarget_mode": "correction",
                "trajectory_start_index": int(anchor_index),
                "target_meta_sequence": None,
                "source_indices": None,
            }
    return None


def _direct_follow_source_indices(old_target_meta: np.ndarray, *, trajectory_start_index: int) -> np.ndarray:
    horizon = int(old_target_meta.shape[0])
    if horizon <= 0:
        return np.zeros((0,), dtype=np.int32)
    start = min(max(int(trajectory_start_index), 0), horizon - 1)
    return np.minimum(np.arange(horizon, dtype=np.int32) + start, horizon - 1)


def _rotation_between_unit_vectors(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    a = _normalize(start)
    b = _normalize(end)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + 1e-6:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(helper, a))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = _normalize(np.cross(a, helper))
        return _rotvec_to_matrix(axis * np.pi)
    axis = _normalize(np.cross(a, b))
    angle = float(np.arccos(dot))
    return _rotvec_to_matrix(axis * angle)


def _angle_between_unit_vectors_deg(a: np.ndarray, b: np.ndarray, *, unsigned: bool = False) -> float:
    va = _normalize(a)
    vb = _normalize(b)
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    if unsigned:
        dot = abs(dot)
    return float(np.rad2deg(np.arccos(dot)))


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = np.asarray(R_a, dtype=np.float64).reshape(3, 3) @ np.asarray(R_b, dtype=np.float64).reshape(3, 3).T
    trace = float(np.trace(R_delta))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def _smoothstep_unit(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _clipped_smoothstep_with_tail(value: float, low: float, high: float, tail_slope: float) -> float:
    value = float(value)
    low = float(low)
    high = float(high)
    if high <= low:
        return 0.0 if value <= low else 1.0 + max(0.0, value - high) * float(tail_slope)
    if value <= low:
        return 0.0
    if value <= high:
        return _smoothstep_unit((value - low) / (high - low))
    return 1.0 + (value - high) * float(tail_slope)


def _wrist_roll_domain_loss(R_base_wrist: np.ndarray, config: MetaRetargetGeneratorConfig) -> float:
    wrist_x = np.asarray(R_base_wrist, dtype=np.float32).reshape(3, 3)[:, 0]
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    danger_angle = min(
        _angle_between_unit_vectors_deg(wrist_x, world_z),
        _angle_between_unit_vectors_deg(wrist_x, -world_z),
    )
    if danger_angle >= float(config.wrist_target_roll_free_angle_deg):
        return 0.0
    roll_error = float(config.wrist_target_roll_free_angle_deg) - danger_angle
    return _clipped_smoothstep_with_tail(
        roll_error,
        config.wrist_target_roll_clip_low_deg,
        config.wrist_target_roll_clip_high_deg,
        config.wrist_target_roll_tail_slope_per_deg,
    )


def _follow_wrist_target_config(config: MetaRetargetGeneratorConfig) -> MetaRetargetGeneratorConfig:
    """Use stronger wrist-pose continuity regularization during trajectory following."""

    return dataclasses.replace(
        config,
        wrist_target_shape_weight=float(config.follow_wrist_target_shape_weight),
        wrist_target_approach_weight=float(config.follow_wrist_target_approach_weight),
        wrist_target_pos_weight=float(config.follow_wrist_target_pos_weight),
        wrist_target_rot_weight=float(config.follow_wrist_target_rot_weight),
        wrist_target_pos_clip_low_m=float(config.follow_wrist_target_pos_clip_low_m),
        wrist_target_pos_clip_high_m=float(config.follow_wrist_target_pos_clip_high_m),
        wrist_target_distance_tail_slope_per_m=float(config.follow_wrist_target_pos_tail_slope_per_m),
        wrist_target_rot_clip_low_deg=float(config.follow_wrist_target_rot_clip_low_deg),
        wrist_target_rot_clip_high_deg=float(config.follow_wrist_target_rot_clip_high_deg),
        wrist_target_rot_tail_slope_per_deg=float(config.follow_wrist_target_rot_tail_slope_per_deg),
    )


def _candidate_roll_angles(config: MetaRetargetGeneratorConfig) -> np.ndarray:
    samples = max(1, int(config.wrist_target_roll_samples))
    if samples == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(-np.pi, np.pi, samples, endpoint=True, dtype=np.float32)


def _shape_angle_deg_for_pose12(
    area_type: str,
    actual_pose: np.ndarray,
    target_pose: np.ndarray,
    dim_mask12: np.ndarray | None,
) -> float:
    if area_type == "point":
        return 0.0
    axis_actual = _shape_axis_from_pose12(area_type, actual_pose, dim_mask12)
    axis_target = _shape_axis_from_pose12(area_type, target_pose, dim_mask12)
    return _angle_between_unit_vectors_deg(axis_actual, axis_target, unsigned=True)


def _approach_angle_deg_for_pose12(
    actual_pose: np.ndarray,
    target_pose: np.ndarray,
    dim_mask12: np.ndarray | None,
) -> float | None:
    if not (_pose12_has_approach(actual_pose, dim_mask12) and _pose12_has_approach(target_pose, dim_mask12)):
        return None
    return _angle_between_unit_vectors_deg(actual_pose[9:12], target_pose[9:12], unsigned=False)


def _rotation_candidate_score(
    *,
    R_base_wrist: np.ndarray,
    p_base_wrist: np.ndarray,
    R_ref_wrist: np.ndarray,
    p_ref_wrist: np.ndarray,
    shape_angle_deg: float,
    approach_angle_deg: float | None,
    config: MetaRetargetGeneratorConfig,
) -> float:
    shape_loss = _clipped_smoothstep_with_tail(
        shape_angle_deg,
        config.wrist_target_shape_clip_low_deg,
        config.wrist_target_shape_clip_high_deg,
        config.wrist_target_shape_tail_slope_per_deg,
    )
    approach_loss = 0.0
    if approach_angle_deg is not None:
        approach_loss = _clipped_smoothstep_with_tail(
            approach_angle_deg,
            config.wrist_target_approach_clip_low_deg,
            config.wrist_target_approach_clip_high_deg,
            config.wrist_target_approach_tail_slope_per_deg,
        )
    pos_loss = _clipped_smoothstep_with_tail(
        float(np.linalg.norm(np.asarray(p_base_wrist, dtype=np.float32) - np.asarray(p_ref_wrist, dtype=np.float32))),
        config.wrist_target_pos_clip_low_m,
        config.wrist_target_pos_clip_high_m,
        config.wrist_target_distance_tail_slope_per_m,
    )
    rot_loss = _clipped_smoothstep_with_tail(
        _rotation_angle_deg(R_base_wrist, R_ref_wrist),
        config.wrist_target_rot_clip_low_deg,
        config.wrist_target_rot_clip_high_deg,
        config.wrist_target_rot_tail_slope_per_deg,
    )
    roll_loss = _wrist_roll_domain_loss(R_base_wrist, config)
    return float(
        config.wrist_target_shape_weight * shape_loss
        + config.wrist_target_approach_weight * approach_loss
        + config.wrist_target_pos_weight * pos_loss
        + config.wrist_target_rot_weight * rot_loss
        + config.wrist_target_roll_weight * roll_loss
    )


def _make_wrist_pose(R_base_wrist: np.ndarray, p_base_wrist: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R_base_wrist, dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(p_base_wrist, dtype=np.float32).reshape(3)
    return T


def _interpolate_wrist_pose(T_start: np.ndarray, T_end: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(T_start, dtype=np.float32).reshape(4, 4)
    end = np.asarray(T_end, dtype=np.float32).reshape(4, 4)
    t = float(np.clip(alpha, 0.0, 1.0))
    R_delta = end[:3, :3] @ start[:3, :3].T
    R = _rotvec_to_matrix(_matrix_to_rotvec(R_delta) * t) @ start[:3, :3]
    p = (1.0 - t) * start[:3, 3] + t * end[:3, 3]
    return _make_wrist_pose(R, p)


def _target_wrist_pose_from_pose12(
    *,
    area_type: str,
    wrist_pose12: np.ndarray,
    target_pose12: np.ndarray,
    reference_T_base_wrist: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    dim_mask12: np.ndarray | None,
) -> np.ndarray:
    wrist_pose = np.asarray(wrist_pose12, dtype=np.float32).reshape(META12_DIM)
    target_pose = np.asarray(target_pose12, dtype=np.float32).reshape(META12_DIM)
    T_ref = np.asarray(reference_T_base_wrist, dtype=np.float32).reshape(4, 4)
    R_ref = T_ref[:3, :3]
    p_ref = T_ref[:3, 3]

    if area_type == "point":
        R = R_ref.copy()
        return _make_wrist_pose(R, target_pose[:3] - R @ wrist_pose[:3])

    local_axis = _shape_axis_from_pose12(area_type, wrist_pose, dim_mask12)
    target_axis = _shape_axis_from_pose12(area_type, target_pose, dim_mask12)
    local_has_approach = _pose12_has_approach(wrist_pose, dim_mask12)
    target_has_approach = _pose12_has_approach(target_pose, dim_mask12)
    use_approach = local_has_approach and target_has_approach

    base_rotations: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        signed_target_axis = target_axis * sign
        base_rotations.append(_rotation_between_unit_vectors(local_axis, signed_target_axis))
        if use_approach:
            base_rotations.append(
                _rotation_from_direction_pairs(
                    [local_axis, wrist_pose[9:12]],
                    [signed_target_axis, target_pose[9:12]],
                )
            )
    base_rotations.append(R_ref.copy())

    best_T: np.ndarray | None = None
    best_score = np.inf
    roll_axis = target_axis
    roll_angles = _candidate_roll_angles(config)
    for base_R in base_rotations:
        for angle in roll_angles:
            R = _rotvec_to_matrix(roll_axis * float(angle)) @ base_R
            p = target_pose[:3] - R @ wrist_pose[:3]
            T = _make_wrist_pose(R, p)
            actual_pose = _wrist_pose12_to_base_pose(wrist_pose, T)
            shape_angle = _shape_angle_deg_for_pose12(area_type, actual_pose, target_pose, dim_mask12)
            approach_angle = _approach_angle_deg_for_pose12(actual_pose, target_pose, dim_mask12)
            score = _rotation_candidate_score(
                R_base_wrist=R,
                p_base_wrist=p,
                R_ref_wrist=R_ref,
                p_ref_wrist=p_ref,
                shape_angle_deg=shape_angle,
                approach_angle_deg=approach_angle,
                config=config,
            )
            if score < best_score:
                best_score = score
                best_T = T
    if best_T is None:
        return _make_wrist_pose(R_ref, target_pose[:3] - R_ref @ wrist_pose[:3])
    return best_T


def _target_wrist_pose_from_pose6d(
    *,
    area_type: str,
    T_wrist_meta: np.ndarray,
    target_pose6d: np.ndarray,
    reference_T_base_wrist: np.ndarray,
    config: MetaRetargetGeneratorConfig,
) -> np.ndarray:
    T_local = np.asarray(T_wrist_meta, dtype=np.float32).reshape(4, 4)
    target_pose = np.asarray(target_pose6d, dtype=np.float32).reshape(6)
    T_ref = np.asarray(reference_T_base_wrist, dtype=np.float32).reshape(4, 4)
    R_ref = T_ref[:3, :3]
    p_ref = T_ref[:3, 3]
    local_p = T_local[:3, 3]

    if area_type == "point":
        return _make_wrist_pose(R_ref, target_pose[:3] - R_ref @ local_p)

    local_axis = _normalize(T_local[:3, 0] if area_type == "line" else T_local[:3, 2])
    target_axis = _normalize(target_pose[3:6])
    base_rotations = [
        _rotation_between_unit_vectors(local_axis, target_axis),
        _rotation_between_unit_vectors(local_axis, -target_axis),
        R_ref.copy(),
    ]
    best_T: np.ndarray | None = None
    best_score = np.inf
    for base_R in base_rotations:
        for angle in _candidate_roll_angles(config):
            R = _rotvec_to_matrix(target_axis * float(angle)) @ base_R
            p = target_pose[:3] - R @ local_p
            actual_axis = _normalize(R @ local_axis)
            shape_angle = _angle_between_unit_vectors_deg(actual_axis, target_axis, unsigned=True)
            score = _rotation_candidate_score(
                R_base_wrist=R,
                p_base_wrist=p,
                R_ref_wrist=R_ref,
                p_ref_wrist=p_ref,
                shape_angle_deg=shape_angle,
                approach_angle_deg=None,
                config=config,
            )
            if score < best_score:
                best_score = score
                best_T = _make_wrist_pose(R, p)
    if best_T is None:
        return _make_wrist_pose(R_ref, target_pose[:3] - R_ref @ local_p)
    return best_T


def _target_wrist_pose_from_meta_target(
    *,
    area_type: str,
    target_meta_pose: np.ndarray,
    T_wrist_meta: np.ndarray,
    reference_T_base_wrist: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    dim_mask12: np.ndarray | None,
) -> np.ndarray:
    target = np.asarray(target_meta_pose, dtype=np.float32).reshape(-1)
    wrist_meta = np.asarray(T_wrist_meta, dtype=np.float32)
    if target.shape[0] == META12_DIM:
        return _target_wrist_pose_from_pose12(
            area_type=area_type,
            wrist_pose12=wrist_meta.reshape(META12_DIM),
            target_pose12=target,
            reference_T_base_wrist=reference_T_base_wrist,
            config=config,
            dim_mask12=dim_mask12,
        )
    return _target_wrist_pose_from_pose6d(
        area_type=area_type,
        T_wrist_meta=wrist_meta.reshape(4, 4),
        target_pose6d=target.reshape(META6_DIM),
        reference_T_base_wrist=reference_T_base_wrist,
        config=config,
    )


def _apply_pose_offset_from_anchor(
    area_type: str,
    pose: np.ndarray,
    *,
    old_anchor_pose: np.ndarray,
    new_anchor_pose: np.ndarray,
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    shifted = np.asarray(pose, dtype=np.float32).reshape(-1).copy()
    old_anchor_pose = np.asarray(old_anchor_pose, dtype=np.float32).reshape(-1)
    new_anchor_pose = np.asarray(new_anchor_pose, dtype=np.float32).reshape(-1)
    shifted[:3] += new_anchor_pose[:3] - old_anchor_pose[:3]
    if shifted.shape[0] == META12_DIM:
        if area_type != "point":
            start_dirs = [_shape_axis_from_pose12(area_type, old_anchor_pose, dim_mask12)]
            end_dirs = [_shape_axis_from_pose12(area_type, new_anchor_pose, dim_mask12)]
            if _pose12_has_approach(old_anchor_pose, dim_mask12) or _pose12_has_approach(new_anchor_pose, dim_mask12):
                start_dirs.append(old_anchor_pose[9:12])
                end_dirs.append(new_anchor_pose[9:12])
            rot_delta = _rotation_from_direction_pairs(start_dirs, end_dirs)
            shifted[3:9] = _matrix_to_shape6(_project_psd_trace1(rot_delta @ _shape6_to_matrix(shifted[3:9]) @ rot_delta.T))
            if _pose12_has_approach(shifted, dim_mask12):
                shifted[9:12] = _normalize(rot_delta @ shifted[9:12])
        else:
            shifted[3:12] = 0.0
        return shifted.astype(np.float32)

    if area_type != "point":
        rot_delta = _rotation_between_unit_vectors(old_anchor_pose[3:6], new_anchor_pose[3:6])
        shifted[3:6] = _normalize(rot_delta @ _normalize(shifted[3:6]))
    else:
        shifted[3:6] = 0.0
    return shifted.astype(np.float32)


def _build_smooth_chase_targets(
    area_type: str,
    old_target_meta: np.ndarray,
    *,
    old_anchor_pose: np.ndarray,
    new_anchor_pose: np.ndarray,
    trajectory_start_index: int,
    transition_steps: int,
    dim_mask12: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = int(old_target_meta.shape[0])
    source_indices = np.minimum(np.arange(horizon, dtype=np.int32) + int(trajectory_start_index), horizon - 1)
    targets = np.zeros_like(old_target_meta, dtype=np.float32)
    transition = max(1, int(transition_steps))
    for step, source_idx in enumerate(source_indices):
        base_pose = old_target_meta[int(source_idx)]
        shifted_pose = _apply_pose_offset_from_anchor(
            area_type,
            base_pose,
            old_anchor_pose=old_anchor_pose,
            new_anchor_pose=new_anchor_pose,
            dim_mask12=dim_mask12,
        )
        alpha = min(float(step + 1) / float(transition), 1.0)
        targets[step] = _interpolate_meta_pose(area_type, shifted_pose, base_pose, alpha, dim_mask12=dim_mask12)
    return targets.astype(np.float32), source_indices


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _correction_path_pose(
    area_type: str,
    start_pose: np.ndarray,
    join_pose: np.ndarray,
    alpha: float,
    *,
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    mid_pose = _interpolate_meta_pose(area_type, start_pose, join_pose, 0.5, dim_mask12=dim_mask12)
    eased = _smoothstep(alpha)
    if eased <= 0.5:
        return _interpolate_meta_pose(area_type, start_pose, mid_pose, eased / 0.5, dim_mask12=dim_mask12)
    return _interpolate_meta_pose(area_type, mid_pose, join_pose, (eased - 0.5) / 0.5, dim_mask12=dim_mask12)


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
    has_pose12 = "pose12d" in meta_areas
    meta_dim = META12_DIM if has_pose12 else META6_DIM
    action_meta_slice = _meta_slice_for_dim(meta_dim)
    action_camera_slice = _camera_slice_for_dim(meta_dim)
    meta_targets = data.get("meta_action_targets")
    if has_pose12:
        meta_area_pose12d = np.asarray(meta_areas["pose12d"], dtype=np.float32).copy()
        raw_dim_masks = meta_areas.get("dim_mask12")
        meta_area_dim_mask12 = (
            np.asarray(raw_dim_masks, dtype=bool).copy()
            if raw_dim_masks is not None
            else np.ones(meta_area_pose12d.shape, dtype=bool)
        )
        meta_area_pose6d = np.zeros((meta_area_pose12d.shape[0], 6), dtype=np.float32)
    else:
        meta_area_pose12d = None
        meta_area_dim_mask12 = None
        meta_area_pose6d = np.asarray(meta_areas["pose6d"], dtype=np.float32).copy()
    meta_area_type = np.asarray(meta_areas["type"], dtype=np.int32).copy()
    meta_area_mask = np.asarray(meta_areas["mask"], dtype=bool).copy()

    if actions.ndim != 2 or actions.shape[0] == 0:
        return None
    if state.shape[-1] < 14 or actions.shape[-1] < 14:
        return None
    if state.shape[-1] >= STATE_META_SLICE.stop:
        state[STATE_META_SLICE] = 0.0
    if (meta_area_pose12d.shape[0] if has_pose12 else meta_area_pose6d.shape[0]) == 0 or not bool(meta_area_mask[0]):
        return None

    horizon = int(actions.shape[0])
    state_qpos = state[:14].copy()
    T_state_wrist = np.asarray(helpers["fk"](state_qpos, "right_wrist"), dtype=np.float32)
    dim_mask12 = meta_area_dim_mask12[0].copy() if meta_area_dim_mask12 is not None else None
    fixed_input_pose12d = data.get("_retarget_fixed_input_pose12d")
    fixed_dim_mask12 = None
    if fixed_input_pose12d is not None:
        if not has_pose12:
            return None
        fixed_area_type_id = int(data.get("_retarget_fixed_area_type_id", int(meta_area_type[0])))
        meta_area_type[0] = fixed_area_type_id
        fixed_dim_mask12 = data.get("_retarget_fixed_dim_mask12")
        if fixed_dim_mask12 is not None and meta_area_dim_mask12 is not None:
            meta_area_dim_mask12[0] = np.asarray(fixed_dim_mask12, dtype=bool).reshape(META12_DIM)
            dim_mask12 = meta_area_dim_mask12[0].copy()
    area_type = META_AREA_ID_TO_TYPE.get(int(meta_area_type[0]), "line")
    if has_pose12:
        if actions.shape[-1] >= ACTION_META_SLICE.stop:
            actions[..., ACTION_META_SLICE] = 0.0
        old_input_pose = meta_area_pose12d[0].copy()
        meta_area_pose6d[0] = _pose12_to_pose6d(area_type, old_input_pose, dim_mask12)
        if meta_targets is None or "pose12d" not in meta_targets:
            return None
        target_pose12d = np.asarray(meta_targets["pose12d"], dtype=np.float32).copy()
        if target_pose12d.ndim == 3:
            old_target_meta = target_pose12d[:, 0, :].copy()
        elif target_pose12d.ndim == 2:
            old_target_meta = target_pose12d.copy()
            target_pose12d = target_pose12d[:, None, :]
        else:
            raise ValueError(f"Expected meta_action_targets.pose12d with 2 or 3 dims, got {target_pose12d.shape}")
        raw_target_dim_mask12 = meta_targets.get("dim_mask12")
        if raw_target_dim_mask12 is None:
            meta_action_target_dim_mask12 = np.ones(target_pose12d.shape, dtype=bool)
        else:
            meta_action_target_dim_mask12 = np.asarray(raw_target_dim_mask12, dtype=bool).copy()
            if meta_action_target_dim_mask12.ndim == 2:
                meta_action_target_dim_mask12 = meta_action_target_dim_mask12[:, None, :]
        raw_target_mask = meta_targets.get("mask")
        if raw_target_mask is None:
            meta_action_target_mask = np.ones(target_pose12d.shape[:2], dtype=bool)
        else:
            meta_action_target_mask = np.asarray(raw_target_mask, dtype=bool).copy()
            if meta_action_target_mask.ndim == 3 and meta_action_target_mask.shape[-1] == 1:
                meta_action_target_mask = np.squeeze(meta_action_target_mask, axis=-1)
            if meta_action_target_mask.ndim == 1:
                meta_action_target_mask = meta_action_target_mask[:, None]
        if fixed_dim_mask12 is not None:
            effective = np.asarray(fixed_dim_mask12, dtype=bool).reshape(META12_DIM)
            meta_action_target_dim_mask12[:, 0, :] = effective
    else:
        old_input_pose = meta_area_pose6d[0].copy()
        if actions.shape[-1] < action_meta_slice.stop:
            return None
        old_target_meta = actions[:, action_meta_slice].copy()
        target_pose12d = None
        meta_action_target_dim_mask12 = None
        meta_action_target_mask = None

    retarget_mode = data.get("_retarget_mode")
    if retarget_mode is None:
        retarget_mode = "future_near" if rng.random() < config.future_near_mode_prob else "correction"
    if retarget_mode not in ("future_near", "correction"):
        raise ValueError(f"Unsupported retarget mode: {retarget_mode!r}")
    trajectory_start_index = 0
    target_meta_sequence: np.ndarray | None = None
    source_indices: np.ndarray | None = None

    if fixed_input_pose12d is not None:
        new_input_pose = np.asarray(fixed_input_pose12d, dtype=np.float32).reshape(META12_DIM)
        selected = _select_pair_retarget_anchor(
            area_type,
            new_input_pose,
            old_input_pose,
            old_target_meta,
            config=config,
            dim_mask12=dim_mask12,
        )
        if selected is None:
            return None
        retarget_mode = selected["retarget_mode"]
        trajectory_start_index = int(selected["trajectory_start_index"])
        target_meta_sequence = selected["target_meta_sequence"]
        source_indices = selected["source_indices"]
    elif retarget_mode == "future_near":
        max_frame = min(max(int(config.future_near_window_frames) - 1, 0), horizon - 1)
        selected_frame_index = int(rng.integers(max_frame + 1))
        selected_pose = _trajectory_frame_pose(old_input_pose, old_target_meta, selected_frame_index)
        new_input_pose = _sample_pose_near_trajectory_frame(
            area_type,
            selected_pose,
            rng=rng,
            position_noise_max_m=config.future_near_position_noise_max_m,
            direction_noise_max_deg=config.future_near_direction_noise_max_deg,
            dim_mask12=dim_mask12,
        )
        trajectory_start_index = min(selected_frame_index, horizon - 1)
        source_indices = _direct_follow_source_indices(old_target_meta, trajectory_start_index=trajectory_start_index)
        target_meta_sequence = old_target_meta[source_indices].astype(np.float32)
    else:
        new_input_pose = _sample_correction_pose(
            area_type,
            old_input_pose,
            old_target_meta,
            rng=rng,
            config=config,
            dim_mask12=dim_mask12,
        )
        if new_input_pose is None:
            return None

    T_wrist_meta_new = (
        _base_pose12_to_wrist_pose(new_input_pose, T_state_wrist)
        if has_pose12
        else _base_meta_pose_to_wrist_transform(area_type, new_input_pose, T_state_wrist)
    )

    return {
        "state": state,
        "actions": actions,
        "meta_area_pose6d": meta_area_pose6d,
        "meta_area_pose12d": meta_area_pose12d,
        "meta_area_dim_mask12": meta_area_dim_mask12,
        "meta_action_target_pose12d": target_pose12d,
        "meta_action_target_dim_mask12": meta_action_target_dim_mask12,
        "meta_action_target_mask": meta_action_target_mask,
        "meta_area_type": meta_area_type,
        "meta_area_mask": meta_area_mask,
        "meta_dim": meta_dim,
        "action_meta_slice": action_meta_slice,
        "action_camera_slice": action_camera_slice,
        "horizon": horizon,
        "area_type": area_type,
        "dim_mask12": dim_mask12,
        "state_qpos": state_qpos,
        "T_wrist_meta_new": T_wrist_meta_new,
        "new_input_pose": new_input_pose,
        "old_target_meta": old_target_meta,
        "target_meta_sequence": target_meta_sequence,
        "source_indices": source_indices,
        "retarget_mode": retarget_mode,
        "trajectory_start_index": trajectory_start_index,
        "initial_seed_sim": helpers["real_to_sim_arm"](state_qpos[RIGHT_ARM_QPOS_SLICE], "right"),
    }


def _solve_smooth_chase_sequence(
    item: dict[str, Any],
    *,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target_meta_sequence = np.asarray(item["target_meta_sequence"], dtype=np.float32)
    if config.ik_backend == "jax":
        solved = _solve_right_arm_meta_ik_sequence_jax(
            target_meta_poses=target_meta_sequence,
            T_wrist_meta=item["T_wrist_meta_new"],
            area_type=item["area_type"],
            initial_qpos_sim=initial_qpos_sim,
            config=config,
        )
        return solved["qpos_sim"], solved["converged"]

    if config.ik_backend != "numpy":
        raise ValueError(f"Unsupported ik_backend={config.ik_backend!r}; expected 'numpy' or 'jax'")

    seed_sim = np.asarray(initial_qpos_sim, dtype=np.float32)
    previous_real_qpos = np.asarray(item["state_qpos"], dtype=np.float32).reshape(-1)[:14].copy()
    follow_config = _follow_wrist_target_config(config)
    qpos_sequence = []
    converged_sequence = []
    for step, target_meta in enumerate(target_meta_sequence):
        source_idx = int(item["source_indices"][step])
        ik_result = _solve_right_arm_meta_ik_numpy(
            template_qpos_real=previous_real_qpos,
            target_meta_pose=target_meta,
            T_wrist_meta=item["T_wrist_meta_new"],
            area_type=item["area_type"],
            initial_qpos_sim=seed_sim,
            config=follow_config,
            helpers=helpers,
            dim_mask12=item.get("dim_mask12"),
        )
        seed_sim = _wrap_near_seed(ik_result["qpos_sim"], seed_sim)
        solved_real = _sim_to_real_right_arm_qpos(seed_sim, helpers)
        previous_real_qpos = np.asarray(item["actions"][source_idx, :14], dtype=np.float32).copy()
        previous_real_qpos[RIGHT_ARM_QPOS_SLICE] = solved_real
        qpos_sequence.append(seed_sim)
        converged_sequence.append(bool(ik_result["converged"]))

    return np.asarray(qpos_sequence, dtype=np.float32), np.asarray(converged_sequence, dtype=bool)


def _assemble_smooth_chase_result(
    item: dict[str, Any],
    *,
    qpos_sim: np.ndarray,
    converged: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> MetaRetargetResult:
    state = item["state"].copy()
    actions = item["actions"]
    meta_area_pose6d = item["meta_area_pose6d"].copy()
    meta_area_pose12d = None if item["meta_area_pose12d"] is None else item["meta_area_pose12d"].copy()
    meta_area_dim_mask12 = None if item["meta_area_dim_mask12"] is None else item["meta_area_dim_mask12"].copy()
    out_target_pose12d = None if item["meta_action_target_pose12d"] is None else item["meta_action_target_pose12d"].copy()
    out_target_dim_mask12 = (
        None if item["meta_action_target_dim_mask12"] is None else item["meta_action_target_dim_mask12"].copy()
    )
    out_target_mask = None if item["meta_action_target_mask"] is None else item["meta_action_target_mask"].copy()
    meta_area_type = item["meta_area_type"].copy()
    meta_area_mask = item["meta_area_mask"].copy()
    meta_dim = int(item.get("meta_dim", META6_DIM))
    action_meta_slice = item.get("action_meta_slice", _meta_slice_for_dim(meta_dim))
    action_camera_slice = item.get("action_camera_slice", _camera_slice_for_dim(meta_dim))
    dim_mask12 = item.get("dim_mask12")
    horizon = int(item["horizon"])
    area_type = item["area_type"]
    state_qpos = item["state_qpos"]
    T_wrist_meta_new = item["T_wrist_meta_new"]
    target_meta_sequence = np.asarray(item["target_meta_sequence"], dtype=np.float32)
    source_indices = np.asarray(item["source_indices"], dtype=np.int32)

    out_actions = actions.copy()
    previous_real_qpos = state_qpos.copy()
    ik_nonconverged = 0
    position_errors: list[float] = []
    direction_errors: list[float] = []
    joint_step_deltas: list[float] = []
    T_wrist_cam = _camera_pose_to_transform(state[action_camera_slice]) if state.shape[-1] >= action_camera_slice.stop else None

    for step in range(horizon):
        source_idx = int(source_indices[step])
        out_action = actions[source_idx].copy()
        solved_real = _sim_to_real_right_arm_qpos(qpos_sim[step], helpers)
        target_meta = target_meta_sequence[step]

        out_action[RIGHT_ARM_QPOS_SLICE] = solved_real
        _set_retarget_meta_action(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim)
        if not bool(converged[step]):
            ik_nonconverged += 1

        if config.recompute_action_camera_pose and T_wrist_cam is not None and out_action.shape[-1] >= action_camera_slice.stop:
            T_action_wrist = np.asarray(helpers["fk"](out_action[:14], "right_wrist"), dtype=np.float32)
            out_action[action_camera_slice] = _transform_to_camera_pose(T_action_wrist @ T_wrist_cam)

        out_actions[step] = out_action
        pos_err, dir_err = _meta_tracking_errors(
            area_type=area_type,
            qpos_real=out_action[:14],
            T_wrist_meta=T_wrist_meta_new,
            target_meta_pose=target_meta,
            dim_mask12=dim_mask12,
            fk_fn=helpers["fk"],
        )
        position_errors.append(pos_err)
        direction_errors.append(dir_err)
        joint_step_deltas.append(float(np.max(np.abs(out_action[:14] - previous_real_qpos[:14]))))
        previous_real_qpos = out_action[:14].copy()

    if meta_area_pose12d is not None:
        meta_area_pose12d[0] = item["new_input_pose"]
        meta_area_pose6d[0] = _pose12_to_pose6d(area_type, item["new_input_pose"], dim_mask12)
    else:
        meta_area_pose6d[0] = item["new_input_pose"]
    max_position_error = float(max(position_errors, default=np.inf))
    max_direction_error = float(max(direction_errors, default=0.0))
    max_joint_delta = float(max(joint_step_deltas, default=0.0))
    (
        values_ok,
        actions_finite,
        max_abs_action,
        max_camera_rotvec_norm,
        value_reason,
    ) = _action_value_diagnostics(out_actions, config, meta_dim=meta_dim)
    accepted = (
        max_position_error <= config.accept_max_position_error_m
        and max_direction_error <= config.accept_max_direction_error_rad
        and max_joint_delta <= config.accept_max_step_joint_delta_rad
        and values_ok
    )
    reason = ""
    if not accepted:
        reason_parts = [
            f"ik_nonconverged={ik_nonconverged}",
            f"max_position_error_m={max_position_error:.6f}",
            f"max_direction_error_rad={max_direction_error:.6f}",
            f"max_step_joint_delta_rad={max_joint_delta:.6f}",
        ]
        if value_reason:
            reason_parts.append(value_reason)
        reason = ", ".join(reason_parts)

    diagnostics = MetaRetargetDiagnostics(
        accepted=accepted,
        area_type=area_type,
        horizon=horizon,
        approach_steps=0,
        ik_nonconverged=ik_nonconverged,
        max_position_error_m=max_position_error,
        max_direction_error_rad=max_direction_error,
        max_step_joint_delta_rad=max_joint_delta,
        reason=reason,
        actions_finite=actions_finite,
        max_abs_action_value=max_abs_action,
        max_camera_rotvec_norm_rad=max_camera_rotvec_norm,
        retarget_mode=item["retarget_mode"],
        trajectory_start_index=int(item["trajectory_start_index"]),
    )
    return MetaRetargetResult(
        state=state.astype(np.float32),
        actions=out_actions.astype(np.float32),
        meta_area_pose6d=meta_area_pose6d.astype(np.float32),
        meta_area_type=meta_area_type.astype(np.int32),
        meta_area_mask=meta_area_mask.astype(bool),
        diagnostics=diagnostics,
        meta_area_pose12d=None if meta_area_pose12d is None else meta_area_pose12d.astype(np.float32),
        meta_area_dim_mask12=None if meta_area_dim_mask12 is None else meta_area_dim_mask12.astype(bool),
        meta_action_target_pose12d=None if out_target_pose12d is None else out_target_pose12d.astype(np.float32),
        meta_action_target_dim_mask12=None if out_target_dim_mask12 is None else out_target_dim_mask12.astype(bool),
        meta_action_target_mask=None if out_target_mask is None else out_target_mask.astype(bool),
    )


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
    meta_area_pose12d = None if item["meta_area_pose12d"] is None else item["meta_area_pose12d"].copy()
    meta_area_dim_mask12 = None if item["meta_area_dim_mask12"] is None else item["meta_area_dim_mask12"].copy()
    out_target_pose12d = None if item["meta_action_target_pose12d"] is None else item["meta_action_target_pose12d"].copy()
    out_target_dim_mask12 = (
        None if item["meta_action_target_dim_mask12"] is None else item["meta_action_target_dim_mask12"].copy()
    )
    out_target_mask = None if item["meta_action_target_mask"] is None else item["meta_action_target_mask"].copy()
    meta_area_type = item["meta_area_type"].copy()
    meta_area_mask = item["meta_area_mask"].copy()
    meta_dim = int(item.get("meta_dim", META6_DIM))
    action_meta_slice = item.get("action_meta_slice", _meta_slice_for_dim(meta_dim))
    action_camera_slice = item.get("action_camera_slice", _camera_slice_for_dim(meta_dim))
    dim_mask12 = item.get("dim_mask12")
    horizon = int(item["horizon"])
    area_type = item["area_type"]
    state_qpos = item["state_qpos"]
    T_wrist_meta_new = item["T_wrist_meta_new"]
    new_input_pose = item["new_input_pose"]
    old_target_meta = item["old_target_meta"]
    trajectory_start_index = min(max(int(item.get("trajectory_start_index", 0)), 0), horizon - 1)
    resume_source_index = min(max(int(config.correction_resume_source_index), 1), horizon - 1)
    join_source_index = min(trajectory_start_index + max(resume_source_index - 1, 0), horizon - 1)
    join_target_meta = old_target_meta[join_source_index]

    approach_steps = approach_plan.steps
    out_actions = actions.copy()
    previous_real_qpos = state_qpos.copy()
    ik_nonconverged = 0 if approach_plan.converged else 1
    position_errors: list[float] = []
    direction_errors: list[float] = []
    joint_step_deltas: list[float] = []
    T_wrist_cam = _camera_pose_to_transform(state[action_camera_slice]) if state.shape[-1] >= action_camera_slice.stop else None
    align_pos_err, align_dir_err = _meta_tracking_errors(
        area_type=area_type,
        qpos_real=approach_plan.aligned_qpos_real,
        T_wrist_meta=T_wrist_meta_new,
        target_meta_pose=join_target_meta,
        dim_mask12=dim_mask12,
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
            target_meta = _correction_path_pose(
                area_type,
                new_input_pose,
                join_target_meta,
                alpha,
                dim_mask12=dim_mask12,
            )
            _set_retarget_meta_action(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim)
        else:
            source_idx = min(step - approach_steps + trajectory_start_index + resume_source_index, horizon - 1)
            out_action = actions[source_idx].copy()
            target_meta = old_target_meta[source_idx]
            solved_real = _sim_to_real_right_arm_qpos(follow_qpos_sim[source_idx], helpers)

            out_action[RIGHT_ARM_QPOS_SLICE] = solved_real
            _set_retarget_meta_action(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim)
            if not bool(follow_converged[source_idx]):
                ik_nonconverged += 1

        if config.recompute_action_camera_pose and T_wrist_cam is not None and out_action.shape[-1] >= action_camera_slice.stop:
            T_action_wrist = np.asarray(helpers["fk"](out_action[:14], "right_wrist"), dtype=np.float32)
            out_action[action_camera_slice] = _transform_to_camera_pose(T_action_wrist @ T_wrist_cam)

        out_actions[step] = out_action

        pos_err, dir_err = _meta_tracking_errors(
            area_type=area_type,
            qpos_real=out_action[:14],
            T_wrist_meta=T_wrist_meta_new,
            target_meta_pose=_retarget_meta_for_error(out_action, out_target_pose12d, step, target_meta, action_meta_slice, meta_dim),
            dim_mask12=dim_mask12,
            fk_fn=helpers["fk"],
        )
        position_errors.append(pos_err)
        direction_errors.append(dir_err)
        joint_step_deltas.append(float(np.max(np.abs(out_action[:14] - previous_real_qpos[:14]))))
        previous_real_qpos = out_action[:14].copy()

    if meta_area_pose12d is not None:
        meta_area_pose12d[0] = new_input_pose
        meta_area_pose6d[0] = _pose12_to_pose6d(area_type, new_input_pose, dim_mask12)
    else:
        meta_area_pose6d[0] = new_input_pose
    max_position_error = float(max(position_errors, default=np.inf))
    max_direction_error = float(max(direction_errors, default=0.0))
    max_joint_delta = float(max(joint_step_deltas, default=0.0))
    (
        values_ok,
        actions_finite,
        max_abs_action,
        max_camera_rotvec_norm,
        value_reason,
    ) = _action_value_diagnostics(out_actions, config, meta_dim=meta_dim)
    accepted = (
        max_position_error <= config.accept_max_position_error_m
        and max_direction_error <= config.accept_max_direction_error_rad
        and max_joint_delta <= config.accept_max_step_joint_delta_rad
        and values_ok
    )
    reason = ""
    if not accepted:
        reason_parts = [
            f"ik_nonconverged={ik_nonconverged}",
            f"max_position_error_m={max_position_error:.6f}",
            f"max_direction_error_rad={max_direction_error:.6f}",
            f"max_step_joint_delta_rad={max_joint_delta:.6f}",
        ]
        if value_reason:
            reason_parts.append(value_reason)
        reason = ", ".join(reason_parts)

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
        actions_finite=actions_finite,
        max_abs_action_value=max_abs_action,
        max_camera_rotvec_norm_rad=max_camera_rotvec_norm,
        retarget_mode=item["retarget_mode"],
        trajectory_start_index=int(item["trajectory_start_index"]),
    )
    return MetaRetargetResult(
        state=state.astype(np.float32),
        actions=out_actions.astype(np.float32),
        meta_area_pose6d=meta_area_pose6d.astype(np.float32),
        meta_area_type=meta_area_type.astype(np.int32),
        meta_area_mask=meta_area_mask.astype(bool),
        diagnostics=diagnostics,
        meta_area_pose12d=None if meta_area_pose12d is None else meta_area_pose12d.astype(np.float32),
        meta_area_dim_mask12=None if meta_area_dim_mask12 is None else meta_area_dim_mask12.astype(bool),
        meta_action_target_pose12d=None if out_target_pose12d is None else out_target_pose12d.astype(np.float32),
        meta_action_target_dim_mask12=None if out_target_dim_mask12 is None else out_target_dim_mask12.astype(bool),
        meta_action_target_mask=None if out_target_mask is None else out_target_mask.astype(bool),
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


def _shape6_to_matrix(shape6: np.ndarray) -> np.ndarray:
    s = np.asarray(shape6, dtype=np.float32).reshape(6)
    return np.array(
        [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]],
        dtype=np.float32,
    )


def _matrix_to_shape6(matrix: np.ndarray) -> np.ndarray:
    M = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    M = 0.5 * (M + M.T)
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], dtype=np.float32)


def _project_psd_trace1(matrix: np.ndarray) -> np.ndarray:
    M = 0.5 * (np.asarray(matrix, dtype=np.float32).reshape(3, 3) + np.asarray(matrix, dtype=np.float32).reshape(3, 3).T)
    eigvals, eigvecs = np.linalg.eigh(M.astype(np.float64))
    eigvals = np.maximum(eigvals, 0.0)
    total = float(np.sum(eigvals))
    if total < 1e-8:
        return (np.eye(3, dtype=np.float32) / 3.0).astype(np.float32)
    eigvals = eigvals / total
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float32)


def _shape_matrix_from_axis(area_type: str, axis: np.ndarray) -> np.ndarray:
    axis = _normalize(axis)
    if area_type == "surface":
        return (0.5 * (np.eye(3, dtype=np.float32) - np.outer(axis, axis))).astype(np.float32)
    if area_type == "point":
        return (np.eye(3, dtype=np.float32) / 3.0).astype(np.float32)
    return np.outer(axis, axis).astype(np.float32)


def _pose12_has_approach(pose12: np.ndarray, dim_mask12: np.ndarray | None = None) -> bool:
    pose = np.asarray(pose12, dtype=np.float32).reshape(12)
    if np.linalg.norm(pose[9:12]) < 1e-6:
        return False
    if dim_mask12 is None:
        return True
    return bool(np.any(np.asarray(dim_mask12, dtype=bool).reshape(12)[9:12]))


def _pose12_payload_dim_mask(payload: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(payload, dict) or "pose12d" not in payload:
        return None
    raw_mask = payload.get("dim_mask12")
    if raw_mask is None:
        return np.ones(META12_DIM, dtype=bool)
    mask = np.asarray(raw_mask, dtype=bool)
    if mask.shape[-1] != META12_DIM:
        return None
    return np.any(mask.reshape(-1, META12_DIM), axis=0).astype(bool)


def _pose12_payload_has_approach(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict) or "pose12d" not in payload:
        return False
    poses = np.asarray(payload["pose12d"], dtype=np.float32)
    if poses.shape[-1] != META12_DIM:
        return False
    poses = poses.reshape(-1, META12_DIM)
    raw_mask = payload.get("dim_mask12")
    if raw_mask is None:
        masks = np.ones_like(poses, dtype=bool)
    else:
        masks = np.asarray(raw_mask, dtype=bool).reshape(-1, META12_DIM)
        if masks.shape[0] == 1 and poses.shape[0] > 1:
            masks = np.broadcast_to(masks, poses.shape)
        elif masks.shape[0] != poses.shape[0]:
            masks = np.broadcast_to(np.any(masks, axis=0, keepdims=True), poses.shape)
    return any(_pose12_has_approach(pose, mask) for pose, mask in zip(poses, masks, strict=True))


def _pose12_effective_dim_mask(mask: np.ndarray | None, *, has_approach: bool) -> np.ndarray:
    effective = np.ones(META12_DIM, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(META12_DIM).copy()
    effective[:3] = True
    if not has_approach:
        effective[9:12] = False
    return effective


def _shape_axis_from_pose12(area_type: str, pose12: np.ndarray, dim_mask12: np.ndarray | None = None) -> np.ndarray:
    pose = np.asarray(pose12, dtype=np.float32).reshape(12)
    M = _project_psd_trace1(_shape6_to_matrix(pose[3:9]))
    eigvals, eigvecs = np.linalg.eigh(M.astype(np.float64))
    axis = eigvecs[:, int(np.argmin(eigvals) if area_type == "surface" else np.argmax(eigvals))].astype(np.float32)
    if _pose12_has_approach(pose, dim_mask12) and float(np.dot(axis, pose[9:12])) < 0.0:
        axis = -axis
    return _normalize(axis)


def _transform_pose12(R_to_from: np.ndarray, t_to_from: np.ndarray, pose12: np.ndarray) -> np.ndarray:
    R = np.asarray(R_to_from, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t_to_from, dtype=np.float32).reshape(3)
    pose = np.asarray(pose12, dtype=np.float32).reshape(12)
    out = np.zeros(12, dtype=np.float32)
    out[:3] = R @ pose[:3] + t
    out[3:9] = _matrix_to_shape6(_project_psd_trace1(R @ _shape6_to_matrix(pose[3:9]) @ R.T))
    out[9:12] = R @ pose[9:12]
    return out


def _base_pose12_to_wrist_pose(pose12_base: np.ndarray, T_base_wrist: np.ndarray) -> np.ndarray:
    T = np.asarray(T_base_wrist, dtype=np.float32).reshape(4, 4)
    R = T[:3, :3].T
    t = -(R @ T[:3, 3])
    return _transform_pose12(R, t, pose12_base)


def _wrist_pose12_to_base_pose(pose12_wrist: np.ndarray, T_base_wrist: np.ndarray) -> np.ndarray:
    T = np.asarray(T_base_wrist, dtype=np.float32).reshape(4, 4)
    return _transform_pose12(T[:3, :3], T[:3, 3], pose12_wrist)


def _pair_source_affordance_as_target_input_pose(
    source_data: dict[str, Any],
    target_data: dict[str, Any],
    *,
    helpers: dict[str, Any],
) -> dict[str, Any] | None:
    source_meta = source_data.get("meta_areas")
    target_state = np.asarray(target_data.get("state", []), dtype=np.float32)
    source_state = np.asarray(source_data.get("state", []), dtype=np.float32)
    if not isinstance(source_meta, dict) or "pose12d" not in source_meta:
        return None
    if source_state.shape[-1] < 14 or target_state.shape[-1] < 14:
        return None

    source_mask = np.asarray(source_meta.get("mask", []), dtype=bool).reshape(-1)
    if source_mask.size == 0 or not bool(source_mask[0]):
        return None
    source_pose12d = np.asarray(source_meta["pose12d"], dtype=np.float32)
    if source_pose12d.ndim != 2 or source_pose12d.shape[0] == 0 or source_pose12d.shape[1] != META12_DIM:
        return None

    source_type = np.asarray(
        source_meta.get("type", np.asarray([META_AREA_TYPE_TO_ID["line"]], dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    area_type_id = int(source_type[0]) if source_type.size else META_AREA_TYPE_TO_ID["line"]
    raw_dim_mask = source_meta.get("dim_mask12")
    if raw_dim_mask is None:
        dim_mask12 = np.ones(META12_DIM, dtype=bool)
    else:
        dim_mask12 = np.asarray(raw_dim_mask, dtype=bool).reshape(-1, META12_DIM)[0].copy()
    source_has_approach = _pose12_has_approach(source_pose12d[0], dim_mask12)
    target_has_approach = _pose12_payload_has_approach(target_data.get("meta_action_targets")) or _pose12_payload_has_approach(
        target_data.get("meta_areas")
    )
    if source_has_approach and not target_has_approach:
        return None
    target_dim_mask = _pose12_payload_dim_mask(target_data.get("meta_action_targets"))
    if target_dim_mask is None:
        target_dim_mask = _pose12_payload_dim_mask(target_data.get("meta_areas"))
    source_effective_mask = _pose12_effective_dim_mask(dim_mask12, has_approach=source_has_approach)
    target_effective_mask = _pose12_effective_dim_mask(target_dim_mask, has_approach=target_has_approach)
    dim_mask12 = (source_effective_mask & target_effective_mask).astype(bool)

    T_source_wrist = np.asarray(helpers["fk"](source_state[:14], "right_wrist"), dtype=np.float32)
    T_target_wrist = np.asarray(helpers["fk"](target_state[:14], "right_wrist"), dtype=np.float32)
    wrist_pose12d = _base_pose12_to_wrist_pose(source_pose12d[0], T_source_wrist)
    target_input_pose12d = _wrist_pose12_to_base_pose(wrist_pose12d, T_target_wrist)
    return {
        "input_pose12d": target_input_pose12d.astype(np.float32),
        "area_type_id": int(area_type_id),
        "dim_mask12": dim_mask12.astype(bool),
    }


def _pose12_to_pose6d(area_type: str, pose12: np.ndarray, dim_mask12: np.ndarray | None = None) -> np.ndarray:
    pose = np.asarray(pose12, dtype=np.float32).reshape(12)
    if area_type == "point":
        tail = np.zeros(3, dtype=np.float32)
    else:
        tail = _shape_axis_from_pose12(area_type, pose, dim_mask12)
    return np.concatenate([pose[:3], tail], axis=0).astype(np.float32)


def _rotation_from_direction_pairs(start_dirs: list[np.ndarray], end_dirs: list[np.ndarray]) -> np.ndarray:
    starts = []
    ends = []
    for start, end in zip(start_dirs, end_dirs, strict=True):
        s = _normalize(start)
        e = _normalize(end)
        if float(np.linalg.norm(s)) > 1e-6 and float(np.linalg.norm(e)) > 1e-6:
            starts.append(s)
            ends.append(e)
    if not starts:
        return np.eye(3, dtype=np.float32)
    A = np.stack(starts, axis=1).astype(np.float64)
    B = np.stack(ends, axis=1).astype(np.float64)
    U, _, Vt = np.linalg.svd(B @ A.T)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R.astype(np.float32)


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


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rot_matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(rot_matrix)):
        return np.full(3, np.nan, dtype=np.float32)

    # Project through SO(3) first. FK products are close to rotations but are
    # often float32; the direct axis formula divides by sin(theta), which is
    # singular near 180 degrees and can turn tiny skew noise into huge rotvecs.
    u, _, vt = np.linalg.svd(rot_matrix)
    rot_matrix = u @ vt
    if np.linalg.det(rot_matrix) < 0:
        u[:, -1] *= -1.0
        rot_matrix = u @ vt

    trace = float(np.trace(rot_matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(max(trace + 1.0, 0.0))
        quat = np.array(
            [
                0.25 * scale,
                (rot_matrix[2, 1] - rot_matrix[1, 2]) / scale,
                (rot_matrix[0, 2] - rot_matrix[2, 0]) / scale,
                (rot_matrix[1, 0] - rot_matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    elif rot_matrix[0, 0] >= rot_matrix[1, 1] and rot_matrix[0, 0] >= rot_matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2], 0.0))
        quat = np.array(
            [
                (rot_matrix[2, 1] - rot_matrix[1, 2]) / scale,
                0.25 * scale,
                (rot_matrix[0, 1] + rot_matrix[1, 0]) / scale,
                (rot_matrix[0, 2] + rot_matrix[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    elif rot_matrix[1, 1] >= rot_matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2], 0.0))
        quat = np.array(
            [
                (rot_matrix[0, 2] - rot_matrix[2, 0]) / scale,
                (rot_matrix[0, 1] + rot_matrix[1, 0]) / scale,
                0.25 * scale,
                (rot_matrix[1, 2] + rot_matrix[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = 2.0 * np.sqrt(max(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1], 0.0))
        quat = np.array(
            [
                (rot_matrix[1, 0] - rot_matrix[0, 1]) / scale,
                (rot_matrix[0, 2] + rot_matrix[2, 0]) / scale,
                (rot_matrix[1, 2] + rot_matrix[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )

    quat_norm = float(np.linalg.norm(quat))
    if quat_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quat /= quat_norm
    if quat[0] < 0.0:
        quat *= -1.0

    vector = quat[1:4]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(quat[0]))
    return (vector / vector_norm * angle).astype(np.float32)


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
    dim_mask12: np.ndarray | None = None,
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

    state_qpos = np.asarray(state_qpos, dtype=np.float32).reshape(-1)[:14].copy()
    start_T_base_wrist = np.asarray(helpers["fk"](state_qpos, "right_wrist"), dtype=np.float32)
    target_T_base_wrist = _target_wrist_pose_from_meta_target(
        area_type=area_type,
        target_meta_pose=first_target_meta,
        T_wrist_meta=T_wrist_meta,
        reference_T_base_wrist=start_T_base_wrist,
        config=config,
        dim_mask12=dim_mask12,
    )
    alignment = _solve_right_arm_wrist_pose_ik_numpy(
        template_qpos_real=state_qpos,
        target_T_base_wrist=target_T_base_wrist,
        initial_qpos_sim=initial_qpos_sim,
        config=config,
        helpers=helpers,
    )
    solved_sim = _wrap_near_seed(alignment["qpos_sim"], initial_qpos_sim)
    solved_real = _sim_to_real_right_arm_qpos(solved_sim, helpers)
    aligned_qpos_real = state_qpos.copy()
    aligned_qpos_real[RIGHT_ARM_QPOS_SLICE] = solved_real
    right_delta = np.asarray(solved_real, dtype=np.float32) - np.asarray(state_qpos[RIGHT_ARM_QPOS_SLICE], dtype=np.float32)
    max_abs_delta = float(np.max(np.abs(right_delta)))
    steps = int(np.ceil(max_abs_delta / float(config.approach_joint_step_rad)))
    if config.max_approach_steps is not None:
        steps = min(steps, int(config.max_approach_steps))
    steps = min(max(steps, 0), int(horizon))
    qpos_sequence = _build_approach_qpos_sequence_from_wrist_keyframes(
        state_qpos=state_qpos,
        start_T_base_wrist=start_T_base_wrist,
        target_T_base_wrist=target_T_base_wrist,
        steps=steps,
        initial_qpos_sim=initial_qpos_sim,
        final_qpos_sim=solved_sim,
        config=config,
        helpers=helpers,
    )
    return _DynamicApproachPlan(
        steps=steps,
        aligned_qpos_real=(
            qpos_sequence[-1].astype(np.float32) if qpos_sequence is not None and qpos_sequence.shape[0] else aligned_qpos_real.astype(np.float32)
        ),
        converged=bool(alignment["converged"]),
        final_error=float(alignment["final_error"]),
        qpos_real_sequence=qpos_sequence,
    )


def _build_approach_qpos_sequence_from_wrist_keyframes(
    *,
    state_qpos: np.ndarray,
    start_T_base_wrist: np.ndarray,
    target_T_base_wrist: np.ndarray,
    steps: int,
    initial_qpos_sim: np.ndarray,
    final_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> np.ndarray | None:
    if steps <= 0:
        return np.zeros((0, 14), dtype=np.float32)

    state_qpos = np.asarray(state_qpos, dtype=np.float32).reshape(-1)[:14].copy()
    key_alphas = [1.0 / 3.0, 2.0 / 3.0, 1.0]
    key_qpos = [state_qpos.copy()]
    seed_sim = np.asarray(initial_qpos_sim, dtype=np.float32).reshape(6)
    for alpha in key_alphas:
        if alpha >= 1.0:
            solved_sim = _wrap_near_seed(final_qpos_sim, seed_sim)
        else:
            target_T = _interpolate_wrist_pose(start_T_base_wrist, target_T_base_wrist, alpha)
            ik_result = _solve_right_arm_wrist_pose_ik_numpy(
                template_qpos_real=state_qpos,
                target_T_base_wrist=target_T,
                initial_qpos_sim=seed_sim,
                config=config,
                helpers=helpers,
            )
            solved_sim = _wrap_near_seed(ik_result["qpos_sim"], seed_sim)
        solved_real = _sim_to_real_right_arm_qpos(solved_sim, helpers)
        qpos = state_qpos.copy()
        qpos[RIGHT_ARM_QPOS_SLICE] = solved_real
        key_qpos.append(qpos.astype(np.float32))
        seed_sim = solved_sim.astype(np.float32)

    key_qpos_arr = np.stack(key_qpos, axis=0)
    sequence = np.zeros((steps, 14), dtype=np.float32)
    for step in range(steps):
        u = float(step + 1) / float(steps)
        segment = min(int(np.ceil(u * 3.0)), 3)
        segment = max(segment, 1)
        left_u = float(segment - 1) / 3.0
        right_u = float(segment) / 3.0
        local = 1.0 if right_u <= left_u else (u - left_u) / (right_u - left_u)
        sequence[step] = (
            (1.0 - local) * key_qpos_arr[segment - 1] + local * key_qpos_arr[segment]
        ).astype(np.float32)
    return sequence


def _interpolate_meta_pose(
    area_type: str,
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Cannot interpolate different meta pose shapes: {a.shape} vs {b.shape}")
    if a.shape[0] == META12_DIM:
        t = float(np.clip(alpha, 0.0, 1.0))
        out = np.zeros(12, dtype=np.float32)
        out[:3] = (1.0 - t) * a[:3] + t * b[:3]
        if area_type == "point":
            out[3:12] = 0.0
            return out

        has_approach = _pose12_has_approach(a, dim_mask12) or _pose12_has_approach(b, dim_mask12)
        if has_approach:
            start_dirs = [_shape_axis_from_pose12(area_type, a, dim_mask12), a[9:12]]
            end_dirs = [_shape_axis_from_pose12(area_type, b, dim_mask12), b[9:12]]
            R_delta = _rotation_from_direction_pairs(start_dirs, end_dirs)
            R_alpha = _rotvec_to_matrix(_matrix_to_rotvec(R_delta) * t)
            out[3:9] = _matrix_to_shape6(_project_psd_trace1(R_alpha @ _shape6_to_matrix(a[3:9]) @ R_alpha.T))
            out[9:12] = _normalize(R_alpha @ a[9:12])
        else:
            axis_a = _shape_axis_from_pose12(area_type, a, dim_mask12)
            axis_b = _shape_axis_from_pose12(area_type, b, dim_mask12)
            if float(np.dot(axis_a, axis_b)) < 0.0:
                axis_b = -axis_b
            axis = _slerp_unit_vectors(axis_a, axis_b, t)
            out[3:9] = _matrix_to_shape6(_shape_matrix_from_axis(area_type, axis))
            out[9:12] = 0.0
        return out.astype(np.float32)

    a = a.reshape(6)
    b = b.reshape(6)
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
    dim_mask12: np.ndarray | None = None,
) -> dict[str, Any]:
    if config.ik_backend == "jax" and np.asarray(target_meta_pose).reshape(-1).shape[0] == META6_DIM:
        return _solve_right_arm_meta_ik_jax(
            target_meta_pose=target_meta_pose,
            T_wrist_meta=T_wrist_meta,
            area_type=area_type,
            initial_qpos_sim=initial_qpos_sim,
            config=config,
        )
    if config.ik_backend not in ("numpy", "jax"):
        raise ValueError(f"Unsupported ik_backend={config.ik_backend!r}; expected 'numpy' or 'jax'")
    return _solve_right_arm_meta_ik_numpy(
        template_qpos_real=template_qpos_real,
        target_meta_pose=target_meta_pose,
        T_wrist_meta=T_wrist_meta,
        area_type=area_type,
        initial_qpos_sim=initial_qpos_sim,
        config=config,
        helpers=helpers,
        dim_mask12=dim_mask12,
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
    dim_mask12: np.ndarray | None = None,
) -> dict[str, Any]:
    template_qpos = np.asarray(template_qpos_real, dtype=np.float32).reshape(-1)[:14].copy()
    reference_T_base_wrist = np.asarray(helpers["fk"](template_qpos, "right_wrist"), dtype=np.float32)
    target_T_base_wrist = _target_wrist_pose_from_meta_target(
        area_type=area_type,
        target_meta_pose=target_meta_pose,
        T_wrist_meta=T_wrist_meta,
        reference_T_base_wrist=reference_T_base_wrist,
        config=config,
        dim_mask12=dim_mask12,
    )
    return _solve_right_arm_wrist_pose_ik_numpy(
        template_qpos_real=template_qpos,
        target_T_base_wrist=target_T_base_wrist,
        initial_qpos_sim=initial_qpos_sim,
        config=config,
        helpers=helpers,
    )


def _solve_right_arm_wrist_pose_ik_numpy(
    *,
    template_qpos_real: np.ndarray,
    target_T_base_wrist: np.ndarray,
    initial_qpos_sim: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> dict[str, Any]:
    q = np.asarray(initial_qpos_sim, dtype=np.float64).reshape(6).copy()
    template_qpos = np.asarray(template_qpos_real, dtype=np.float32).reshape(-1)[:14].copy()
    target_T_base_wrist = np.asarray(target_T_base_wrist, dtype=np.float32).reshape(4, 4)
    converged = False
    final_error = np.inf
    iterations = 0

    for iterations in range(1, config.ik_max_iters + 1):
        error = _wrist_pose_residual_from_sim_qpos(
            q,
            template_qpos_real=template_qpos,
            target_T_base_wrist=target_T_base_wrist,
            config=config,
            helpers=helpers,
        )
        final_error = float(np.linalg.norm(error))
        if final_error < config.ik_tolerance:
            converged = True
            break

        J = _numerical_wrist_pose_residual_jacobian(
            q,
            template_qpos_real=template_qpos,
            target_T_base_wrist=target_T_base_wrist,
            config=config,
            helpers=helpers,
        )
        JT = J.T
        dq = -JT @ np.linalg.solve(J @ JT + (config.ik_damping**2) * np.eye(J.shape[0]), error)
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
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    pose = np.asarray(pose6d, dtype=np.float32).reshape(-1)
    if pose.shape[0] == META12_DIM:
        parts = [pose[:3] * config.ik_position_weight]
        if area_type != "point":
            mask = np.ones(12, dtype=bool) if dim_mask12 is None else np.asarray(dim_mask12, dtype=bool).reshape(12)
            if bool(np.any(mask[3:9])):
                parts.append(_matrix_to_shape6(_project_psd_trace1(_shape6_to_matrix(pose[3:9]))) * config.ik_direction_weight)
            if _pose12_has_approach(pose, mask):
                parts.append(_normalize(pose[9:12]) * config.ik_direction_weight)
        return np.concatenate(parts, axis=0).astype(np.float64)

    pose6d = pose.reshape(6)
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
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    full_real = np.asarray(template_qpos_real, dtype=np.float32).reshape(-1)[:14].copy()
    full_real[RIGHT_ARM_QPOS_SLICE] = _sim_to_real_right_arm_qpos(np.asarray(qpos_sim, dtype=np.float32), helpers)
    T_base_wrist = np.asarray(helpers["fk"](full_real, "right_wrist"), dtype=np.float32)
    wrist_meta_arr = np.asarray(T_wrist_meta, dtype=np.float32)
    if wrist_meta_arr.shape == (12,):
        pose = _wrist_pose12_to_base_pose(wrist_meta_arr, T_base_wrist)
    else:
        pose = _encode_meta_from_transform(area_type, T_base_wrist @ wrist_meta_arr.reshape(4, 4))
    return _weighted_meta_feature_from_pose(area_type, pose, config, dim_mask12=dim_mask12)


def _numerical_meta_feature_jacobian(
    qpos_sim: np.ndarray,
    *,
    template_qpos_real: np.ndarray,
    T_wrist_meta: np.ndarray,
    area_type: str,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
    dim_mask12: np.ndarray | None = None,
) -> np.ndarray:
    q = np.asarray(qpos_sim, dtype=np.float64).reshape(6)
    base_feature = _weighted_meta_feature_from_sim_qpos(
        q,
        template_qpos_real=template_qpos_real,
        T_wrist_meta=T_wrist_meta,
        area_type=area_type,
        config=config,
        helpers=helpers,
        dim_mask12=dim_mask12,
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
            dim_mask12=dim_mask12,
        )
        J[:, dim] = (feature - base_feature) / config.ik_finite_diff_eps
    return J


def _wrist_pose_from_sim_qpos(
    qpos_sim: np.ndarray,
    *,
    template_qpos_real: np.ndarray,
    helpers: dict[str, Any],
) -> np.ndarray:
    full_real = np.asarray(template_qpos_real, dtype=np.float32).reshape(-1)[:14].copy()
    full_real[RIGHT_ARM_QPOS_SLICE] = _sim_to_real_right_arm_qpos(np.asarray(qpos_sim, dtype=np.float32), helpers)
    return np.asarray(helpers["fk"](full_real, "right_wrist"), dtype=np.float32)


def _wrist_pose_residual_from_sim_qpos(
    qpos_sim: np.ndarray,
    *,
    template_qpos_real: np.ndarray,
    target_T_base_wrist: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> np.ndarray:
    current_T = _wrist_pose_from_sim_qpos(qpos_sim, template_qpos_real=template_qpos_real, helpers=helpers)
    target_T = np.asarray(target_T_base_wrist, dtype=np.float32).reshape(4, 4)
    pos_error = target_T[:3, 3] - current_T[:3, 3]
    rot_error = _matrix_to_rotvec(target_T[:3, :3] @ current_T[:3, :3].T)
    return np.concatenate(
        [
            pos_error.astype(np.float64) * float(config.ik_position_weight),
            rot_error.astype(np.float64),
        ],
        axis=0,
    )


def _numerical_wrist_pose_residual_jacobian(
    qpos_sim: np.ndarray,
    *,
    template_qpos_real: np.ndarray,
    target_T_base_wrist: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    helpers: dict[str, Any],
) -> np.ndarray:
    q = np.asarray(qpos_sim, dtype=np.float64).reshape(6)
    base_error = _wrist_pose_residual_from_sim_qpos(
        q,
        template_qpos_real=template_qpos_real,
        target_T_base_wrist=target_T_base_wrist,
        config=config,
        helpers=helpers,
    )
    J = np.zeros((base_error.shape[0], q.shape[0]), dtype=np.float64)
    for dim in range(q.shape[0]):
        q_perturbed = q.copy()
        q_perturbed[dim] += config.ik_finite_diff_eps
        error = _wrist_pose_residual_from_sim_qpos(
            q_perturbed,
            template_qpos_real=template_qpos_real,
            target_T_base_wrist=target_T_base_wrist,
            config=config,
            helpers=helpers,
        )
        J[:, dim] = (error - base_error) / config.ik_finite_diff_eps
    return J


def _meta_tracking_errors(
    *,
    area_type: str,
    qpos_real: np.ndarray,
    T_wrist_meta: np.ndarray,
    target_meta_pose: np.ndarray,
    dim_mask12: np.ndarray | None = None,
    fk_fn,
) -> tuple[float, float]:
    T_base_wrist = np.asarray(fk_fn(qpos_real, "right_wrist"), dtype=np.float32)
    wrist_meta_arr = np.asarray(T_wrist_meta, dtype=np.float32)
    target_meta_pose = np.asarray(target_meta_pose, dtype=np.float32).reshape(-1)
    if target_meta_pose.shape[0] == META12_DIM:
        actual_pose = _wrist_pose12_to_base_pose(wrist_meta_arr.reshape(12), T_base_wrist)
        position_error = float(np.linalg.norm(actual_pose[:3] - target_meta_pose[:3]))
        if area_type == "point":
            return position_error, 0.0
        shape_error = _direction_angle_rad(area_type, actual_pose, target_meta_pose, dim_mask12=dim_mask12)
        return position_error, shape_error

    actual_pose = _encode_meta_from_transform(area_type, T_base_wrist @ wrist_meta_arr.reshape(4, 4))
    target_meta_pose = target_meta_pose.reshape(6)
    position_error = float(np.linalg.norm(actual_pose[:3] - target_meta_pose[:3]))
    if area_type == "point":
        direction_error = 0.0
    else:
        dot = float(np.clip(np.dot(_normalize(actual_pose[3:6]), _normalize(target_meta_pose[3:6])), -1.0, 1.0))
        direction_error = float(np.arccos(dot))
    return position_error, direction_error


def _action_value_diagnostics(
    actions: np.ndarray,
    config: MetaRetargetGeneratorConfig,
    *,
    meta_dim: int = META6_DIM,
) -> tuple[bool, bool, float, float, str]:
    actions = np.asarray(actions)
    finite = bool(np.all(np.isfinite(actions)))
    finite_values = actions[np.isfinite(actions)]
    max_abs_action = float(np.max(np.abs(finite_values))) if finite_values.size else np.inf

    camera_slice = _camera_slice_for_dim(meta_dim)
    if actions.ndim >= 2 and actions.shape[-1] >= camera_slice.stop:
        camera_rotvec = actions[..., camera_slice][..., 3:6]
        rot_norms = np.linalg.norm(camera_rotvec.astype(np.float64), axis=-1)
        finite_rot_norms = rot_norms[np.isfinite(rot_norms)]
        max_camera_rotvec_norm = float(np.max(finite_rot_norms)) if finite_rot_norms.size else np.inf
    else:
        max_camera_rotvec_norm = 0.0

    ok = (
        finite
        and max_abs_action <= config.accept_max_abs_action_value
        and max_camera_rotvec_norm <= config.accept_max_camera_rotvec_norm_rad
    )
    reason = ""
    if not ok:
        reason = (
            f"actions_finite={finite}, "
            f"max_abs_action_value={max_abs_action:.6f}, "
            f"max_camera_rotvec_norm_rad={max_camera_rotvec_norm:.6f}"
        )
    return ok, finite, max_abs_action, max_camera_rotvec_norm, reason


def _camera_pose_to_transform(pose6d: np.ndarray) -> np.ndarray:
    pose6d = np.asarray(pose6d, dtype=np.float32).reshape(6)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose6d[:3]
    T[:3, :3] = _rotvec_to_matrix(pose6d[3:6])
    return T


def _transform_to_camera_pose(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    return np.concatenate([T[:3, 3], _matrix_to_rotvec(T[:3, :3])], axis=0).astype(np.float32)
