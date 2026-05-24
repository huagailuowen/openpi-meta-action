import numpy as np

from openpi.policies import xtrainer_meta_retarget as retarget


def test_retarget_robot_type_defaults_to_xtrainer():
    config = retarget.MetaRetargetGeneratorConfig()

    assert config.robot_type == retarget.RETARGET_ROBOT_XTRAINER
    helpers = retarget._load_robot_helpers(config.robot_type)  # noqa: SLF001
    assert callable(helpers["fk"])


def test_aloha_robot_helpers_load_robotwin_world_fk():
    helpers = retarget._load_robot_helpers(retarget.RETARGET_ROBOT_ALOHA)  # noqa: SLF001
    T_right = helpers["fk"](np.zeros(14, dtype=np.float32), "right_wrist")

    assert T_right.shape == (4, 4)
    assert np.all(np.isfinite(T_right))
    np.testing.assert_allclose(helpers["right_real_to_sim_sign"], np.ones(6, dtype=np.float32))


def test_invalid_retarget_robot_type_is_rejected():
    config = retarget.MetaRetargetGeneratorConfig(robot_type="unsupported_robot")

    try:
        retarget._validate_retarget_algorithm(config)  # noqa: SLF001
    except ValueError as exc:
        assert "Unsupported robot_type" in str(exc)
    else:
        raise AssertionError("Unsupported robot_type should raise ValueError")


def test_canonicalize_structured_meta_chunk_fills_legacy_action_slice():
    state = np.arange(32, dtype=np.float32)
    actions = np.zeros((4, 32), dtype=np.float32)
    target_pose6d = np.arange(4 * 2 * 6, dtype=np.float32).reshape(4, 2, 6)

    out = retarget.canonicalize_repacked_xtrainer_chunk(
        {
            "state": state,
            "actions": actions,
            "meta_areas": {
                "pose6d": np.ones((2, 6), dtype=np.float32),
                "type": np.array([[0], [2]], dtype=np.int32),
                "mask": np.array([[1], [0]], dtype=bool),
            },
            "meta_action_targets": {
                "pose6d": target_pose6d,
                "mask": np.ones((4, 2), dtype=bool),
            },
        },
        max_meta_areas=2,
    )

    np.testing.assert_array_equal(out["state"][14:20], np.zeros((6,), dtype=np.float32))
    np.testing.assert_array_equal(out["meta_areas"]["type"], np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(out["meta_areas"]["mask"], np.array([True, False], dtype=bool))
    np.testing.assert_array_equal(out["actions"][:, 14:20], target_pose6d[:, 0, :])


def test_canonicalize_structured_meta12_chunk_keeps_12d_targets_out_of_action():
    state = np.arange(32, dtype=np.float32)
    actions = np.zeros((4, 32), dtype=np.float32)
    actions[:, 14:20] = 99.0
    actions[:, 20:26] = np.arange(24, dtype=np.float32).reshape(4, 6)
    expected_actions = actions.copy()
    expected_actions[:, 14:20] = 0.0
    target_pose12d = np.arange(4 * 1 * 12, dtype=np.float32).reshape(4, 1, 12)
    dim_mask12 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=bool)

    out = retarget.canonicalize_repacked_xtrainer_chunk(
        {
            "state": state,
            "actions": actions,
            "meta_areas": {
                "pose12d": np.ones((1, 12), dtype=np.float32),
                "dim_mask12": dim_mask12,
                "type": np.array([0], dtype=np.int32),
                "mask": np.array([1], dtype=bool),
            },
            "meta_action_targets": {
                "pose12d": target_pose12d,
                "dim_mask12": np.tile(dim_mask12[None, :, :], (4, 1, 1)),
                "mask": np.ones((4, 1), dtype=bool),
            },
        },
        max_meta_areas=1,
    )

    np.testing.assert_array_equal(out["state"][14:20], np.zeros((6,), dtype=np.float32))
    np.testing.assert_array_equal(out["state"][20:26], state[20:26])
    np.testing.assert_array_equal(out["meta_areas"]["pose12d"], np.ones((1, 12), dtype=np.float32))
    np.testing.assert_array_equal(out["meta_areas"]["dim_mask12"], dim_mask12)
    np.testing.assert_array_equal(out["actions"], expected_actions)
    np.testing.assert_array_equal(out["meta_action_targets"]["pose12d"], target_pose12d)


def test_pose12_anchor_offset_rotates_shape_and_approach_together():
    old_anchor = np.zeros(12, dtype=np.float32)
    old_anchor[:3] = [0.0, 0.0, 0.0]
    old_anchor[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    old_anchor[9:12] = [0.0, 1.0, 0.0]

    new_anchor = np.zeros(12, dtype=np.float32)
    new_anchor[:3] = [0.1, 0.0, 0.0]
    new_anchor[3:9] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    new_anchor[9:12] = [-1.0, 0.0, 0.0]
    dim_mask12 = np.ones(12, dtype=bool)

    shifted = retarget._apply_pose_offset_from_anchor(  # noqa: SLF001
        "line",
        old_anchor,
        old_anchor_pose=old_anchor,
        new_anchor_pose=new_anchor,
        dim_mask12=dim_mask12,
    )

    np.testing.assert_allclose(shifted[:3], new_anchor[:3], atol=1e-6)
    np.testing.assert_allclose(shifted[3:9], new_anchor[3:9], atol=1e-6)
    np.testing.assert_allclose(shifted[9:12], new_anchor[9:12], atol=1e-6)


def test_legacy_min_rotation_aligns_line_shape_as_undirected_axis():
    r_small = retarget._rotvec_to_matrix(np.array([0.0, 0.0, np.deg2rad(10.0)], dtype=np.float32))  # noqa: SLF001
    source = np.zeros(12, dtype=np.float32)
    source[3:9] = retarget._matrix_to_shape6(np.outer([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]))  # noqa: SLF001
    source[9:12] = [0.0, 1.0, 0.0]
    target = np.zeros(12, dtype=np.float32)
    target_axis = r_small @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target[3:9] = retarget._matrix_to_shape6(np.outer(target_axis, target_axis))  # noqa: SLF001
    target[9:12] = r_small @ np.array([0.0, 1.0, 0.0], dtype=np.float32)

    rotation = retarget._legacy_rotation_between_pose12(  # noqa: SLF001
        "line",
        source,
        target,
        dim_mask12=np.ones(12, dtype=bool),
        config=retarget.MetaRetargetGeneratorConfig(
            retarget_algorithm=retarget.RETARGET_ALGORITHM_LEGACY_STRUCTURED_MIN_ROTATION
        ),
    )

    assert np.rad2deg(np.linalg.norm(retarget._matrix_to_rotvec(rotation))) < 10.1  # noqa: SLF001
    assert retarget._angle_between_unit_vectors_deg(rotation @ source[9:12], target[9:12]) < 1e-3  # noqa: SLF001
    assert retarget._shape_angle_deg_for_pose12(  # noqa: SLF001
        "line",
        retarget._interpolate_meta_pose_legacy(  # noqa: SLF001
            "line",
            source,
            target,
            1.0,
            dim_mask12=np.ones(12, dtype=bool),
            config=retarget.MetaRetargetGeneratorConfig(),
        ),
        target,
        np.ones(12, dtype=bool),
    ) < 1e-3


def test_legacy_min_rotation_applies_to_surface_normal_axis():
    r_small = retarget._rotvec_to_matrix(np.array([0.0, 0.0, np.deg2rad(-12.0)], dtype=np.float32))  # noqa: SLF001
    source = np.zeros(12, dtype=np.float32)
    source[3:9] = retarget._matrix_to_shape6(retarget._shape_matrix_from_axis("surface", np.array([1.0, 0.0, 0.0])))  # noqa: SLF001
    source[9:12] = [0.0, 1.0, 0.0]
    target = np.zeros(12, dtype=np.float32)
    target_normal = r_small @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target[3:9] = retarget._matrix_to_shape6(retarget._shape_matrix_from_axis("surface", target_normal))  # noqa: SLF001
    target[9:12] = r_small @ np.array([0.0, 1.0, 0.0], dtype=np.float32)

    rotation = retarget._legacy_rotation_between_pose12(  # noqa: SLF001
        "surface",
        source,
        target,
        dim_mask12=np.ones(12, dtype=bool),
        config=retarget.MetaRetargetGeneratorConfig(),
    )

    assert np.rad2deg(np.linalg.norm(retarget._matrix_to_rotvec(rotation))) < 12.1  # noqa: SLF001
    assert retarget._angle_between_unit_vectors_deg(rotation @ source[9:12], target[9:12]) < 1e-3  # noqa: SLF001


def test_legacy_min_rotation_near_parallel_shape_approach_does_not_flip():
    r_small = retarget._rotvec_to_matrix(np.array([0.0, 0.0, np.deg2rad(20.0)], dtype=np.float32))  # noqa: SLF001
    source = np.zeros(12, dtype=np.float32)
    source[3:9] = retarget._matrix_to_shape6(np.outer([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]))  # noqa: SLF001
    source[9:12] = retarget._normalize(np.array([1.0, 0.01, 0.0], dtype=np.float32))  # noqa: SLF001
    target = np.zeros(12, dtype=np.float32)
    target_axis = r_small @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target[3:9] = retarget._matrix_to_shape6(np.outer(target_axis, target_axis))  # noqa: SLF001
    target[9:12] = retarget._normalize(r_small @ source[9:12])  # noqa: SLF001

    rotation = retarget._legacy_rotation_between_pose12(  # noqa: SLF001
        "line",
        source,
        target,
        dim_mask12=np.ones(12, dtype=bool),
        config=retarget.MetaRetargetGeneratorConfig(),
    )

    assert np.all(np.isfinite(rotation))
    assert np.rad2deg(np.linalg.norm(retarget._matrix_to_rotvec(rotation))) < 25.0  # noqa: SLF001
    assert retarget._angle_between_unit_vectors_deg(rotation @ source[9:12], target[9:12]) < 1.0  # noqa: SLF001


def test_legacy_future_near_keeps_old_smooth_transition_distinct_from_v2_direct_follow():
    old_input = np.zeros(12, dtype=np.float32)
    old_input[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    old_targets = np.stack([old_input.copy() for _ in range(5)], axis=0)
    for i in range(old_targets.shape[0]):
        old_targets[i, 0] = 0.02 * float(i + 1)

    new_anchor = old_targets[1].copy()
    new_anchor[1] += 0.006
    legacy_targets, source_indices = retarget._build_smooth_chase_targets_legacy(  # noqa: SLF001
        "line",
        old_targets,
        old_anchor_pose=old_targets[1],
        new_anchor_pose=new_anchor,
        trajectory_start_index=1,
        transition_steps=4,
        dim_mask12=np.ones(12, dtype=bool),
        config=retarget.MetaRetargetGeneratorConfig(
            retarget_algorithm=retarget.RETARGET_ALGORITHM_LEGACY_STRUCTURED_MIN_ROTATION
        ),
    )
    v2_direct_targets = old_targets[source_indices]

    np.testing.assert_array_equal(source_indices, np.array([1, 2, 3, 4, 4], dtype=np.int32))
    assert not np.allclose(legacy_targets[0, :3], v2_direct_targets[0, :3])
    np.testing.assert_allclose(legacy_targets[-1, :3], v2_direct_targets[-1, :3], atol=1e-6)


def test_meta_retarget_generator_default_uses_legacy_algorithm():
    assert (
        retarget.MetaRetargetGeneratorConfig().retarget_algorithm
        == retarget.RETARGET_ALGORITHM_LEGACY_STRUCTURED_MIN_ROTATION
    )


def test_generate_retargeted_chunk_accepts_zero_noise_line_case():
    helpers = retarget._load_xtrainer_helpers()
    qpos = np.zeros(14, dtype=np.float32)
    T_wrist = helpers["fk"](qpos, "right_wrist")
    line_pose = np.concatenate([T_wrist[:3, 3], T_wrist[:3, 0]], axis=0).astype(np.float32)

    state = np.zeros(32, dtype=np.float32)
    state[:14] = qpos
    actions = np.zeros((5, 32), dtype=np.float32)
    actions[:, :14] = qpos
    actions[:, 14:20] = line_pose

    result = retarget.generate_retargeted_chunk(
        {
            "state": state,
            "actions": actions,
            "meta_areas": {
                "pose6d": line_pose.reshape(1, 6),
                "type": np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
                "mask": np.array([True], dtype=bool),
            },
        },
        rng=np.random.default_rng(0),
        config=retarget.MetaRetargetGeneratorConfig(
            position_noise_max_m=0.0,
            direction_noise_max_deg=0.0,
            future_near_mode_prob=1.0,
            future_near_position_noise_max_m=0.0,
            future_near_direction_noise_max_deg=0.0,
            approach_joint_step_rad=0.04,
            ik_max_iters=10,
            accept_max_step_joint_delta_rad=1.0,
        ),
    )

    assert result is not None
    assert result.diagnostics.accepted
    assert result.diagnostics.retarget_mode == "future_near"
    assert result.diagnostics.approach_steps == 0
    assert result.actions.shape == (5, 32)
    assert result.meta_area_pose6d.shape == (1, 6)


def test_generate_retargeted_chunk_accepts_zero_noise_12d_line_approach_case():
    helpers = retarget._load_xtrainer_helpers()
    qpos = np.zeros(14, dtype=np.float32)
    T_wrist = helpers["fk"](qpos, "right_wrist")
    local_pose = np.zeros(12, dtype=np.float32)
    local_pose[:3] = [0.02, 0.0, 0.0]
    local_pose[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    local_pose[9:12] = [0.0, 1.0, 0.0]
    base_pose = retarget._wrist_pose12_to_base_pose(local_pose, T_wrist)  # noqa: SLF001

    state = np.zeros(32, dtype=np.float32)
    state[:14] = qpos
    actions = np.zeros((5, 32), dtype=np.float32)
    actions[:, :14] = qpos
    pose_targets = np.tile(base_pose.reshape(1, 1, 12), (5, 1, 1))
    dim_masks = np.ones_like(pose_targets, dtype=bool)

    result = retarget.generate_retargeted_chunk(
        {
            "state": state,
            "actions": actions,
            "meta_areas": {
                "pose12d": base_pose.reshape(1, 12),
                "dim_mask12": np.ones((1, 12), dtype=bool),
                "type": np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
                "mask": np.array([True], dtype=bool),
            },
            "meta_action_targets": {
                "pose12d": pose_targets,
                "dim_mask12": dim_masks,
                "mask": np.ones((5, 1), dtype=bool),
            },
        },
        rng=np.random.default_rng(0),
        config=retarget.MetaRetargetGeneratorConfig(
            position_noise_max_m=0.0,
            direction_noise_max_deg=0.0,
            future_near_mode_prob=1.0,
            future_near_position_noise_max_m=0.0,
            future_near_direction_noise_max_deg=0.0,
            approach_joint_step_rad=0.04,
            ik_max_iters=30,
            accept_max_step_joint_delta_rad=1.0,
        ),
    )

    assert result is not None
    assert result.diagnostics.accepted
    assert result.meta_action_target_pose12d is not None
    assert result.diagnostics.max_position_error_m < 1e-5
    assert result.diagnostics.max_direction_error_rad < 1e-4


def test_future_near_targets_follow_directly_after_selected_frame():
    old_targets = np.array(
        [
            [0.01, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.03, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.04, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_indices = retarget._direct_follow_source_indices(  # noqa: SLF001
        old_targets,
        trajectory_start_index=2,
    )
    targets = old_targets[source_indices]

    np.testing.assert_array_equal(source_indices, np.array([2, 3, 3, 3], dtype=np.int32))
    np.testing.assert_allclose(targets, old_targets[source_indices], atol=1e-6)


def test_pair_retarget_anchor_selection_finds_future_lookahead():
    old_input = np.zeros(12, dtype=np.float32)
    old_input[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    old_targets = np.stack([old_input.copy() for _ in range(6)], axis=0)
    for i in range(old_targets.shape[0]):
        old_targets[i, 0] = 0.01 * float(i + 1)

    candidate = old_targets[2].copy()
    candidate[0] += 0.002
    selected = retarget._select_pair_retarget_anchor(  # noqa: SLF001
        "line",
        candidate,
        old_input,
        old_targets,
        config=retarget.MetaRetargetGeneratorConfig(pair_retarget_lookahead_frames=5),
        dim_mask12=np.ones(12, dtype=bool),
    )

    assert selected is not None
    assert selected["retarget_mode"] == "future_near"
    assert selected["trajectory_start_index"] == 3
    np.testing.assert_array_equal(selected["source_indices"], np.array([3, 4, 5, 5, 5, 5], dtype=np.int32))


def test_pair_retarget_anchor_selection_rejects_forward_forbidden_zone():
    old_input = np.zeros(12, dtype=np.float32)
    old_input[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    old_targets = np.stack([old_input.copy() for _ in range(6)], axis=0)
    for i in range(old_targets.shape[0]):
        old_targets[i, 0] = 0.01 * float(i + 1)

    forward_candidate = old_input.copy()
    forward_candidate[0] += 0.03
    selected = retarget._select_pair_retarget_anchor(  # noqa: SLF001
        "line",
        forward_candidate,
        old_input,
        old_targets,
        config=retarget.MetaRetargetGeneratorConfig(
            future_near_position_noise_max_m=0.001,
            pair_retarget_lookahead_frames=0,
        ),
        dim_mask12=np.ones(12, dtype=bool),
    )

    assert selected is None


def test_pair_retarget_anchor_selection_accepts_lateral_correction():
    old_input = np.zeros(12, dtype=np.float32)
    old_input[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    old_targets = np.stack([old_input.copy() for _ in range(6)], axis=0)
    for i in range(old_targets.shape[0]):
        old_targets[i, 0] = 0.01 * float(i + 1)

    lateral_candidate = old_input.copy()
    lateral_candidate[1] += 0.03
    selected = retarget._select_pair_retarget_anchor(  # noqa: SLF001
        "line",
        lateral_candidate,
        old_input,
        old_targets,
        config=retarget.MetaRetargetGeneratorConfig(
            future_near_position_noise_max_m=0.001,
            pair_retarget_lookahead_frames=0,
        ),
        dim_mask12=np.ones(12, dtype=bool),
    )

    assert selected is not None
    assert selected["retarget_mode"] == "correction"
    assert selected["trajectory_start_index"] == 0


def test_pose12_feature_respects_approach_dim_mask():
    pose = np.zeros(12, dtype=np.float32)
    pose[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pose[9:12] = [0.0, 1.0, 0.0]
    no_approach_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    feature_no_approach = retarget._weighted_meta_feature_from_pose(  # noqa: SLF001
        "line",
        pose,
        retarget.MetaRetargetGeneratorConfig(),
        dim_mask12=no_approach_mask,
    )
    feature_with_approach = retarget._weighted_meta_feature_from_pose(  # noqa: SLF001
        "line",
        pose,
        retarget.MetaRetargetGeneratorConfig(),
        dim_mask12=np.ones(12, dtype=bool),
    )

    assert feature_no_approach.shape == (9,)
    assert feature_with_approach.shape == (12,)


def test_pair_source_affordance_rejects_source_approach_to_target_no_approach():
    helpers = retarget._load_xtrainer_helpers()
    state = np.zeros(32, dtype=np.float32)
    source_pose = np.zeros((1, 12), dtype=np.float32)
    source_pose[0, 3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    source_pose[0, 9:12] = [0.0, 1.0, 0.0]
    target_pose = source_pose.copy()
    target_pose[0, 9:12] = 0.0
    target_mask = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=bool)

    fixed = retarget._pair_source_affordance_as_target_input_pose(  # noqa: SLF001
        {
            "state": state,
            "meta_areas": {
                "pose12d": source_pose,
                "dim_mask12": np.ones((1, 12), dtype=bool),
                "type": np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
                "mask": np.array([True], dtype=bool),
            },
        },
        {
            "state": state,
            "meta_areas": {
                "pose12d": target_pose,
                "dim_mask12": target_mask,
                "type": np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
                "mask": np.array([True], dtype=bool),
            },
        },
        helpers=helpers,
    )

    assert fixed is None


def test_pair_source_affordance_masks_target_approach_when_source_has_none():
    helpers = retarget._load_xtrainer_helpers()
    state = np.zeros(32, dtype=np.float32)
    source_pose = np.zeros((1, 12), dtype=np.float32)
    source_pose[0, 3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_pose = source_pose.copy()
    target_pose[0, 9:12] = [0.0, 1.0, 0.0]

    fixed = retarget._pair_source_affordance_as_target_input_pose(  # noqa: SLF001
        {
            "state": state,
            "meta_areas": {
                "pose12d": source_pose,
                "dim_mask12": np.ones((1, 12), dtype=bool),
                "type": np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
                "mask": np.array([True], dtype=bool),
            },
        },
        {
            "state": state,
            "meta_areas": {
                "pose12d": target_pose,
                "dim_mask12": np.ones((1, 12), dtype=bool),
                "type": np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
                "mask": np.array([True], dtype=bool),
            },
        },
        helpers=helpers,
    )

    assert fixed is not None
    np.testing.assert_array_equal(fixed["dim_mask12"][9:12], np.array([False, False, False]))


def test_matrix_to_rotvec_near_pi_is_bounded():
    axis = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    rot_matrix = retarget._rotvec_to_matrix(axis * np.pi).astype(np.float64)  # noqa: SLF001
    rot_matrix[0, 1] += 1e-6
    rot_matrix[1, 0] -= 1e-6

    rotvec = retarget._matrix_to_rotvec(rot_matrix)  # noqa: SLF001

    assert np.all(np.isfinite(rotvec))
    assert np.linalg.norm(rotvec) <= retarget.MetaRetargetGeneratorConfig().accept_max_camera_rotvec_norm_rad
    np.testing.assert_allclose(np.linalg.norm(rotvec), np.pi, atol=1e-3)


def test_action_value_diagnostics_rejects_bad_camera_rotvec():
    actions = np.zeros((2, 32), dtype=np.float32)
    actions[:, 23] = 1e12

    ok, finite, _, max_camera_rotvec_norm, reason = retarget._action_value_diagnostics(  # noqa: SLF001
        actions,
        retarget.MetaRetargetGeneratorConfig(),
    )

    assert not ok
    assert finite
    assert max_camera_rotvec_norm > retarget.MetaRetargetGeneratorConfig().accept_max_camera_rotvec_norm_rad
    assert "max_camera_rotvec_norm_rad" in reason


def test_clipped_smoothstep_with_tail_has_zero_band_and_slow_tail():
    loss = retarget._clipped_smoothstep_with_tail  # noqa: SLF001

    assert loss(2.0, 3.0, 30.0, 0.01) == 0.0
    assert 0.0 < loss(15.0, 3.0, 30.0, 0.01) < 1.0
    assert loss(30.0, 3.0, 30.0, 0.01) == 1.0
    np.testing.assert_allclose(loss(40.0, 3.0, 30.0, 0.01), 1.1, atol=1e-6)


def test_roll_domain_loss_only_penalizes_wrist_x_near_world_z():
    config = retarget.MetaRetargetGeneratorConfig()
    safe = np.eye(3, dtype=np.float32)
    dangerous = np.eye(3, dtype=np.float32)
    dangerous[:, 0] = [0.0, 0.0, 1.0]
    dangerous[:, 1] = [0.0, 1.0, 0.0]
    dangerous[:, 2] = [-1.0, 0.0, 0.0]

    assert retarget._wrist_roll_domain_loss(safe, config) == 0.0  # noqa: SLF001
    assert retarget._wrist_roll_domain_loss(dangerous, config) > 1.0  # noqa: SLF001


def test_target_wrist_pose_uses_reference_roll_when_shape_is_underdetermined():
    local_pose = np.zeros(12, dtype=np.float32)
    local_pose[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    target_pose = local_pose.copy()
    target_pose[:3] = [0.2, -0.1, 0.3]
    R_ref = retarget._rotvec_to_matrix(np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float32))  # noqa: SLF001
    T_ref = np.eye(4, dtype=np.float32)
    T_ref[:3, :3] = R_ref

    T = retarget._target_wrist_pose_from_pose12(  # noqa: SLF001
        area_type="line",
        wrist_pose12=local_pose,
        target_pose12=target_pose,
        reference_T_base_wrist=T_ref,
        config=retarget.MetaRetargetGeneratorConfig(),
        dim_mask12=mask,
    )
    actual_pose = retarget._wrist_pose12_to_base_pose(local_pose, T)  # noqa: SLF001

    np.testing.assert_allclose(actual_pose[:3], target_pose[:3], atol=1e-6)
    assert retarget._shape_angle_deg_for_pose12("line", actual_pose, target_pose, mask) < 1e-4  # noqa: SLF001
    assert retarget._rotation_angle_deg(T[:3, :3], R_ref) < 1e-4  # noqa: SLF001


def test_target_wrist_pose_aligns_shape_and_approach_when_both_are_active():
    local_pose = np.zeros(12, dtype=np.float32)
    local_pose[:3] = [0.03, 0.01, -0.02]
    local_pose[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    local_pose[9:12] = [0.0, 1.0, 0.0]
    target_pose = np.zeros(12, dtype=np.float32)
    target_pose[:3] = [0.2, -0.1, 0.3]
    target_pose[3:9] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    target_pose[9:12] = [-1.0, 0.0, 0.0]
    mask = np.ones(12, dtype=bool)

    T = retarget._target_wrist_pose_from_pose12(  # noqa: SLF001
        area_type="line",
        wrist_pose12=local_pose,
        target_pose12=target_pose,
        reference_T_base_wrist=np.eye(4, dtype=np.float32),
        config=retarget.MetaRetargetGeneratorConfig(),
        dim_mask12=mask,
    )
    actual_pose = retarget._wrist_pose12_to_base_pose(local_pose, T)  # noqa: SLF001

    np.testing.assert_allclose(actual_pose[:3], target_pose[:3], atol=1e-6)
    assert retarget._shape_angle_deg_for_pose12("line", actual_pose, target_pose, mask) < 1e-4  # noqa: SLF001
    assert retarget._approach_angle_deg_for_pose12(actual_pose, target_pose, mask) < 1e-4  # noqa: SLF001


def test_wrist_pose_ik_tracks_exact_meta_pose_for_current_configuration():
    helpers = retarget._load_xtrainer_helpers()
    qpos = np.zeros(14, dtype=np.float32)
    T_wrist = helpers["fk"](qpos, "right_wrist")
    local_pose = np.zeros(12, dtype=np.float32)
    local_pose[:3] = [0.02, 0.0, 0.0]
    local_pose[3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    local_pose[9:12] = [0.0, 1.0, 0.0]
    target_pose = retarget._wrist_pose12_to_base_pose(local_pose, T_wrist)  # noqa: SLF001

    solved = retarget._solve_right_arm_meta_ik(  # noqa: SLF001
        template_qpos_real=qpos,
        target_meta_pose=target_pose,
        T_wrist_meta=local_pose,
        area_type="line",
        initial_qpos_sim=helpers["real_to_sim_arm"](qpos[retarget.RIGHT_ARM_QPOS_SLICE], "right"),
        config=retarget.MetaRetargetGeneratorConfig(ik_max_iters=20),
        helpers=helpers,
        dim_mask12=np.ones(12, dtype=bool),
    )
    solved_qpos = qpos.copy()
    solved_qpos[retarget.RIGHT_ARM_QPOS_SLICE] = retarget._sim_to_real_right_arm_qpos(  # noqa: SLF001
        solved["qpos_sim"],
        helpers,
    )
    pos_err, dir_err = retarget._meta_tracking_errors(  # noqa: SLF001
        area_type="line",
        qpos_real=solved_qpos,
        T_wrist_meta=local_pose,
        target_meta_pose=target_pose,
        dim_mask12=np.ones(12, dtype=bool),
        fk_fn=helpers["fk"],
    )

    assert solved["final_error"] < 1e-3
    assert pos_err < 1e-5
    assert dir_err < 1e-4
