import numpy as np

from openpi.policies import xtrainer_meta_retarget as retarget


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


def test_future_near_targets_start_after_selected_frame():
    old_input = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    old_targets = np.array(
        [
            [0.01, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.03, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.04, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    selected = old_targets[1]
    new_anchor = selected.copy()
    new_anchor[:3] += np.array([0.004, 0.0, 0.0], dtype=np.float32)

    targets, source_indices = retarget._build_smooth_chase_targets(  # noqa: SLF001
        "line",
        old_targets,
        old_anchor_pose=selected,
        new_anchor_pose=new_anchor,
        trajectory_start_index=2,
        transition_steps=2,
    )

    np.testing.assert_array_equal(source_indices, np.array([2, 3, 3, 3], dtype=np.int32))
    assert targets[0, 0] > old_targets[2, 0]
    np.testing.assert_allclose(targets[1:], old_targets[source_indices[1:]], atol=1e-6)


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
