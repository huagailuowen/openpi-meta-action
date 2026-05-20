import numpy as np

from openpi.policies.xtrainer_meta_policy import XTrainerMetaInputs
from openpi.policies.xtrainer_meta_policy import XTrainerMetaOutputs
from openpi.policies.xtrainer_meta_policy import XTrainerStructuredMetaInputs


def test_xtrainer_meta_inputs_derive_meta_and_zero_state_slice():
    transform = XTrainerMetaInputs(max_meta_areas=1)
    state = np.arange(32, dtype=np.float32)
    sample = {
        "images": {
            "cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 8, 8), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 8, 8), dtype=np.uint8),
        },
        "state": state,
    }

    out = transform(sample)

    np.testing.assert_array_equal(out["meta_areas"]["pose6d"][0], state[14:20])
    np.testing.assert_array_equal(out["state"][:14], state[:14])
    np.testing.assert_array_equal(out["state"][20:26], state[20:26])
    np.testing.assert_array_equal(out["state"][14:20], np.zeros((6,), dtype=np.float32))
    assert bool(out["meta_areas"]["mask"][0])


def test_xtrainer_structured_meta_inputs_use_explicit_meta_and_targets():
    transform = XTrainerStructuredMetaInputs(max_meta_areas=2)
    state = np.arange(32, dtype=np.float32)
    actions = np.zeros((3, 32), dtype=np.float32)
    target_pose6d = np.arange(3 * 2 * 6, dtype=np.float32).reshape(3, 2, 6)
    sample = {
        "images": {
            "cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 8, 8), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 8, 8), dtype=np.uint8),
        },
        "state": state,
        "actions": actions,
        "meta_areas": {
            "pose6d": np.ones((2, 6), dtype=np.float32),
            "type": np.array([[1], [2]], dtype=np.float32),
            "mask": np.array([[1.0], [0.0]], dtype=np.float32),
        },
        "meta_action_targets": {
            "pose6d": target_pose6d,
            "mask": np.ones((3, 2), dtype=bool),
        },
    }

    out = transform(sample)

    np.testing.assert_array_equal(out["meta_areas"]["pose6d"], sample["meta_areas"]["pose6d"])
    np.testing.assert_array_equal(out["meta_areas"]["type"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(out["meta_areas"]["mask"], np.array([True, False], dtype=bool))
    np.testing.assert_array_equal(out["state"][14:20], np.zeros((6,), dtype=np.float32))
    np.testing.assert_array_equal(out["actions"][:, 14:20], target_pose6d[:, 0, :])


def test_xtrainer_structured_meta_inputs_use_explicit_12d_meta_and_dim_masks():
    transform = XTrainerStructuredMetaInputs(max_meta_areas=2, meta_area_pose_dim=12)
    state = np.arange(32, dtype=np.float32)
    actions = np.zeros((3, 32), dtype=np.float32)
    actions[:, 14:20] = 99.0
    actions[:, 20:26] = np.arange(18, dtype=np.float32).reshape(3, 6)
    expected_actions = actions.copy()
    expected_actions[:, 14:20] = 0.0
    target_pose12d = np.arange(3 * 2 * 12, dtype=np.float32).reshape(3, 2, 12)
    dim_mask12 = np.ones((2, 12), dtype=bool)
    dim_mask12[1, 9:12] = False
    sample = {
        "images": {
            "cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 8, 8), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 8, 8), dtype=np.uint8),
        },
        "state": state,
        "actions": actions,
        "meta_areas": {
            "pose12d": np.ones((2, 12), dtype=np.float32),
            "dim_mask12": dim_mask12,
            "type": np.array([[1], [2]], dtype=np.float32),
            "mask": np.array([[1.0], [0.0]], dtype=np.float32),
        },
        "meta_action_targets": {
            "pose12d": target_pose12d,
            "dim_mask12": np.broadcast_to(dim_mask12, (3, 2, 12)),
            "mask": np.ones((3, 2), dtype=bool),
        },
        "execution_meta_areas": {
            "pose12d": np.full((2, 12), 7.0, dtype=np.float32),
            "dim_mask12": dim_mask12,
            "type": np.array([3, 4], dtype=np.int32),
            "mask": np.array([True, True], dtype=bool),
        },
        "reference_actions": np.full((50, 14), 0.25, dtype=np.float32),
        "reference_action_mask": np.array(1, dtype=bool),
        "contrastive_meta_areas": {
            "pose12d": np.full((2, 12), 9.0, dtype=np.float32),
            "dim_mask12": dim_mask12,
            "type": np.array([5, 6], dtype=np.int32),
            "mask": np.array([True, False], dtype=bool),
        },
        "contrastive_reference_actions": np.full((50, 14), 0.5, dtype=np.float32),
        "contrastive_reference_action_mask": np.array(1, dtype=bool),
    }

    out = transform(sample)

    np.testing.assert_array_equal(out["meta_areas"]["pose12d"], sample["meta_areas"]["pose12d"])
    np.testing.assert_array_equal(out["meta_areas"]["dim_mask12"], dim_mask12)
    np.testing.assert_array_equal(out["meta_areas"]["type"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(out["meta_areas"]["mask"], np.array([True, False], dtype=bool))
    np.testing.assert_array_equal(out["state"][14:20], np.zeros((6,), dtype=np.float32))
    np.testing.assert_array_equal(out["state"][20:26], state[20:26])
    np.testing.assert_array_equal(out["actions"], expected_actions)
    np.testing.assert_array_equal(out["meta_action_targets"]["pose12d"], target_pose12d)
    np.testing.assert_array_equal(out["meta_action_targets"]["dim_mask12"], np.broadcast_to(dim_mask12, (3, 2, 12)))
    np.testing.assert_array_equal(out["meta_action_targets"]["mask"], np.ones((3, 2), dtype=bool))
    np.testing.assert_array_equal(out["execution_meta_areas"], sample["execution_meta_areas"])
    np.testing.assert_array_equal(out["reference_actions"], sample["reference_actions"])
    assert bool(out["reference_action_mask"])
    np.testing.assert_array_equal(
        out["contrastive_meta_areas"]["pose12d"], sample["contrastive_meta_areas"]["pose12d"]
    )
    np.testing.assert_array_equal(out["contrastive_meta_areas"]["dim_mask12"], dim_mask12)
    np.testing.assert_array_equal(out["contrastive_meta_areas"]["type"], np.array([5, 6], dtype=np.int32))
    np.testing.assert_array_equal(out["contrastive_meta_areas"]["mask"], np.array([True, False], dtype=bool))
    np.testing.assert_array_equal(out["contrastive_reference_actions"], sample["contrastive_reference_actions"])
    assert bool(out["contrastive_reference_action_mask"])


def test_xtrainer_meta_outputs_can_zero_unused_12d_slice():
    transform = XTrainerMetaOutputs(action_dim=32, zero_unused_meta_slice=True)
    actions = np.ones((2, 32), dtype=np.float32)
    actions[:, 20:26] = np.arange(12, dtype=np.float32).reshape(2, 6)

    out = transform({"actions": actions})

    np.testing.assert_array_equal(out["actions"][:, 14:20], np.zeros((2, 6), dtype=np.float32))
    np.testing.assert_array_equal(out["actions"][:, 20:26], actions[:, 20:26])
