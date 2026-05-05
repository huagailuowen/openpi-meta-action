import numpy as np

from openpi.policies.xtrainer_meta_policy import XTrainerMetaInputs, XTrainerStructuredMetaInputs


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
