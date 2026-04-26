import numpy as np

from openpi.policies.xtrainer_meta_policy import XTrainerMetaInputs


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
