from types import SimpleNamespace

import numpy as np

from openpi.policies import xtrainer_meta_retarget as retarget
import openpi.transforms as transforms
from scripts import build_xtrainer_meta_retarget_cache as cache_builder


def _dummy_result(actions: np.ndarray, state: np.ndarray) -> retarget.MetaRetargetResult:
    return retarget.MetaRetargetResult(
        state=state.astype(np.float32),
        actions=actions.astype(np.float32),
        meta_area_pose6d=np.zeros((1, 6), dtype=np.float32),
        meta_area_type=np.array([retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
        meta_area_mask=np.array([True], dtype=bool),
        diagnostics=retarget.MetaRetargetDiagnostics(
            accepted=True,
            area_type="line",
            horizon=actions.shape[0],
            approach_steps=0,
            ik_nonconverged=0,
            max_position_error_m=0.0,
            max_direction_error_rad=0.0,
            max_step_joint_delta_rad=0.0,
        ),
    )


def test_infer_delta_action_masks():
    data_config = SimpleNamespace(
        data_transforms=SimpleNamespace(
            inputs=[
                transforms.SubsampleActions(2),
                transforms.DeltaActions(mask=[True, False, True]),
                transforms.DeltaActions(mask=None),
            ]
        )
    )

    masks = cache_builder._infer_delta_action_masks(data_config)  # noqa: SLF001

    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], np.array([True, False, True], dtype=bool))


def test_canonicalize_sample_subsamples_meta_action_targets(monkeypatch):
    actions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    pose12d = np.arange(6 * 1 * 12, dtype=np.float32).reshape(6, 1, 12)
    dim_mask12 = (np.arange(6 * 1 * 12).reshape(6, 1, 12) % 2) == 0
    mask = np.array([[True], [False], [True], [True], [False], [True]])
    sample_level = np.array([123], dtype=np.int32)

    def fake_canonicalize_repacked_xtrainer_chunk(repacked, *, max_meta_areas):
        del repacked, max_meta_areas
        return {
            "actions": actions.copy(),
            "meta_action_targets": {
                "pose12d": pose12d.copy(),
                "dim_mask12": dim_mask12.copy(),
                "mask": mask.copy(),
                "sample_level": sample_level.copy(),
            },
        }

    monkeypatch.setattr(
        cache_builder._retarget,
        "canonicalize_repacked_xtrainer_chunk",
        fake_canonicalize_repacked_xtrainer_chunk,
    )
    data_config = SimpleNamespace(repack_transforms=SimpleNamespace(inputs=[]))

    canonical = cache_builder._canonicalize_sample(
        {},
        data_config,
        max_meta_areas=1,
        action_stride=3,
    )

    np.testing.assert_array_equal(canonical["actions"], actions[::3])
    np.testing.assert_array_equal(canonical["meta_action_targets"]["pose12d"], pose12d[::3])
    np.testing.assert_array_equal(canonical["meta_action_targets"]["dim_mask12"], dim_mask12[::3])
    np.testing.assert_array_equal(canonical["meta_action_targets"]["mask"], mask[::3])
    np.testing.assert_array_equal(canonical["meta_action_targets"]["sample_level"], sample_level)


def test_apply_delta_action_masks_matches_training_transform():
    state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    actions = np.array([[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]], dtype=np.float32)
    result = _dummy_result(actions, state)

    delta_result = cache_builder._apply_delta_action_masks(result, [[True, False, True]])  # noqa: SLF001

    transformed = transforms.DeltaActions(mask=[True, False, True])({"state": state.copy(), "actions": actions.copy()})
    np.testing.assert_array_equal(delta_result.actions, transformed["actions"])
    np.testing.assert_array_equal(result.actions, actions)
    assert delta_result.diagnostics is result.diagnostics
