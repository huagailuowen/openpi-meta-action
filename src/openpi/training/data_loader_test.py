import dataclasses
import time

import jax
import numpy as np

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.policies import xtrainer_meta_retarget as _meta_retarget
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_beta_pair_dataset_reference_condition_uses_source_actions():
    class TinyDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "state": np.zeros((32,), dtype=np.float32),
                "actions": np.full((4, 32), idx + 1, dtype=np.float32),
                "meta_areas": {
                    "pose12d": np.zeros((1, 12), dtype=np.float32),
                    "dim_mask12": np.ones((1, 12), dtype=bool),
                    "type": np.zeros((1,), dtype=np.int32),
                    "mask": np.ones((1,), dtype=bool),
                },
                "meta_action_targets": {
                    "pose12d": np.zeros((4, 1, 12), dtype=np.float32),
                    "dim_mask12": np.ones((4, 1, 12), dtype=bool),
                    "mask": np.ones((4, 1), dtype=bool),
                },
                "tool_instance_hash": np.asarray([7], dtype=np.int32),
                "source_type_id": np.asarray([0], dtype=np.int32),
                "episode_index": np.asarray(idx, dtype=np.int64),
            }

    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_beta_seed=0,
        meta_beta_self_same_chunk_prob=1.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=0.0,
        meta_beta_meta_area_condition_prob=0.0,
        meta_beta_reference_action_condition_prob=1.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_non_retarget_obs_only_condition_prob=0.0,
    )
    wrapped = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="delta",
        delta_action_masks=[],
    )
    sample = wrapped[0]
    np.testing.assert_array_equal(sample["reference_actions"], np.ones((4, 14), dtype=np.float32))
    assert bool(sample["reference_action_mask"])
    assert not bool(np.asarray(sample["meta_areas"]["mask"]).reshape(-1)[0])


def test_beta_reference_actions_are_delta_relative_to_condition_state():
    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            state = np.arange(32, dtype=np.float32)
            actions = np.tile(state[None, :], (4, 1)).astype(np.float32)
            actions[:, 0:6] += 2.0
            actions[:, 7:13] -= 3.0
            actions[:, 14:] = 99.0
            return {
                "state": state,
                "actions": actions,
                "meta_areas": {
                    "pose12d": np.zeros((1, 12), dtype=np.float32),
                    "dim_mask12": np.ones((1, 12), dtype=bool),
                    "type": np.zeros((1,), dtype=np.int32),
                    "mask": np.ones((1,), dtype=bool),
                },
                "meta_action_targets": {
                    "pose12d": np.zeros((4, 1, 12), dtype=np.float32),
                    "dim_mask12": np.ones((4, 1, 12), dtype=bool),
                    "mask": np.ones((4, 1), dtype=bool),
                },
                "tool_instance_hash": np.asarray([7], dtype=np.int32),
                "source_type_id": np.asarray([0], dtype=np.int32),
                "episode_index": np.asarray(idx, dtype=np.int64),
            }

    delta_mask = _transforms.make_bool_mask(6, -1, 6, -1, -18)
    transformed = _data_loader.TransformedDataset(TinyDataset(), [_transforms.DeltaActions(delta_mask)])
    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_beta_seed=0,
        meta_beta_self_same_chunk_prob=1.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=0.0,
        meta_beta_meta_area_condition_prob=0.0,
        meta_beta_reference_action_condition_prob=1.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_non_retarget_obs_only_condition_prob=0.0,
    )
    sample = _data_loader.BetaStructuredMetaPairDataset(
        transformed,
        data_config,
        expected_action_space="delta",
        delta_action_masks=[np.asarray(delta_mask, dtype=bool)],
    )[0]

    expected = np.zeros((4, 14), dtype=np.float32)
    expected[:, 0:6] = 2.0
    expected[:, 6] = 6.0
    expected[:, 7:13] = -3.0
    expected[:, 13] = 13.0
    np.testing.assert_array_equal(sample["condition_state"], np.arange(32, dtype=np.float32))
    np.testing.assert_array_equal(sample["reference_actions"], expected)


def test_beta_pair_dataset_reference_condition_carries_source_observation():
    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            sample = _tiny_beta_sample(int(idx), tool=7, episode=0)
            image = np.full((4, 4, 3), 17, dtype=np.uint8)
            sample["image"] = {
                "base_0_rgb": image,
                "left_wrist_0_rgb": image + 1,
                "right_wrist_0_rgb": image + 2,
            }
            sample["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }
            sample["state"] = np.arange(32, dtype=np.float32)
            return sample

    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_beta_seed=0,
        meta_beta_self_same_chunk_prob=1.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=0.0,
        meta_beta_meta_area_condition_prob=0.0,
        meta_beta_reference_action_condition_prob=1.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_non_retarget_obs_only_condition_prob=0.0,
    )
    sample = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="delta",
        delta_action_masks=[],
    )[0]
    np.testing.assert_array_equal(sample["condition_image"]["base_0_rgb"], np.full((4, 4, 3), 17, dtype=np.uint8))
    np.testing.assert_array_equal(sample["condition_state"], np.arange(32, dtype=np.float32))


def test_beta_pair_dataset_meta_and_obs_conditions():
    class TinyDataset:
        def __init__(self, source_type: int):
            self.source_type = source_type

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return _tiny_beta_sample(int(idx), tool=7, episode=0, source_type=self.source_type)

    base = {
        "meta_beta_seed": 0,
        "meta_beta_self_same_chunk_prob": 1.0,
        "meta_beta_same_episode_diff_chunk_prob": 0.0,
        "meta_beta_same_tool_diff_episode_prob": 0.0,
        "meta_beta_retarget_conditioned_prob": 0.0,
        "meta_beta_reference_action_condition_prob": 0.0,
    }
    meta_config = dataclasses.replace(
        _config.DataConfig(),
        **base,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_non_retarget_obs_only_condition_prob=0.0,
    )
    meta_sample = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(source_type=1),
        meta_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )[0]
    assert bool(meta_sample["meta_areas"]["mask"][0])
    np.testing.assert_array_equal(
        meta_sample["execution_meta_areas"]["pose12d"],
        TinyDataset(source_type=1)[0]["meta_areas"]["pose12d"],
    )
    assert not bool(meta_sample["reference_action_mask"])
    assert float(meta_sample["meta_control"]["imagination_alpha"]) == 0.0

    obs_config = dataclasses.replace(
        _config.DataConfig(),
        **base,
        meta_beta_meta_area_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=1.0,
        meta_beta_non_retarget_obs_only_condition_prob=1.0,
    )
    obs_sample = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(source_type=0),
        obs_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )[0]
    assert not bool(obs_sample["meta_areas"]["mask"][0])
    assert bool(obs_sample["execution_meta_areas"]["mask"][0])
    assert not bool(obs_sample["reference_action_mask"])
    assert float(obs_sample["meta_control"]["imagination_alpha"]) == 0.0


def test_beta_pair_dataset_obs_only_uses_source_observation_without_meta_tokens():
    class TinyDataset:
        def __init__(self):
            self.samples = [
                _tiny_beta_sample(0, tool=7, episode=0, source_type=0),
                _tiny_beta_sample(1, tool=7, episode=1, source_type=0),
            ]
            for idx, sample in enumerate(self.samples):
                image = np.full((4, 4, 3), 10 + idx, dtype=np.uint8)
                sample["image"] = {
                    "base_0_rgb": image,
                    "left_wrist_0_rgb": image + 1,
                    "right_wrist_0_rgb": image + 2,
                }
                sample["image_mask"] = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                }
                sample["state"] = np.full((32,), idx + 3, dtype=np.float32)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[int(idx)]

    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_beta_seed=1,
        meta_beta_self_same_chunk_prob=0.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=1.0,
        meta_beta_retarget_conditioned_prob=0.0,
        meta_beta_meta_area_condition_prob=0.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_non_retarget_obs_only_condition_prob=1.0,
    )
    sample = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )[0]
    assert not bool(sample["meta_areas"]["mask"][0])
    assert not bool(sample["reference_action_mask"])
    assert "condition_image" in sample
    assert "condition_state" in sample
    assert not np.array_equal(sample["condition_image"]["base_0_rgb"], sample["image"]["base_0_rgb"])


def test_beta_pair_dataset_imagined_source_never_uses_obs_only_condition():
    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return _tiny_beta_sample(int(idx), tool=7, episode=0, source_type=1)

    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_beta_seed=0,
        meta_beta_self_same_chunk_prob=1.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=0.0,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=1.0,
        meta_beta_non_retarget_obs_only_condition_prob=1.0,
    )
    sample = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )[0]

    assert int(sample["_beta_debug"]["condition_id"]) == 0
    assert bool(sample["meta_areas"]["mask"][0])
    assert not bool(sample["reference_action_mask"])
    assert float(sample["meta_control"]["imagination_alpha"]) == 0.0


def test_beta_non_retarget_condition_distribution_uses_ten_percent_obs_only():
    probs = _data_loader._condition_probabilities_with_obs_fraction(0.45, 0.50, 0.10)  # noqa: SLF001

    np.testing.assert_allclose(probs, np.asarray([0.42631579, 0.47368421, 0.10]), rtol=1e-6)
    np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-6)


def _tiny_beta_sample(idx: int, *, tool: int, episode: int, source_type: int = 0) -> dict:
    pose = np.zeros((1, 12), dtype=np.float32)
    pose[0, :3] = [float(idx), 0.0, 0.0]
    pose[0, 3:9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_pose = np.tile(pose[None, :, :], (4, 1, 1))
    return {
        "state": np.zeros((32,), dtype=np.float32),
        "actions": np.full((4, 32), idx + 1, dtype=np.float32),
        "meta_areas": {
            "pose12d": pose,
            "dim_mask12": np.ones((1, 12), dtype=bool),
            "type": np.zeros((1,), dtype=np.int32),
            "mask": np.ones((1,), dtype=bool),
        },
        "meta_action_targets": {
            "pose12d": target_pose,
            "dim_mask12": np.ones((4, 1, 12), dtype=bool),
            "mask": np.ones((4, 1), dtype=bool),
        },
        "tool_instance_hash": np.asarray([tool], dtype=np.int32),
        "source_type_id": np.asarray([source_type], dtype=np.int32),
        "episode_index": np.asarray(episode, dtype=np.int64),
    }


def test_beta_pair_dataset_retarget_condition_uses_source_and_retarget_target(monkeypatch):
    class TinyDataset:
        def __init__(self):
            self.samples = [
                _tiny_beta_sample(0, tool=1, episode=0),
                _tiny_beta_sample(1, tool=2, episode=1),
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[int(idx)]

    def fake_pair_retarget(target_data, source_data, *, rng, config):
        del rng, config
        assert np.asarray(source_data["tool_instance_hash"]).reshape(-1)[0] in (1, 2)
        actions = np.full_like(target_data["actions"], 9.0)
        retargeted_meta = np.asarray(target_data["meta_areas"]["pose12d"], dtype=np.float32).copy()
        retargeted_meta[:, :3] = 77.0
        return _meta_retarget.MetaRetargetResult(
            state=np.asarray(target_data["state"], dtype=np.float32),
            actions=actions,
            meta_area_pose6d=np.zeros((1, 6), dtype=np.float32),
            meta_area_type=np.array([_meta_retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
            meta_area_mask=np.array([True], dtype=bool),
            meta_area_pose12d=retargeted_meta,
            meta_area_dim_mask12=np.asarray(target_data["meta_areas"]["dim_mask12"], dtype=bool),
            meta_action_target_pose12d=np.asarray(target_data["meta_action_targets"]["pose12d"], dtype=np.float32),
            meta_action_target_dim_mask12=np.asarray(target_data["meta_action_targets"]["dim_mask12"], dtype=bool),
            meta_action_target_mask=np.asarray(target_data["meta_action_targets"]["mask"], dtype=bool),
            diagnostics=_meta_retarget.MetaRetargetDiagnostics(
                accepted=True,
                area_type="line",
                horizon=4,
                approach_steps=0,
                ik_nonconverged=0,
                max_position_error_m=0.0,
                max_direction_error_rad=0.0,
                max_step_joint_delta_rad=0.0,
                retarget_mode="future_near",
                trajectory_start_index=0,
            ),
        )

    monkeypatch.setattr(_data_loader._meta_retarget, "generate_pair_retargeted_chunk", fake_pair_retarget)  # noqa: SLF001
    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_retarget_cache_prob=1.0,
        meta_beta_seed=1,
        meta_beta_self_same_chunk_prob=0.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=1.0,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=0.0,
    )
    wrapped = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )

    sample = wrapped[0]

    assert bool(sample["_beta_debug"]["retarget_applied"])
    assert int(sample["_beta_debug"]["retarget_mode_id"]) == 0
    assert int(sample["_beta_debug"]["retarget_source_id"]) == 2
    np.testing.assert_array_equal(sample["actions"], np.full((4, 32), 9.0, dtype=np.float32))
    assert not bool(sample["reference_action_mask"])
    assert bool(sample["meta_areas"]["mask"][0])
    np.testing.assert_array_equal(sample["execution_meta_areas"]["pose12d"][:, :3], np.full((1, 3), 77.0))
    assert float(sample["meta_control"]["imagination_alpha"]) == 0.0


def test_beta_pair_dataset_retarget_failure_retries_random_target(monkeypatch):
    class TinyDataset:
        def __init__(self):
            self.samples = [
                _tiny_beta_sample(0, tool=1, episode=0),
                _tiny_beta_sample(1, tool=2, episode=1),
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[int(idx)]

    attempted_target_positions = []

    def fake_pair_retarget(target_data, source_data, *, rng, config):
        del source_data, rng, config
        target_pos = float(np.asarray(target_data["meta_areas"]["pose12d"])[0, 0])
        attempted_target_positions.append(target_pos)
        if target_pos == 0.0:
            return None
        actions = np.full_like(target_data["actions"], 8.0)
        return _meta_retarget.MetaRetargetResult(
            state=np.asarray(target_data["state"], dtype=np.float32),
            actions=actions,
            meta_area_pose6d=np.zeros((1, 6), dtype=np.float32),
            meta_area_type=np.array([_meta_retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
            meta_area_mask=np.array([True], dtype=bool),
            meta_area_pose12d=np.asarray(target_data["meta_areas"]["pose12d"], dtype=np.float32),
            meta_area_dim_mask12=np.asarray(target_data["meta_areas"]["dim_mask12"], dtype=bool),
            meta_action_target_pose12d=np.asarray(target_data["meta_action_targets"]["pose12d"], dtype=np.float32),
            meta_action_target_dim_mask12=np.asarray(target_data["meta_action_targets"]["dim_mask12"], dtype=bool),
            meta_action_target_mask=np.asarray(target_data["meta_action_targets"]["mask"], dtype=bool),
            diagnostics=_meta_retarget.MetaRetargetDiagnostics(
                accepted=True,
                area_type="line",
                horizon=4,
                approach_steps=0,
                ik_nonconverged=0,
                max_position_error_m=0.0,
                max_direction_error_rad=0.0,
                max_step_joint_delta_rad=0.0,
                retarget_mode="correction",
                trajectory_start_index=0,
            ),
        )

    monkeypatch.setattr(_data_loader._meta_retarget, "generate_pair_retargeted_chunk", fake_pair_retarget)  # noqa: SLF001
    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_retarget_cache_prob=1.0,
        meta_beta_pair_retarget_max_attempts=2,
        meta_beta_seed=1,
        meta_beta_self_same_chunk_prob=0.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=1.0,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=0.0,
    )
    wrapped = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )

    sample = wrapped[0]

    assert attempted_target_positions[0] == 0.0
    assert any(pos != 0.0 for pos in attempted_target_positions[1:])
    assert bool(sample["_beta_debug"]["retarget_applied"])
    assert int(sample["_beta_debug"]["relation_id"]) == 3
    np.testing.assert_array_equal(sample["actions"], np.full((4, 32), 8.0, dtype=np.float32))


def test_beta_pair_dataset_failed_retarget_falls_back_to_origin(monkeypatch):
    class TinyDataset:
        def __init__(self):
            self.samples = [
                _tiny_beta_sample(0, tool=1, episode=0),
                _tiny_beta_sample(1, tool=2, episode=1, source_type=1),
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[int(idx)]

    def fake_pair_retarget(target_data, source_data, *, rng, config):
        del target_data, source_data, rng, config

    monkeypatch.setattr(_data_loader._meta_retarget, "generate_pair_retargeted_chunk", fake_pair_retarget)  # noqa: SLF001
    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_retarget_cache_prob=1.0,
        meta_beta_seed=1,
        meta_beta_self_same_chunk_prob=0.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=1.0,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_non_retarget_obs_only_condition_prob=0.0,
    )
    wrapped = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )

    sample = wrapped[0]

    assert not bool(sample["_beta_debug"]["retarget_applied"])
    assert int(sample["_beta_debug"]["relation_id"]) == 0
    assert int(sample["_beta_debug"]["retarget_status_id"]) == 2
    np.testing.assert_array_equal(sample["actions"], np.ones((4, 32), dtype=np.float32))
    np.testing.assert_array_equal(sample["meta_areas"]["pose12d"][:, :3], np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(sample["execution_meta_areas"]["pose12d"][:, :3], np.zeros((1, 3), dtype=np.float32))
    assert float(sample["meta_control"]["imagination_alpha"]) == 0.0


def test_beta_pair_dataset_uses_pair_cache_when_online_queue_empty(tmp_path):
    class TinyDataset:
        def __init__(self):
            self.samples = [
                _tiny_beta_sample(0, tool=1, episode=0),
                _tiny_beta_sample(1, tool=2, episode=1),
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[int(idx)]

    relpath = "pairs/000000/000000000_000000001_00.npz"
    payload_path = tmp_path / relpath
    payload_path.parent.mkdir(parents=True)
    np.savez(
        payload_path,
        state=np.full((32,), 3.0, dtype=np.float32),
        actions=np.full((4, 32), 7.0, dtype=np.float32),
        meta_area_pose12d=np.full((1, 12), 5.0, dtype=np.float32),
        meta_area_dim_mask12=np.ones((1, 12), dtype=bool),
        meta_area_type=np.array([_meta_retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
        meta_area_mask=np.array([True], dtype=bool),
        meta_action_target_pose12d=np.full((4, 1, 12), 6.0, dtype=np.float32),
        meta_action_target_dim_mask12=np.ones((4, 1, 12), dtype=bool),
        meta_action_target_mask=np.ones((4, 1), dtype=bool),
    )
    (tmp_path / "metadata.json").write_text('{"cache_action_space": "absolute"}', encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text(
        (
            '{"accepted": true, "target_index": 0, "source_index": 1, "variant_id": 0, '
            f'"path": "{relpath}", "retarget_mode": "future_near", "trajectory_start_index": 2, '
            '"approach_steps": 3}\n'
        ),
        encoding="utf-8",
    )

    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_retarget_cache_prob=1.0,
        meta_beta_pair_cache_dir=str(tmp_path),
        meta_beta_seed=0,
        meta_beta_self_same_chunk_prob=0.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=1.0,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_online_async_enabled=True,
        meta_beta_online_prefer_prob=0.0,
    )
    wrapped = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )
    sample = wrapped[0]

    assert bool(sample["_beta_debug"]["retarget_applied"])
    assert int(sample["_beta_debug"]["retarget_mode_id"]) == 0
    assert int(sample["_beta_debug"]["retarget_source_id"]) == 1
    assert int(sample["_beta_debug"]["trajectory_start_index"]) == 2
    assert int(sample["_beta_debug"]["approach_steps"]) == 3
    assert wrapped._stats_online_submitted == 0  # noqa: SLF001
    np.testing.assert_array_equal(sample["actions"], np.full((4, 32), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(sample["execution_meta_areas"]["pose12d"], np.full((1, 12), 5.0, dtype=np.float32))
    np.testing.assert_array_equal(sample["meta_areas"]["pose12d"][:, :3], np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert float(sample["meta_control"]["imagination_alpha"]) == 0.0


def test_beta_pair_dataset_consumes_ready_online_retarget(monkeypatch):
    class TinyDataset:
        def __init__(self):
            self.samples = [
                _tiny_beta_sample(0, tool=1, episode=0),
                _tiny_beta_sample(1, tool=2, episode=1),
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[int(idx)]

    def fake_pair_retarget(target_data, source_data, *, rng, config):
        del source_data, rng, config
        actions = np.full_like(target_data["actions"], 11.0)
        return _meta_retarget.MetaRetargetResult(
            state=np.asarray(target_data["state"], dtype=np.float32),
            actions=actions,
            meta_area_pose6d=np.zeros((1, 6), dtype=np.float32),
            meta_area_type=np.array([_meta_retarget.META_AREA_TYPE_TO_ID["line"]], dtype=np.int32),
            meta_area_mask=np.array([True], dtype=bool),
            meta_area_pose12d=np.asarray(target_data["meta_areas"]["pose12d"], dtype=np.float32),
            meta_area_dim_mask12=np.asarray(target_data["meta_areas"]["dim_mask12"], dtype=bool),
            meta_action_target_pose12d=np.asarray(target_data["meta_action_targets"]["pose12d"], dtype=np.float32),
            meta_action_target_dim_mask12=np.asarray(target_data["meta_action_targets"]["dim_mask12"], dtype=bool),
            meta_action_target_mask=np.asarray(target_data["meta_action_targets"]["mask"], dtype=bool),
            diagnostics=_meta_retarget.MetaRetargetDiagnostics(
                accepted=True,
                area_type="line",
                horizon=4,
                approach_steps=0,
                ik_nonconverged=0,
                max_position_error_m=0.0,
                max_direction_error_rad=0.0,
                max_step_joint_delta_rad=0.0,
                retarget_mode="future_near",
                trajectory_start_index=0,
            ),
        )

    monkeypatch.setattr(_data_loader._meta_retarget, "generate_pair_retargeted_chunk", fake_pair_retarget)  # noqa: SLF001
    data_config = dataclasses.replace(
        _config.DataConfig(),
        meta_retarget_cache_prob=1.0,
        meta_beta_seed=0,
        meta_beta_self_same_chunk_prob=0.0,
        meta_beta_same_episode_diff_chunk_prob=0.0,
        meta_beta_same_tool_diff_episode_prob=0.0,
        meta_beta_retarget_conditioned_prob=1.0,
        meta_beta_meta_area_condition_prob=1.0,
        meta_beta_reference_action_condition_prob=0.0,
        meta_beta_obs_only_condition_prob=0.0,
        meta_beta_online_async_enabled=True,
        meta_beta_online_queue_size=2,
    )
    wrapped = _data_loader.BetaStructuredMetaPairDataset(
        TinyDataset(),
        data_config,
        expected_action_space="absolute",
        delta_action_masks=[],
    )
    for _ in range(20):
        wrapped._poll_and_prefill_online_retarget_queue()  # noqa: SLF001
        if wrapped._online_ready:  # noqa: SLF001
            break
        time.sleep(0.01)

    sample = wrapped[0]

    assert bool(sample["_beta_debug"]["retarget_applied"])
    assert int(sample["_beta_debug"]["retarget_mode_id"]) == 0
    assert int(sample["_beta_debug"]["retarget_source_id"]) == 0
    np.testing.assert_array_equal(sample["actions"], np.full((4, 32), 11.0, dtype=np.float32))
    assert float(sample["meta_control"]["imagination_alpha"]) == 0.0
    if wrapped._online_executor is not None:  # noqa: SLF001
        wrapped._online_executor.shutdown(wait=False, cancel_futures=True)  # noqa: SLF001


def test_normalize_applies_action_stats_to_reference_actions():
    stats = {
        "actions": _transforms.NormStats(
            mean=np.asarray([1.0, 2.0], dtype=np.float32),
            std=np.asarray([2.0, 4.0], dtype=np.float32),
            q01=None,
            q99=None,
        )
    }
    item = {
        "actions": np.asarray([[3.0, 10.0]], dtype=np.float32),
        "reference_actions": np.asarray([[5.0, 6.0]], dtype=np.float32),
    }
    out = _transforms.Normalize(stats)(item)
    np.testing.assert_allclose(out["actions"], np.asarray([[1.0, 2.0]], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(out["reference_actions"], np.asarray([[2.0, 1.0]], dtype=np.float32), rtol=1e-5)
