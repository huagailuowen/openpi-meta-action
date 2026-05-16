# ruff: noqa: SLF001

from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import ClassVar, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.xtrainer_meta_retarget as _meta_retarget
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class RetargetCacheDataset(Dataset[T_co]):
    """Sample precomputed x-trainer meta-retarget chunks before normalization.

    The wrapped dataset is expected to already be in canonical OpenPI format
    with ``state``, ``actions`` and ``meta_areas`` fields. Images and prompts are
    kept from the original sample; cached files only replace the low-dimensional
    supervision fields.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: str | pathlib.Path,
        sample_prob: float,
        seed: int,
        expected_action_space: str,
    ):
        self._dataset = dataset
        self._cache_dir = pathlib.Path(cache_dir).expanduser()
        self._sample_prob = float(sample_prob)
        self._rng = np.random.default_rng(seed)
        self._records_by_index = self._load_manifest(self._cache_dir)
        self._warn_if_action_space_mismatch(self._cache_dir, expected_action_space)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = self._dataset[index]
        if self._sample_prob <= 0.0:
            return sample

        base_index = int(index.__index__())
        records = self._records_by_index.get(base_index)
        if not records or self._rng.random() >= self._sample_prob:
            return sample

        record = records[int(self._rng.integers(len(records)))]
        variant_path = self._cache_dir / record["path"]
        try:
            with np.load(variant_path) as cached:
                retargeted = {key: cached[key].copy() for key in cached.files}
        except FileNotFoundError:
            logging.warning("Retarget cache entry missing: %s", variant_path)
            return sample

        return typing.cast(T_co, _apply_retargeted_payload(sample, retargeted))

    def __len__(self) -> int:
        return len(self._dataset)

    @staticmethod
    def _load_manifest(cache_dir: pathlib.Path) -> dict[int, list[dict[str, typing.Any]]]:
        manifest_path = cache_dir / "manifest.jsonl"
        if not manifest_path.exists():
            logging.warning("Retarget cache manifest not found: %s", manifest_path)
            return {}

        records_by_index: dict[int, list[dict[str, typing.Any]]] = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                record = json.loads(stripped_line)
                if not bool(record.get("accepted", True)):
                    continue
                base_index = int(record["base_index"])
                records_by_index.setdefault(base_index, []).append(record)
        logging.info(
            "Loaded %d retarget cache variants for %d base chunks from %s",
            sum(len(v) for v in records_by_index.values()),
            len(records_by_index),
            manifest_path,
        )
        return records_by_index

    @staticmethod
    def _warn_if_action_space_mismatch(cache_dir: pathlib.Path, expected_action_space: str) -> None:
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            if expected_action_space == "delta":
                logging.warning(
                    "Retarget cache metadata not found at %s; rebuild older delta caches to ensure cached actions "
                    "are stored in delta action space.",
                    metadata_path,
                )
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Could not parse retarget cache metadata: %s", metadata_path)
            return

        actual_action_space = metadata.get("cache_action_space")
        if actual_action_space is None:
            if expected_action_space == "delta":
                logging.warning(
                    "Retarget cache metadata at %s does not record cache_action_space; rebuild older delta caches "
                    "to avoid mixing absolute actions into a delta training pipeline.",
                    metadata_path,
                )
            return

        if actual_action_space != expected_action_space:
            logging.warning(
                "Retarget cache action space mismatch: cache has %s actions but this data config expects %s actions.",
                actual_action_space,
                expected_action_space,
            )


def _clone_sample(sample: dict[str, typing.Any]) -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = dict(sample)
    for key in ("meta_areas", "execution_meta_areas", "meta_action_targets", "meta_control"):
        if isinstance(out.get(key), dict):
            out[key] = dict(out[key])
    return out


def _copy_meta_areas(meta_areas: dict[str, typing.Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in meta_areas.items()}


def _apply_retargeted_payload(sample: dict[str, typing.Any], retargeted: dict[str, np.ndarray]) -> dict[str, typing.Any]:
    out = _clone_sample(sample)
    out["state"] = retargeted["state"].astype(np.float32)
    out["actions"] = retargeted["actions"].astype(np.float32)
    meta_areas = dict(out.get("meta_areas", {}))
    if "meta_area_pose12d" in retargeted:
        meta_areas["pose12d"] = retargeted["meta_area_pose12d"].astype(np.float32)
        meta_areas.pop("pose6d", None)
        if "meta_area_dim_mask12" in retargeted:
            meta_areas["dim_mask12"] = retargeted["meta_area_dim_mask12"].astype(bool)
    else:
        meta_areas["pose6d"] = retargeted["meta_area_pose6d"].astype(np.float32)
        meta_areas.pop("pose12d", None)
        meta_areas.pop("dim_mask12", None)
    meta_areas["type"] = retargeted["meta_area_type"].astype(np.int32)
    meta_areas["mask"] = retargeted["meta_area_mask"].astype(bool)
    out["meta_areas"] = meta_areas
    if "meta_action_target_pose12d" in retargeted:
        meta_targets = dict(out.get("meta_action_targets", {}))
        meta_targets["pose12d"] = retargeted["meta_action_target_pose12d"].astype(np.float32)
        meta_targets.pop("pose6d", None)
        if "meta_action_target_dim_mask12" in retargeted:
            meta_targets["dim_mask12"] = retargeted["meta_action_target_dim_mask12"].astype(bool)
        if "meta_action_target_mask" in retargeted:
            meta_targets["mask"] = retargeted["meta_action_target_mask"].astype(bool)
        else:
            meta_targets["mask"] = np.ones(meta_targets["pose12d"].shape[:2], dtype=bool)
        out["meta_action_targets"] = meta_targets
    return out


class AlphaMetaRetargetDataset(Dataset[T_co]):
    """Alpha-controlled structured-meta augmentation before normalization.

    Original samples are kept at a fixed probability. The remaining probability
    mass is split continuously between cached retarget samples and counterfactual
    samples as a sigmoid function of alpha.
    """

    def __init__(
        self,
        dataset: Dataset,
        data_config: _config.DataConfig,
        expected_action_space: str,
        delta_action_masks: Sequence[np.ndarray],
    ):
        self._dataset = dataset
        self._cache_dir = pathlib.Path(data_config.meta_retarget_cache_dir).expanduser() if data_config.meta_retarget_cache_dir else None
        self._rng = np.random.default_rng(data_config.meta_alpha_seed)
        self._records_by_index = (
            RetargetCacheDataset._load_manifest(self._cache_dir) if self._cache_dir is not None else {}
        )
        if self._cache_dir is not None:
            RetargetCacheDataset._warn_if_action_space_mismatch(self._cache_dir, expected_action_space)
        self._delta_action_masks = [np.asarray(mask, dtype=bool) for mask in delta_action_masks]
        self._original_prob = float(np.clip(data_config.meta_alpha_original_prob, 0.0, 1.0))
        self._sigmoid_k = float(data_config.meta_alpha_sigmoid_k)
        self._retarget_scale_min = float(data_config.meta_alpha_retarget_scale_min)
        self._retarget_scale_max = float(data_config.meta_alpha_retarget_scale_max)
        self._cf_pos_full_m = float(data_config.meta_alpha_counterfactual_pos_full_m)
        self._cf_shape_full_deg = float(data_config.meta_alpha_counterfactual_shape_full_deg)
        self._cf_approach_full_deg = float(data_config.meta_alpha_counterfactual_approach_full_deg)
        self._cf_scale_floor = float(np.clip(data_config.meta_alpha_counterfactual_scale_floor, 0.0, 1.0))
        self._cf_near_target_prob = float(np.clip(data_config.meta_alpha_counterfactual_near_target_prob, 0.0, 1.0))
        self._near_pos_m = float(data_config.meta_alpha_near_target_pos_max_m)
        self._near_shape_deg = float(data_config.meta_alpha_near_target_shape_max_deg)
        self._near_approach_deg = float(data_config.meta_alpha_near_target_approach_max_deg)
        self._helpers: dict[str, typing.Any] | None = None

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = typing.cast(dict[str, typing.Any], self._dataset[index])
        alpha = float(self._rng.random())
        out = self._choose_augmented_sample(sample, int(index.__index__()), alpha)
        out = _clone_sample(out)
        out["meta_control"] = {**dict(out.get("meta_control", {})), "alpha": np.asarray(alpha, dtype=np.float32)}
        return typing.cast(T_co, out)

    def __len__(self) -> int:
        return len(self._dataset)

    def _choose_augmented_sample(
        self, sample: dict[str, typing.Any], base_index: int, alpha: float
    ) -> dict[str, typing.Any]:
        if self._rng.random() < self._original_prob:
            return sample

        q_retarget = _sigmoid(self._sigmoid_k * (alpha - 0.5))
        if self._rng.random() < q_retarget:
            retargeted = self._sample_retarget_payload(base_index, alpha)
            if retargeted is not None:
                return _apply_retargeted_payload(sample, retargeted)
            return sample
        return self._make_counterfactual_sample(sample, alpha)

    def _sample_retarget_payload(self, base_index: int, alpha: float) -> dict[str, np.ndarray] | None:
        if self._cache_dir is None:
            return None
        records = self._records_by_index.get(base_index)
        if not records:
            return None
        desired_scale = self._retarget_scale_min + (self._retarget_scale_max - self._retarget_scale_min) * alpha
        scaled_records = [record for record in records if "retarget_scale" in record]
        if scaled_records:
            record = min(scaled_records, key=lambda r: abs(float(r["retarget_scale"]) - desired_scale))
        else:
            record = records[int(self._rng.integers(len(records)))]
        try:
            with np.load(self._cache_dir / record["path"]) as cached:
                return {key: cached[key].copy() for key in cached.files}
        except FileNotFoundError:
            logging.warning("Retarget cache entry missing: %s", self._cache_dir / record["path"])
            return None

    def _make_counterfactual_sample(self, sample: dict[str, typing.Any], alpha: float) -> dict[str, typing.Any]:
        meta_areas = sample.get("meta_areas")
        if not isinstance(meta_areas, dict) or "pose12d" not in meta_areas:
            return sample
        masks = np.asarray(meta_areas.get("mask", []), dtype=bool)
        if masks.size == 0 or not bool(masks[0]):
            return sample

        out = _clone_sample(sample)
        out_meta = dict(out["meta_areas"])
        poses = np.asarray(out_meta["pose12d"], dtype=np.float32).copy()
        dim_masks = np.asarray(out_meta.get("dim_mask12", np.ones_like(poses, dtype=bool)), dtype=bool).copy()
        types = np.asarray(out_meta.get("type", np.zeros((poses.shape[0],), dtype=np.int32)), dtype=np.int32)
        area_type = _meta_retarget.META_AREA_ID_TO_TYPE.get(int(types.reshape(-1)[0]), "line")
        dim_mask = dim_masks[0]

        scale = self._counterfactual_scale(alpha)
        poses[0] = _perturb_pose12(
            poses[0],
            area_type,
            dim_mask,
            self._rng,
            position_max_m=self._cf_pos_full_m * scale,
            shape_max_deg=self._cf_shape_full_deg * scale,
            approach_max_deg=self._cf_approach_full_deg * scale,
        )
        out_meta["pose12d"] = poses
        out["meta_areas"] = out_meta

        if self._rng.random() < self._cf_near_target_prob:
            out = self._with_near_original_meta_targets(out, area_type, dim_mask, poses_before=np.asarray(meta_areas["pose12d"], dtype=np.float32))
        return out

    def _counterfactual_scale(self, alpha: float) -> float:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return self._cf_scale_floor + (1.0 - self._cf_scale_floor) * (1.0 - alpha * alpha)

    def _helpers_once(self) -> dict[str, typing.Any]:
        if self._helpers is None:
            self._helpers = _meta_retarget._load_xtrainer_helpers()
        return self._helpers

    def _with_near_original_meta_targets(
        self,
        sample: dict[str, typing.Any],
        area_type: str,
        dim_mask: np.ndarray,
        *,
        poses_before: np.ndarray,
    ) -> dict[str, typing.Any]:
        meta_targets = sample.get("meta_action_targets")
        if not isinstance(meta_targets, dict) or "pose12d" not in meta_targets:
            return sample

        state = np.asarray(sample["state"], dtype=np.float32)
        actions = np.asarray(sample["actions"], dtype=np.float32)
        if state.shape[-1] < 14 or actions.ndim != 2 or actions.shape[-1] < 14:
            return sample

        small_input_pose = _perturb_pose12(
            poses_before[0],
            area_type,
            dim_mask,
            self._rng,
            position_max_m=self._near_pos_m,
            shape_max_deg=self._near_shape_deg,
            approach_max_deg=self._near_approach_deg,
        )
        helpers = self._helpers_once()
        t_state_wrist = np.asarray(helpers["fk"](state[:14], "right_wrist"), dtype=np.float32)
        wrist_pose12 = _meta_retarget._base_pose12_to_wrist_pose(small_input_pose, t_state_wrist)
        absolute_actions = _absolute_actions_for_fk(state, actions, self._delta_action_masks)
        target_pose12 = np.asarray(meta_targets["pose12d"], dtype=np.float32).copy()
        if target_pose12.ndim == 2:
            target_pose12 = target_pose12[:, None, :]
        for step in range(min(target_pose12.shape[0], absolute_actions.shape[0])):
            t_base_wrist = np.asarray(helpers["fk"](absolute_actions[step, :14], "right_wrist"), dtype=np.float32)
            target_pose12[step, 0, :] = _meta_retarget._wrist_pose12_to_base_pose(wrist_pose12, t_base_wrist)

        out = _clone_sample(sample)
        out_targets = dict(meta_targets)
        out_targets["pose12d"] = target_pose12.astype(np.float32)
        out["meta_action_targets"] = out_targets
        return out


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _perturb_pose12(
    pose12: np.ndarray,
    area_type: str,
    dim_mask12: np.ndarray,
    rng: np.random.Generator,
    *,
    position_max_m: float,
    shape_max_deg: float,
    approach_max_deg: float,
) -> np.ndarray:
    pose = np.asarray(pose12, dtype=np.float32).reshape(12).copy()
    dim_mask = np.asarray(dim_mask12, dtype=bool).reshape(12)
    pose[:3] += _meta_retarget._sample_random_vector_in_ball(rng, position_max_m)
    if area_type == "point":
        pose[3:12] = 0.0
        return pose.astype(np.float32)

    if np.any(dim_mask[3:9]) and shape_max_deg > 0.0:
        r_shape = _meta_retarget._random_small_rotation(rng, shape_max_deg)
        shape_matrix = _meta_retarget._shape6_to_matrix(pose[3:9])
        pose[3:9] = _meta_retarget._matrix_to_shape6(
            _meta_retarget._project_psd_trace1(r_shape @ shape_matrix @ r_shape.T)
        )
    if np.any(dim_mask[9:12]) and approach_max_deg > 0.0 and float(np.linalg.norm(pose[9:12])) > 1e-8:
        r_app = _meta_retarget._random_small_rotation(rng, approach_max_deg)
        pose[9:12] = _meta_retarget._normalize(r_app @ pose[9:12])
    elif not np.any(dim_mask[9:12]):
        pose[9:12] = 0.0
    return pose.astype(np.float32)


def _absolute_actions_for_fk(
    state: np.ndarray,
    actions: np.ndarray,
    delta_action_masks: Sequence[np.ndarray],
) -> np.ndarray:
    absolute = np.asarray(actions, dtype=np.float32).copy()
    state = np.asarray(state, dtype=np.float32)
    for mask_values in delta_action_masks:
        mask = np.asarray(mask_values, dtype=bool)
        dims = int(mask.shape[-1])
        if absolute.shape[-1] < dims or state.shape[-1] < dims:
            continue
        absolute[..., :dims] += np.expand_dims(np.where(mask, state[:dims], 0.0), axis=0)
    return absolute


def _condition_probabilities(meta_prob: float, reference_prob: float, obs_prob: float) -> np.ndarray:
    probs = np.asarray([meta_prob, reference_prob, obs_prob], dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if total <= 1e-8:
        return np.asarray([0.45, 0.50, 0.05], dtype=np.float64)
    return probs / total


def _condition_probabilities_with_obs_fraction(
    meta_prob: float,
    reference_prob: float,
    obs_fraction: float,
) -> np.ndarray:
    obs_fraction = float(np.clip(obs_fraction, 0.0, 1.0))
    meta_reference = np.asarray([meta_prob, reference_prob], dtype=np.float64)
    meta_reference = np.maximum(meta_reference, 0.0)
    total = float(np.sum(meta_reference))
    if total <= 1e-8:
        return np.asarray([(1.0 - obs_fraction) * 0.5, (1.0 - obs_fraction) * 0.5, obs_fraction], dtype=np.float64)
    scaled = meta_reference / total * (1.0 - obs_fraction)
    return np.asarray([scaled[0], scaled[1], obs_fraction], dtype=np.float64)


def _retarget_result_to_payload(
    result: _meta_retarget.MetaRetargetResult,
    *,
    delta_action_masks: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    actions = np.asarray(result.actions, dtype=np.float32).copy()
    for mask_values in delta_action_masks:
        mask = np.asarray(mask_values, dtype=bool)
        dims = int(mask.shape[-1])
        if actions.shape[-1] < dims or result.state.shape[-1] < dims:
            continue
        actions[..., :dims] -= np.expand_dims(np.where(mask, result.state[:dims], 0.0), axis=0)

    payload = {
        "state": np.asarray(result.state, dtype=np.float32),
        "actions": actions.astype(np.float32),
        "meta_area_pose6d": np.asarray(result.meta_area_pose6d, dtype=np.float32),
        "meta_area_type": np.asarray(result.meta_area_type, dtype=np.int32),
        "meta_area_mask": np.asarray(result.meta_area_mask, dtype=bool),
    }
    if result.meta_area_pose12d is not None:
        payload["meta_area_pose12d"] = np.asarray(result.meta_area_pose12d, dtype=np.float32)
    if result.meta_area_dim_mask12 is not None:
        payload["meta_area_dim_mask12"] = np.asarray(result.meta_area_dim_mask12, dtype=bool)
    if result.meta_action_target_pose12d is not None:
        payload["meta_action_target_pose12d"] = np.asarray(result.meta_action_target_pose12d, dtype=np.float32)
    if result.meta_action_target_dim_mask12 is not None:
        payload["meta_action_target_dim_mask12"] = np.asarray(result.meta_action_target_dim_mask12, dtype=bool)
    if result.meta_action_target_mask is not None:
        payload["meta_action_target_mask"] = np.asarray(result.meta_action_target_mask, dtype=bool)
    return payload


class BetaStructuredMetaPairDataset(Dataset[T_co]):
    """Sample beta latent conditions from a chunk1/chunk2 pair.

    This first beta version keeps retarget target generation compatible with the
    existing chunk-centric cache. The condition path is already pair-based:
    chunk1 can provide meta-area tokens, reference-action tokens, or no tokens.
    """

    _ORIGIN_SOURCE_TYPE = 0
    _RETARGET_MODE_TO_ID: ClassVar[dict[str, int]] = {"none": -1, "future_near": 0, "correction": 1}

    def __init__(
        self,
        dataset: Dataset,
        data_config: _config.DataConfig,
        expected_action_space: str,
        delta_action_masks: Sequence[np.ndarray],
    ):
        self._dataset = dataset
        self._rng = np.random.default_rng(data_config.meta_beta_seed)
        self._cache_dir = pathlib.Path(data_config.meta_retarget_cache_dir).expanduser() if data_config.meta_retarget_cache_dir else None
        self._retarget_prob = float(data_config.meta_retarget_cache_prob)
        self._delta_action_masks = [np.asarray(mask, dtype=bool) for mask in delta_action_masks]
        self._pair_retarget_max_attempts = max(1, int(data_config.meta_beta_pair_retarget_max_attempts))
        self._retarget_config = _meta_retarget.MetaRetargetGeneratorConfig()
        self._records_by_index = (
            RetargetCacheDataset._load_manifest(self._cache_dir) if self._cache_dir is not None else {}
        )
        if self._cache_dir is not None:
            RetargetCacheDataset._warn_if_action_space_mismatch(self._cache_dir, expected_action_space)

        relation_probs = np.asarray(
            [
                data_config.meta_beta_self_same_chunk_prob,
                data_config.meta_beta_same_episode_diff_chunk_prob,
                data_config.meta_beta_same_tool_diff_episode_prob,
                data_config.meta_beta_retarget_conditioned_prob,
            ],
            dtype=np.float64,
        )
        self._relation_probs = relation_probs / max(float(np.sum(relation_probs)), 1e-8)
        self._retarget_condition_probs = _condition_probabilities(
            data_config.meta_beta_meta_area_condition_prob,
            data_config.meta_beta_reference_action_condition_prob,
            data_config.meta_beta_obs_only_condition_prob,
        )
        self._non_retarget_condition_probs = _condition_probabilities_with_obs_fraction(
            data_config.meta_beta_meta_area_condition_prob,
            data_config.meta_beta_reference_action_condition_prob,
            data_config.meta_beta_non_retarget_obs_only_condition_prob,
        )

    def __getitem__(self, index: SupportsIndex) -> T_co:
        base_index = int(index.__index__())
        origin_sample = typing.cast(dict[str, typing.Any], self._dataset[index])
        target_sample = origin_sample
        relation_id = int(self._rng.choice(4, p=self._relation_probs))
        source_sample = self._sample_source(base_index, target_sample, relation_id)
        beta_debug: dict[str, typing.Any] = {"relation_id": relation_id, "retarget_applied": False}
        if relation_id == 3:
            target_sample, source_sample, beta_debug = self._maybe_apply_pair_retarget_with_retries(
                base_index,
                target_sample,
                source_sample,
            )
            if not bool(beta_debug.get("retarget_applied", False)):
                # Only fall back after exhausting fresh random target/source
                # attempts. This prevents failed/dismissed pairs from becoming
                # cross-tool non-retarget samples while avoiding an infinite loop.
                target_sample = origin_sample
                source_sample = origin_sample
                relation_id = 0
                beta_debug = {
                    **beta_debug,
                    "relation_id": relation_id,
                }

        out = _clone_sample(target_sample)
        self._set_execution_meta_from_current_meta(out)
        condition_probs = self._retarget_condition_probs if relation_id == 3 else self._non_retarget_condition_probs
        condition_id = int(self._rng.choice(3, p=condition_probs))
        if condition_id == 0:
            self._apply_meta_area_condition(out, source_sample)
        elif condition_id == 1:
            self._apply_reference_action_condition(out, source_sample)
        else:
            self._apply_obs_only_condition(out)
        self._ensure_optional_beta_fields(out)
        out["_beta_debug"] = self._stable_beta_debug(beta_debug, condition_id=condition_id)
        return typing.cast(T_co, out)

    def __len__(self) -> int:
        return len(self._dataset)

    def _sample_source(self, base_index: int, target_sample: dict[str, typing.Any], relation_id: int) -> dict[str, typing.Any]:
        if relation_id == 0:
            return target_sample
        target_tool = _scalar_int(target_sample.get("tool_instance_hash"), default=-1)
        target_episode = _scalar_int(target_sample.get("episode_index"), default=-1)
        require_same_episode = relation_id == 1
        require_same_tool = relation_id == 2
        for _ in range(24):
            candidate_index = int(self._rng.integers(len(self._dataset)))
            candidate = typing.cast(dict[str, typing.Any], self._dataset[candidate_index])
            candidate_tool = _scalar_int(candidate.get("tool_instance_hash"), default=-2)
            candidate_episode = _scalar_int(candidate.get("episode_index"), default=-2)
            if require_same_episode and (candidate_episode != target_episode or candidate_index == base_index):
                continue
            if require_same_tool and (candidate_tool != target_tool or candidate_episode == target_episode):
                continue
            return candidate
        return target_sample

    def _sample_retarget_target(self, base_index: int) -> tuple[int, dict[str, typing.Any]]:
        if len(self._dataset) <= 1:
            return base_index, typing.cast(dict[str, typing.Any], self._dataset[base_index])
        for _ in range(24):
            candidate_index = int(self._rng.integers(len(self._dataset)))
            if candidate_index != base_index:
                return candidate_index, typing.cast(dict[str, typing.Any], self._dataset[candidate_index])
        return base_index, typing.cast(dict[str, typing.Any], self._dataset[base_index])

    def _maybe_apply_pair_retarget_with_retries(
        self,
        base_index: int,
        target_sample: dict[str, typing.Any],
        initial_source_sample: dict[str, typing.Any],
    ) -> tuple[dict[str, typing.Any], dict[str, typing.Any], dict[str, typing.Any]]:
        if self._retarget_prob <= 0.0 or self._rng.random() >= self._retarget_prob:
            return target_sample, initial_source_sample, {
                "relation_id": 3,
                "retarget_applied": False,
                "retarget_status_id": 1,
            }

        for attempt in range(self._pair_retarget_max_attempts):
            if attempt == 0:
                candidate_base_index = base_index
                candidate_target = target_sample
                source_sample = initial_source_sample
            else:
                candidate_base_index, candidate_target = self._sample_retarget_target(base_index)
                source_sample = self._sample_source(candidate_base_index, candidate_target, 3)

            target_for_ik = _clone_sample(candidate_target)
            target_for_ik["actions"] = _absolute_actions_for_fk(
                np.asarray(candidate_target["state"], dtype=np.float32),
                np.asarray(candidate_target["actions"], dtype=np.float32),
                self._delta_action_masks,
            )
            result = _meta_retarget.generate_pair_retargeted_chunk(
                target_for_ik,
                source_sample,
                rng=np.random.default_rng(int(self._rng.integers(2**31 - 1))),
                config=self._retarget_config,
            )
            if result is None:
                continue
            diagnostics = result.diagnostics.to_json_dict()
            if not result.diagnostics.accepted:
                continue
            payload = _retarget_result_to_payload(result, delta_action_masks=self._delta_action_masks)
            out = _apply_retargeted_payload(candidate_target, payload)
            return out, source_sample, {
                "relation_id": 3,
                "retarget_applied": True,
                "retarget_status_id": 0,
                "retarget_attempt": int(attempt),
                **diagnostics,
            }

        return target_sample, initial_source_sample, {
            "relation_id": 3,
            "retarget_applied": False,
            "retarget_status_id": 2,
        }

    def _apply_meta_area_condition(self, out: dict[str, typing.Any], source_sample: dict[str, typing.Any]) -> None:
        out["meta_areas"] = _copy_meta_areas(dict(source_sample.get("meta_areas", {})))
        out.pop("reference_actions", None)
        out["reference_action_mask"] = np.asarray(0, dtype=bool)
        source_type = _scalar_int(source_sample.get("source_type_id"), default=self._ORIGIN_SOURCE_TYPE)
        out["meta_control"] = {
            "imagination_alpha": np.asarray(0.0 if source_type == self._ORIGIN_SOURCE_TYPE else 1.0, dtype=np.float32)
        }

    def _apply_reference_action_condition(self, out: dict[str, typing.Any], source_sample: dict[str, typing.Any]) -> None:
        out["reference_actions"] = np.asarray(source_sample["actions"], dtype=np.float32).copy()
        out["reference_action_mask"] = np.asarray(1, dtype=bool)
        self._drop_meta_tokens(out)
        out["meta_control"] = {"imagination_alpha": np.asarray(0.0, dtype=np.float32)}

    def _apply_obs_only_condition(self, out: dict[str, typing.Any]) -> None:
        out.pop("reference_actions", None)
        out["reference_action_mask"] = np.asarray(0, dtype=bool)
        self._drop_meta_tokens(out)
        out["meta_control"] = {"imagination_alpha": np.asarray(0.0, dtype=np.float32)}

    @staticmethod
    def _set_execution_meta_from_current_meta(out: dict[str, typing.Any]) -> None:
        meta_areas = out.get("meta_areas")
        if isinstance(meta_areas, dict):
            out["execution_meta_areas"] = _copy_meta_areas(meta_areas)

    @staticmethod
    def _ensure_optional_beta_fields(out: dict[str, typing.Any]) -> None:
        if "reference_actions" not in out:
            out["reference_actions"] = np.zeros_like(np.asarray(out["actions"], dtype=np.float32))
        out.setdefault("reference_action_mask", np.asarray(0, dtype=bool))
        out.setdefault("meta_control", {"imagination_alpha": np.asarray(0.0, dtype=np.float32)})

    def _stable_beta_debug(self, debug: dict[str, typing.Any], *, condition_id: int) -> dict[str, np.ndarray]:
        mode_name = str(debug.get("retarget_mode", "none"))
        return {
            "relation_id": np.asarray(int(debug.get("relation_id", -1)), dtype=np.int32),
            "condition_id": np.asarray(int(condition_id), dtype=np.int32),
            "retarget_applied": np.asarray(bool(debug.get("retarget_applied", False)), dtype=bool),
            "retarget_status_id": np.asarray(int(debug.get("retarget_status_id", -1)), dtype=np.int32),
            "retarget_mode_id": np.asarray(int(self._RETARGET_MODE_TO_ID.get(mode_name, -1)), dtype=np.int32),
            "retarget_attempt": np.asarray(int(debug.get("retarget_attempt", -1)), dtype=np.int32),
            "trajectory_start_index": np.asarray(int(debug.get("trajectory_start_index", -1)), dtype=np.int32),
            "approach_steps": np.asarray(int(debug.get("approach_steps", -1)), dtype=np.int32),
        }

    @staticmethod
    def _drop_meta_tokens(out: dict[str, typing.Any]) -> None:
        if "meta_areas" not in out:
            return
        meta_areas = dict(out["meta_areas"])
        meta_areas["mask"] = np.zeros_like(np.asarray(meta_areas["mask"], dtype=bool))
        out["meta_areas"] = meta_areas


def _scalar_int(value: typing.Any, default: int) -> int:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return int(arr.reshape(-1)[0])


def _expected_retarget_cache_action_space(data_config: _config.DataConfig) -> str:
    for transform in data_config.data_transforms.inputs:
        if transform.__class__.__name__ == "DeltaActions" and transform.mask is not None:
            return "delta"
    return "absolute"


def _delta_action_masks(data_config: _config.DataConfig) -> list[np.ndarray]:
    return [
        np.asarray(transform.mask, dtype=bool)
        for transform in data_config.data_transforms.inputs
        if transform.__class__.__name__ == "DeltaActions" and transform.mask is not None
    ]


def maybe_wrap_retarget_cache_dataset(dataset: Dataset, data_config: _config.DataConfig) -> Dataset:
    if data_config.meta_beta_enabled:
        return BetaStructuredMetaPairDataset(
            dataset,
            data_config,
            expected_action_space=_expected_retarget_cache_action_space(data_config),
            delta_action_masks=_delta_action_masks(data_config),
        )
    if data_config.meta_alpha_enabled:
        return AlphaMetaRetargetDataset(
            dataset,
            data_config,
            expected_action_space=_expected_retarget_cache_action_space(data_config),
            delta_action_masks=_delta_action_masks(data_config),
        )
    if not data_config.meta_retarget_cache_dir or data_config.meta_retarget_cache_prob <= 0.0:
        return dataset
    return RetargetCacheDataset(
        dataset,
        data_config.meta_retarget_cache_dir,
        sample_prob=data_config.meta_retarget_cache_prob,
        seed=data_config.meta_retarget_cache_seed,
        expected_action_space=_expected_retarget_cache_action_space(data_config),
    )


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    dataset = maybe_wrap_retarget_cache_dataset(dataset, data_config)
    return TransformedDataset(
        dataset,
        [
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    if data_config.meta_retarget_cache_dir and data_config.meta_retarget_cache_prob > 0.0:
        logging.warning("meta_retarget_cache_dir is ignored for iterable/RLDS datasets.")

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=data_config.data_action_horizon_override or config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=data_config.data_action_horizon_override or config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
