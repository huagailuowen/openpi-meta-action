"""Build an offline beta pair-retarget cache for structured meta training.

The cache stores retargeted target chunks conditioned on a separate source
chunk. Training can later sample these records as a non-blocking fallback when
the online beta pair-retarget producer has not filled its queue fast enough.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
import json
import pathlib
import shutil
import sys
from typing import Any

import numpy as np
import tqdm
import tyro

import openpi.policies.xtrainer_meta_retarget as _retarget

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import build_xtrainer_meta_retarget_cache as _chunk_cache  # noqa: E402


def _pair_relpath(target_index: int, source_index: int, variant_id: int) -> pathlib.Path:
    shard = target_index // 1000
    return (
        pathlib.Path("pairs")
        / f"{shard:06d}"
        / f"{target_index:09d}_{source_index:09d}_{variant_id:02d}.npz"
    )


def _load_existing_pairs(cache_dir: pathlib.Path) -> set[tuple[int, int, int]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return set()
    pairs: set[tuple[int, int, int]] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if bool(record.get("accepted", True)):
                pairs.add((int(record["target_index"]), int(record["source_index"]), int(record["variant_id"])))
    return pairs


def _write_jsonl_record(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _scalar_int(value: Any, *, default: int = -1) -> int:
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return default
    return int(array[0])


def _is_origin_sample(sample: dict[str, Any]) -> bool:
    return _scalar_int(sample.get("source_type_id"), default=0) == 0


def _is_imagine_sample(sample: dict[str, Any]) -> bool:
    return _scalar_int(sample.get("source_type_id"), default=0) != 0


def _same_tool_sample_pair(target_sample: dict[str, Any], source_sample: dict[str, Any]) -> bool:
    target_tool = _scalar_int(target_sample.get("tool_instance_hash"), default=-1)
    source_tool = _scalar_int(source_sample.get("tool_instance_hash"), default=-2)
    return target_tool >= 0 and target_tool == source_tool


def _valid_same_tool_pair(target_sample: dict[str, Any], source_sample: dict[str, Any]) -> bool:
    return _is_origin_sample(target_sample) and _is_imagine_sample(source_sample) and _same_tool_sample_pair(target_sample, source_sample)


def _with_beta_metadata(sample: dict[str, Any], *, tool_instance_hash: int, source_type_id: int) -> dict[str, Any]:
    out = dict(sample)
    out["tool_instance_hash"] = np.asarray(tool_instance_hash, dtype=np.int32)
    out["source_type_id"] = np.asarray(source_type_id, dtype=np.int32)
    return out


def _load_chunk_retarget_records_by_tool(
    cache_dir: pathlib.Path,
    *,
    sampling_metadata: Any,
) -> dict[int, list[dict[str, Any]]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Chunk retarget cache manifest not found: {manifest_path}")
    records_by_tool: dict[int, list[dict[str, Any]]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not bool(record.get("accepted", True)):
                continue
            try:
                base_index = int(record["base_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= base_index < len(sampling_metadata.tool_instance_hash)):
                continue
            tool = int(sampling_metadata.tool_instance_hash[base_index])
            if tool < 0:
                continue
            records_by_tool.setdefault(tool, []).append(record)
    return records_by_tool


def _load_npz_payload(cache_dir: pathlib.Path, record: dict[str, Any]) -> dict[str, np.ndarray] | None:
    try:
        with np.load(cache_dir / str(record["path"])) as cached:
            return {key: cached[key].copy() for key in cached.files}
    except FileNotFoundError:
        return None


def _apply_retargeted_payload_to_canonical(
    sample: dict[str, Any],
    payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    if "meta_area_pose12d" not in payload:
        raise ValueError("Beta same-tool pair source reconstruction currently requires 12D chunk retarget payloads.")
    out = dict(sample)
    out["state"] = np.asarray(payload["state"], dtype=np.float32)
    out["actions"] = np.asarray(payload["actions"], dtype=np.float32)
    source_type_shape = np.asarray(sample.get("source_type_id", np.asarray(0, dtype=np.int32))).shape
    out["source_type_id"] = np.full(source_type_shape, 2, dtype=np.int32)

    meta_areas = dict(out.get("meta_areas", {}))
    meta_areas["pose12d"] = np.asarray(payload["meta_area_pose12d"], dtype=np.float32)
    meta_areas.pop("pose6d", None)
    if "meta_area_dim_mask12" in payload:
        meta_areas["dim_mask12"] = np.asarray(payload["meta_area_dim_mask12"], dtype=bool)
    meta_areas["type"] = np.asarray(payload["meta_area_type"], dtype=np.int32)
    meta_areas["mask"] = np.asarray(payload["meta_area_mask"], dtype=bool)
    out["meta_areas"] = meta_areas

    if "meta_action_target_pose12d" in payload:
        meta_targets = dict(out.get("meta_action_targets", {}))
        meta_targets["pose12d"] = np.asarray(payload["meta_action_target_pose12d"], dtype=np.float32)
        meta_targets.pop("pose6d", None)
        if "meta_action_target_dim_mask12" in payload:
            meta_targets["dim_mask12"] = np.asarray(payload["meta_action_target_dim_mask12"], dtype=bool)
        if "meta_action_target_mask" in payload:
            meta_targets["mask"] = np.asarray(payload["meta_action_target_mask"], dtype=bool)
        else:
            meta_targets["mask"] = np.ones(meta_targets["pose12d"].shape[:2], dtype=bool)
        out["meta_action_targets"] = meta_targets
    return out


def _generate_pair_worker(
    *,
    target_index: int,
    source_index: int,
    variant_id: int,
    seed: int,
    target_sample: dict[str, Any],
    source_sample: dict[str, Any],
    cache_dir: str,
    generator_config: dict[str, Any],
    delta_action_masks: list[list[bool]],
    max_attempts: int,
    retarget_mode: str,
    source_retarget_path: str | None = None,
) -> dict[str, Any]:
    relpath = _pair_relpath(target_index, source_index, variant_id)
    base_record = {
        "target_index": int(target_index),
        "source_index": int(source_index),
        "variant_id": int(variant_id),
        "path": str(relpath),
        "retarget_mode": retarget_mode,
    }
    if source_retarget_path is not None:
        base_record["source_retarget_path"] = str(source_retarget_path)
        base_record["source_retarget_base_index"] = int(source_index)
    target_sample = _chunk_cache._with_retarget_mode(target_sample, retarget_mode)  # noqa: SLF001
    config = _retarget.MetaRetargetGeneratorConfig(**generator_config)

    last_record: dict[str, Any] | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(seed + attempt * 9973)
        result = _retarget.generate_pair_retargeted_chunk(
            target_sample,
            source_sample,
            rng=np.random.default_rng(attempt_seed),
            config=config,
        )
        record = {
            **base_record,
            "seed": attempt_seed,
            "attempt": int(attempt),
            "attempts": int(attempt + 1),
        }
        if result is None:
            last_record = {**record, "accepted": False, "reason": "sample_not_usable"}
            continue

        diagnostics = result.diagnostics.to_json_dict()
        if not result.diagnostics.accepted:
            last_record = {**record, "accepted": False, **diagnostics}
            continue

        cache_result = _chunk_cache._apply_delta_action_masks(result, delta_action_masks)  # noqa: SLF001
        _retarget.save_retarget_result(pathlib.Path(cache_dir) / relpath, cache_result)
        return {**record, "accepted": True, **diagnostics}

    assert last_record is not None
    return last_record


def main(
    config_name: str,
    output_dir: str,
    *,
    target_prob: float = 1.0,
    variants_per_target: int = 1,
    seed: int = 0,
    num_workers: int = 8,
    max_targets: int | None = None,
    max_pairs: int | None = None,
    overwrite: bool = False,
    max_attempts_per_pair: int = 4,
    future_near_mode_prob: float = 0.5,
    future_near_window_frames: int = 6,
    future_near_transition_steps: int = 8,
    correction_forward_exclusion_angle_deg: float = 70.0,
    correction_min_position_offset_m: float = 0.02,
    correction_min_direction_offset_deg: float = 13.0,
    correction_max_sample_attempts: int = 64,
    approach_joint_step_rad: float = 0.03,
    max_approach_steps: int | None = None,
    ik_max_iters: int = 80,
    ik_tolerance: float = 1e-3,
    accept_max_position_error_m: float = 0.015,
    accept_max_direction_error_rad: float = 0.4,
    accept_max_step_joint_delta_rad: float = 0.35,
    accept_max_abs_action_value: float = 1e4,
    accept_max_camera_rotvec_norm_rad: float = 3.143,
    retarget_algorithm: str | None = None,
    same_tool_only: bool = True,
    source_retarget_cache_dir: str | None = None,
) -> None:
    """Build a reusable pair-retarget cache for one beta OpenPI training config."""

    if not 0.0 <= target_prob <= 1.0:
        raise ValueError(f"target_prob must be in [0, 1], got {target_prob}")
    if variants_per_target < 1:
        raise ValueError("--variants-per-target must be >= 1")
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    cache_dir = pathlib.Path(output_dir).expanduser().resolve()
    if overwrite and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.jsonl"
    failure_path = cache_dir / "failures.jsonl"

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    configured_retarget_cache_dir = data_config.meta_retarget_cache_dir
    data_config = dataclasses.replace(
        data_config,
        repo_id=str(pathlib.Path.cwd().resolve()) if data_config.repo_id == "." else data_config.repo_id,
        meta_retarget_cache_dir=None,
        meta_retarget_cache_prob=0.0,
        meta_alpha_enabled=False,
        meta_beta_enabled=False,
        meta_beta_pair_cache_dir=None,
        meta_beta_online_async_enabled=False,
    )
    effective_retarget_algorithm = retarget_algorithm or data_config.meta_retarget_algorithm
    if effective_retarget_algorithm not in _retarget.SUPPORTED_RETARGET_ALGORITHMS:
        raise ValueError(
            f"--retarget-algorithm must be one of {sorted(_retarget.SUPPORTED_RETARGET_ALGORITHMS)}, "
            f"got {effective_retarget_algorithm!r}"
        )

    action_horizon = data_config.data_action_horizon_override or train_config.model.action_horizon
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, train_config.model)
    sampling_metadata = _data_loader._build_beta_sampling_metadata(dataset)  # noqa: SLF001
    max_meta_areas = _chunk_cache._infer_max_meta_areas(data_config)  # noqa: SLF001
    action_stride = _chunk_cache._infer_action_stride(data_config)  # noqa: SLF001
    delta_action_masks = _chunk_cache._infer_delta_action_masks(data_config)  # noqa: SLF001
    delta_action_masks_json = [mask.astype(bool).tolist() for mask in delta_action_masks]
    chunk_retarget_cache_dir: pathlib.Path | None = None
    chunk_records_by_tool: dict[int, list[dict[str, Any]]] = {}
    if same_tool_only:
        if sampling_metadata is None:
            raise ValueError("Same-tool beta pair cache build requires lightweight sampling metadata.")
        chunk_retarget_cache_candidates = [
            source_retarget_cache_dir,
            configured_retarget_cache_dir,
            str(cache_dir.parent / "retarget_cache"),
        ]
        for candidate in chunk_retarget_cache_candidates:
            if not candidate:
                continue
            candidate_path = pathlib.Path(candidate).expanduser().resolve()
            if (candidate_path / "manifest.jsonl").exists():
                chunk_retarget_cache_dir = candidate_path
                break
        if chunk_retarget_cache_dir is None:
            raise FileNotFoundError(
                "Same-tool beta pair cache build requires a chunk-level retarget cache. "
                "Pass --source-retarget-cache-dir or set data.meta_retarget_cache_dir."
            )
        chunk_records_by_tool = _load_chunk_retarget_records_by_tool(
            chunk_retarget_cache_dir,
            sampling_metadata=sampling_metadata,
        )
        if not chunk_records_by_tool:
            raise ValueError(f"No accepted same-tool chunk retarget records found in {chunk_retarget_cache_dir}")
        _data_loader.RetargetCacheDataset._warn_if_metadata_mismatch(  # noqa: SLF001
            chunk_retarget_cache_dir,
            expected_action_space="delta" if delta_action_masks else "absolute",
            expected_retarget_algorithm=effective_retarget_algorithm,
        )

    generator_config = _retarget.MetaRetargetGeneratorConfig(
        retarget_algorithm=effective_retarget_algorithm,
        future_near_mode_prob=future_near_mode_prob,
        future_near_window_frames=future_near_window_frames,
        future_near_transition_steps=future_near_transition_steps,
        correction_forward_exclusion_angle_deg=correction_forward_exclusion_angle_deg,
        correction_min_position_offset_m=correction_min_position_offset_m,
        correction_min_direction_offset_deg=correction_min_direction_offset_deg,
        correction_max_sample_attempts=correction_max_sample_attempts,
        approach_joint_step_rad=approach_joint_step_rad,
        max_approach_steps=max_approach_steps,
        ik_max_iters=ik_max_iters,
        ik_tolerance=ik_tolerance,
        ik_backend="numpy",
        accept_max_position_error_m=accept_max_position_error_m,
        accept_max_direction_error_rad=accept_max_direction_error_rad,
        accept_max_step_joint_delta_rad=accept_max_step_joint_delta_rad,
        accept_max_abs_action_value=accept_max_abs_action_value,
        accept_max_camera_rotvec_norm_rad=accept_max_camera_rotvec_norm_rad,
    )

    metadata = {
        "config_name": config_name,
        "repo_id": data_config.repo_id,
        "target_prob": target_prob,
        "variants_per_target": variants_per_target,
        "seed": seed,
        "num_workers": num_workers,
        "max_targets": max_targets,
        "max_pairs": max_pairs,
        "max_attempts_per_pair": max_attempts_per_pair,
        "dataset_len": len(dataset),
        "raw_action_horizon": action_horizon,
        "model_action_horizon": train_config.model.action_horizon,
        "action_stride": action_stride,
        "cache_action_space": "delta" if delta_action_masks else "absolute",
        "retarget_algorithm": effective_retarget_algorithm,
        "delta_action_masks": delta_action_masks_json,
        "max_meta_areas": max_meta_areas,
        "same_tool_only": bool(same_tool_only),
        "metadata_source_sampling": sampling_metadata is not None,
        "target_traversal": "seeded_shuffle",
        "source_retarget_cache_dir": str(chunk_retarget_cache_dir) if chunk_retarget_cache_dir is not None else None,
        "source_retarget_cache_records": int(sum(len(records) for records in chunk_records_by_tool.values())),
        "source_retarget_cache_tools": int(len(chunk_records_by_tool)),
        "generator_config": generator_config.to_json_dict(),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    existing_pairs = set() if overwrite else _load_existing_pairs(cache_dir)
    rng = np.random.default_rng(seed)
    total_targets = len(dataset) if max_targets is None else min(max_targets, len(dataset))
    target_indices = np.arange(total_targets, dtype=np.int64)
    rng.shuffle(target_indices)
    submitted = 0
    accepted = 0
    rejected = 0
    skipped_existing = 0
    skipped_probability = 0
    skipped_no_same_tool_source = 0

    pending: dict[futures.Future, tuple[int, int, int]] = {}

    def handle_record(record: dict[str, Any]) -> None:
        nonlocal accepted, rejected
        if bool(record.get("accepted", False)):
            accepted += 1
            _write_jsonl_record(manifest_path, record)
        else:
            rejected += 1
            _write_jsonl_record(failure_path, record)

    def drain_one() -> None:
        done = next(futures.as_completed(pending))
        pending.pop(done)
        handle_record(done.result())

    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for target_index_raw in tqdm.tqdm(target_indices, desc="Submitting beta pair retargets"):
            target_index = int(target_index_raw)
            if rng.random() > target_prob:
                skipped_probability += variants_per_target
                continue
            target_tool = -1
            if same_tool_only:
                assert sampling_metadata is not None
                if sampling_metadata.source_type_id[target_index] != 0:
                    skipped_no_same_tool_source += variants_per_target
                    continue
                target_tool = int(sampling_metadata.tool_instance_hash[target_index])
                if target_tool < 0 or target_tool not in chunk_records_by_tool:
                    skipped_no_same_tool_source += variants_per_target
                    continue
            raw_target = dataset[target_index]
            target = _chunk_cache._canonicalize_sample(  # noqa: SLF001
                raw_target,
                data_config,
                max_meta_areas=max_meta_areas,
                action_stride=action_stride,
            )
            if same_tool_only:
                target = _with_beta_metadata(target, tool_instance_hash=target_tool, source_type_id=0)
            for variant_id in range(variants_per_target):
                if max_pairs is not None and submitted >= max_pairs:
                    break
                source_index = None
                source = None
                source_retarget_path = None
                if same_tool_only:
                    assert chunk_retarget_cache_dir is not None
                    records = chunk_records_by_tool.get(target_tool, [])
                    if records:
                        source_record = records[int(rng.integers(len(records)))]
                        source_index = int(source_record["base_index"])
                        if 0 <= source_index < len(dataset):
                            payload = _load_npz_payload(chunk_retarget_cache_dir, source_record)
                            if payload is not None:
                                source_tool = int(sampling_metadata.tool_instance_hash[source_index])
                                raw_source = dataset[source_index]
                                source_base = _chunk_cache._canonicalize_sample(  # noqa: SLF001
                                    raw_source,
                                    data_config,
                                    max_meta_areas=max_meta_areas,
                                    action_stride=action_stride,
                                )
                                source_base = _with_beta_metadata(
                                    source_base,
                                    tool_instance_hash=source_tool,
                                    source_type_id=0,
                                )
                                source = _apply_retargeted_payload_to_canonical(source_base, payload)
                                source_retarget_path = str(source_record["path"])
                elif sampling_metadata is not None:
                    source_index = _data_loader._sample_beta_source_index_from_metadata(  # noqa: SLF001
                        sampling_metadata,
                        rng,
                        len(dataset),
                        target_index,
                        3,
                        same_tool_only=same_tool_only,
                    )
                    if source_index is not None:
                        raw_source = dataset[source_index]
                        source = _chunk_cache._canonicalize_sample(  # noqa: SLF001
                            raw_source,
                            data_config,
                            max_meta_areas=max_meta_areas,
                            action_stride=action_stride,
                        )
                else:
                    for _ in range(64):
                        if len(dataset) <= 1:
                            break
                        candidate_index = int(rng.integers(len(dataset) - 1))
                        if candidate_index >= target_index:
                            candidate_index += 1
                        raw_source = dataset[candidate_index]
                        candidate_source = _chunk_cache._canonicalize_sample(  # noqa: SLF001
                            raw_source,
                            data_config,
                            max_meta_areas=max_meta_areas,
                            action_stride=action_stride,
                        )
                        if not same_tool_only or _valid_same_tool_pair(target, candidate_source):
                            source_index = candidate_index
                            source = candidate_source
                            break
                if source_index is None or source is None:
                    skipped_no_same_tool_source += 1
                    continue
                if same_tool_only and not _valid_same_tool_pair(target, source):
                    skipped_no_same_tool_source += 1
                    continue
                pair = (target_index, source_index, variant_id)
                if pair in existing_pairs:
                    skipped_existing += 1
                    continue
                retarget_mode = "future_near" if rng.random() < future_near_mode_prob else "correction"
                pending[
                    executor.submit(
                        _generate_pair_worker,
                        target_index=target_index,
                        source_index=source_index,
                        variant_id=variant_id,
                        seed=int(seed + target_index * 10007 + source_index * 103 + variant_id * 1009),
                        target_sample=target,
                        source_sample=source,
                        cache_dir=str(cache_dir),
                        generator_config=generator_config.to_json_dict(),
                        delta_action_masks=delta_action_masks_json,
                        max_attempts=max_attempts_per_pair,
                        retarget_mode=retarget_mode,
                        source_retarget_path=source_retarget_path,
                    )
                ] = pair
                submitted += 1
                if len(pending) >= max(1, num_workers * 4):
                    drain_one()
            if max_pairs is not None and submitted >= max_pairs:
                break

        while pending:
            drain_one()

    summary = {
        "submitted": submitted,
        "accepted": accepted,
        "rejected": rejected,
        "skipped_existing": skipped_existing,
        "skipped_probability": skipped_probability,
        "skipped_no_same_tool_source": skipped_no_same_tool_source,
        "manifest_path": str(manifest_path),
        "failure_path": str(failure_path),
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
