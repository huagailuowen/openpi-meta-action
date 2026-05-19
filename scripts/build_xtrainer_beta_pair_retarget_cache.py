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


def _same_tool_sample_pair(target_sample: dict[str, Any], source_sample: dict[str, Any]) -> bool:
    target_tool = _scalar_int(target_sample.get("tool_instance_hash"), default=-1)
    source_tool = _scalar_int(source_sample.get("tool_instance_hash"), default=-2)
    return target_tool >= 0 and target_tool == source_tool


def _valid_same_tool_pair(target_sample: dict[str, Any], source_sample: dict[str, Any]) -> bool:
    return _is_origin_sample(target_sample) and _is_origin_sample(source_sample) and _same_tool_sample_pair(target_sample, source_sample)


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
) -> dict[str, Any]:
    relpath = _pair_relpath(target_index, source_index, variant_id)
    base_record = {
        "target_index": int(target_index),
        "source_index": int(source_index),
        "variant_id": int(variant_id),
        "path": str(relpath),
        "retarget_mode": retarget_mode,
    }
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
    same_tool_only: bool = True,
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

    action_horizon = data_config.data_action_horizon_override or train_config.model.action_horizon
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, train_config.model)
    max_meta_areas = _chunk_cache._infer_max_meta_areas(data_config)  # noqa: SLF001
    action_stride = _chunk_cache._infer_action_stride(data_config)  # noqa: SLF001
    delta_action_masks = _chunk_cache._infer_delta_action_masks(data_config)  # noqa: SLF001
    delta_action_masks_json = [mask.astype(bool).tolist() for mask in delta_action_masks]

    generator_config = _retarget.MetaRetargetGeneratorConfig(
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
        "delta_action_masks": delta_action_masks_json,
        "max_meta_areas": max_meta_areas,
        "same_tool_only": bool(same_tool_only),
        "generator_config": generator_config.to_json_dict(),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    existing_pairs = set() if overwrite else _load_existing_pairs(cache_dir)
    rng = np.random.default_rng(seed)
    total_targets = len(dataset) if max_targets is None else min(max_targets, len(dataset))
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
        for target_index in tqdm.trange(total_targets, desc="Submitting beta pair retargets"):
            if rng.random() > target_prob:
                skipped_probability += variants_per_target
                continue
            raw_target = dataset[target_index]
            target = _chunk_cache._canonicalize_sample(  # noqa: SLF001
                raw_target,
                data_config,
                max_meta_areas=max_meta_areas,
                action_stride=action_stride,
            )
            for variant_id in range(variants_per_target):
                if max_pairs is not None and submitted >= max_pairs:
                    break
                source_index = None
                source = None
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
