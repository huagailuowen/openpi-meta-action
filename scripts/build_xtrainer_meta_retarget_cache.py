"""Build an offline x-trainer geometry meta retarget cache.

The cache is chunk-centric. For each LeRobot dataset index, this script first
loads the same action horizon that OpenPI training would see, canonicalizes old
and structured meta formats into ``state/actions/meta_areas``, and then
optionally writes one or more retargeted variants. IK runs in absolute action
space; cache files are saved in the action space expected by the selected
training config.

Training can later sample this cache with:

  data.meta_retarget_cache_dir=<cache-dir>
  data.meta_retarget_cache_prob=<p>

No IK is run during training.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
import json
import pathlib
import shutil
from typing import Any

import numpy as np
import tqdm
import tyro

import openpi.policies.xtrainer_meta_retarget as _retarget


def _apply_transforms(data: dict[str, Any], transforms) -> dict[str, Any]:
    for transform in transforms:
        data = transform(data)
    return data


def _infer_max_meta_areas(data_config: Any) -> int:
    for transform in data_config.data_transforms.inputs:
        if hasattr(transform, "max_meta_areas"):
            return int(transform.max_meta_areas)
    return 1


def _infer_action_stride(data_config: Any) -> int:
    for transform in data_config.data_transforms.inputs:
        if transform.__class__.__name__ == "SubsampleActions":
            return int(transform.stride)
    return 1


def _infer_delta_action_masks(data_config: Any) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for transform in data_config.data_transforms.inputs:
        if transform.__class__.__name__ != "DeltaActions":
            continue
        mask = getattr(transform, "mask", None)
        if mask is not None:
            masks.append(np.asarray(mask, dtype=bool))
    return masks


def _canonicalize_sample(
    raw_sample: dict[str, Any],
    data_config: Any,
    *,
    max_meta_areas: int,
    action_stride: int,
) -> dict[str, Any]:
    repacked = _apply_transforms(raw_sample, data_config.repack_transforms.inputs)
    canonical = _retarget.canonicalize_repacked_xtrainer_chunk(
        repacked,
        max_meta_areas=max_meta_areas,
    )
    if action_stride > 1:
        action_horizon = canonical["actions"].shape[0]
        canonical["actions"] = canonical["actions"][::action_stride]
        meta_targets = canonical.get("meta_action_targets")
        if isinstance(meta_targets, dict):
            canonical["meta_action_targets"] = {
                key: value[::action_stride]
                if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == action_horizon
                else value
                for key, value in meta_targets.items()
            }
    return canonical


def _with_retarget_mode(sample: dict[str, Any], mode: str) -> dict[str, Any]:
    return {**sample, "_retarget_mode": mode}


def _scaled_generator_config(config: _retarget.MetaRetargetGeneratorConfig, scale: float | None):
    if scale is None:
        return config
    scale = float(scale)
    return dataclasses.replace(
        config,
        position_noise_max_m=config.position_noise_max_m * scale,
        direction_noise_max_deg=config.direction_noise_max_deg * scale,
        future_near_position_noise_max_m=config.future_near_position_noise_max_m * scale,
        future_near_direction_noise_max_deg=config.future_near_direction_noise_max_deg * scale,
        correction_min_position_offset_m=config.correction_min_position_offset_m * scale,
        correction_min_direction_offset_deg=config.correction_min_direction_offset_deg * scale,
    )


def _apply_delta_action_masks(
    result: _retarget.MetaRetargetResult,
    delta_action_masks: list[list[bool]],
) -> _retarget.MetaRetargetResult:
    if not delta_action_masks:
        return result

    actions = result.actions.copy()
    state = result.state
    for mask_values in delta_action_masks:
        mask = np.asarray(mask_values, dtype=bool)
        dims = int(mask.shape[-1])
        if actions.shape[-1] < dims or state.shape[-1] < dims:
            raise ValueError(
                f"Cannot apply DeltaActions mask with {dims} dims to "
                f"state/actions shapes {state.shape}/{actions.shape}"
            )
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0.0), axis=-2)

    return dataclasses.replace(result, actions=actions.astype(np.float32))


def _load_existing_pairs(cache_dir: pathlib.Path) -> set[tuple[int, int]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return set()
    pairs: set[tuple[int, int]] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            record = json.loads(stripped_line)
            if bool(record.get("accepted", True)):
                pairs.add((int(record["base_index"]), int(record["variant_id"])))
    return pairs


def _variant_relpath(base_index: int, variant_id: int) -> pathlib.Path:
    shard = base_index // 1000
    return pathlib.Path("variants") / f"{shard:06d}" / f"{base_index:09d}_{variant_id:02d}.npz"


def _generate_variant_worker(
    *,
    base_index: int,
    variant_id: int,
    seed: int,
    sample: dict[str, Any],
    cache_dir: str,
    generator_config: dict[str, Any],
    delta_action_masks: list[list[bool]],
    max_attempts: int,
    retarget_scale: float | None = None,
) -> dict[str, Any]:
    config = _scaled_generator_config(_retarget.MetaRetargetGeneratorConfig(**generator_config), retarget_scale)
    relpath = _variant_relpath(base_index, variant_id)
    base_record = {
        "base_index": int(base_index),
        "variant_id": int(variant_id),
        "path": str(relpath),
    }
    if retarget_scale is not None:
        base_record["retarget_scale"] = float(retarget_scale)

    last_record: dict[str, Any] | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(seed + attempt * 9973)
        rng = np.random.default_rng(attempt_seed)
        result = _retarget.generate_retargeted_chunk(sample, rng=rng, config=config)
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

        result = _apply_delta_action_masks(result, delta_action_masks)
        _retarget.save_retarget_result(pathlib.Path(cache_dir) / relpath, result)
        return {**record, "accepted": True, **diagnostics}

    assert last_record is not None
    return last_record


def _generate_variant_batch(
    *,
    jobs: list[dict[str, Any]],
    cache_dir: str,
    generator_config: dict[str, Any],
    delta_action_masks: list[list[bool]],
    max_attempts: int,
    pad_to_batch_size: int | None,
) -> list[dict[str, Any]]:
    config = _retarget.MetaRetargetGeneratorConfig(**generator_config)
    if any(job.get("retarget_scale") is not None for job in jobs):
        return [
            _generate_variant_worker(
                base_index=int(job["base_index"]),
                variant_id=int(job["variant_id"]),
                seed=int(job["seed"]),
                sample=job["sample"],
                cache_dir=cache_dir,
                generator_config=generator_config,
                delta_action_masks=delta_action_masks,
                max_attempts=max_attempts,
                retarget_scale=job.get("retarget_scale"),
            )
            for job in jobs
        ]
    remaining = list(jobs)
    final_records: dict[tuple[int, int], dict[str, Any]] = {}

    for attempt in range(max(1, int(max_attempts))):
        if not remaining:
            break

        rngs = [np.random.default_rng(int(job["seed"]) + attempt * 9973) for job in remaining]
        samples = [job["sample"] for job in remaining]
        real_count = len(samples)
        if pad_to_batch_size is not None and real_count < pad_to_batch_size:
            pad_count = int(pad_to_batch_size) - real_count
            samples = samples + [samples[-1]] * pad_count
            rngs = rngs + [
                np.random.default_rng(int(remaining[-1]["seed"]) + attempt * 9973 + 104729 * (pad_idx + 1))
                for pad_idx in range(pad_count)
            ]
        results = _retarget.generate_retargeted_chunks_batch(samples, rngs=rngs, config=config)[:real_count]

        next_remaining: list[dict[str, Any]] = []
        for job, result in zip(remaining, results, strict=True):
            base_index = int(job["base_index"])
            variant_id = int(job["variant_id"])
            pair = (base_index, variant_id)
            relpath = _variant_relpath(base_index, variant_id)
            record = {
                "base_index": base_index,
                "variant_id": variant_id,
                "path": str(relpath),
                "seed": int(job["seed"]) + attempt * 9973,
                "attempt": int(attempt),
                "attempts": int(attempt + 1),
            }

            if result is None:
                final_records[pair] = {**record, "accepted": False, "reason": "sample_not_usable"}
                next_remaining.append(job)
                continue

            diagnostics = result.diagnostics.to_json_dict()
            if not result.diagnostics.accepted:
                final_records[pair] = {**record, "accepted": False, **diagnostics}
                next_remaining.append(job)
                continue

            cache_result = _apply_delta_action_masks(result, delta_action_masks)
            _retarget.save_retarget_result(pathlib.Path(cache_dir) / relpath, cache_result)
            final_records[pair] = {**record, "accepted": True, **diagnostics}

        remaining = next_remaining

    return [final_records[(int(job["base_index"]), int(job["variant_id"]))] for job in jobs]


def _write_jsonl_record(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def main(
    config_name: str,
    output_dir: str,
    *,
    retarget_prob: float = 0.5,
    variants_per_selected_chunk: int = 1,
    seed: int = 0,
    num_workers: int = 8,
    batch_size: int = 128,
    max_chunks: int | None = None,
    overwrite: bool = False,
    max_attempts_per_variant: int = 8,
    position_noise_max_m: float = 0.04,
    direction_noise_max_deg: float = 35.0,
    future_near_mode_prob: float = 0.5,
    future_near_window_frames: int = 6,
    future_near_position_noise_max_m: float = 0.008,
    future_near_direction_noise_max_deg: float = 7.0,
    future_near_transition_steps: int = 8,
    correction_forward_exclusion_angle_deg: float = 70.0,
    correction_min_position_offset_m: float = 0.02,
    correction_min_direction_offset_deg: float = 13.0,
    correction_max_sample_attempts: int = 64,
    approach_joint_step_rad: float = 0.04,
    max_approach_steps: int | None = None,
    ik_max_iters: int = 80,
    ik_tolerance: float = 1e-3,
    ik_backend: str = "jax",
    accept_max_position_error_m: float = 0.015,
    accept_max_direction_error_rad: float = 0.25,
    accept_max_step_joint_delta_rad: float = 0.35,
    accept_max_abs_action_value: float = 1e4,
    accept_max_camera_rotvec_norm_rad: float = 3.143,
    sample_retarget_scale: bool = False,
    retarget_scale_min: float = 0.7,
    retarget_scale_max: float = 1.0,
) -> None:
    """Build a reusable retarget cache for one OpenPI training config."""

    if not 0.0 <= retarget_prob <= 1.0:
        raise ValueError(f"retarget_prob must be in [0, 1], got {retarget_prob}")
    if not 0.0 <= future_near_mode_prob <= 1.0:
        raise ValueError(f"future_near_mode_prob must be in [0, 1], got {future_near_mode_prob}")
    if variants_per_selected_chunk < 1:
        raise ValueError("--variants-per-selected-chunk must be >= 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if ik_backend not in ("numpy", "jax"):
        raise ValueError(f"--ik-backend must be 'numpy' or 'jax', got {ik_backend!r}")
    if retarget_scale_min <= 0.0 or retarget_scale_max <= 0.0 or retarget_scale_min > retarget_scale_max:
        raise ValueError(
            f"Invalid retarget scale range: min={retarget_scale_min}, max={retarget_scale_max}"
        )
    if ik_backend == "jax" and num_workers > 1:
        print("Warning: --ik-backend jax uses batched GPU execution in the main process; --num-workers is ignored.")

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
    # Cache generation must not sample from an existing cache while building it.
    data_config = dataclasses.replace(
        data_config,
        repo_id=str(pathlib.Path.cwd().resolve()) if data_config.repo_id == "." else data_config.repo_id,
        meta_retarget_cache_dir=None,
        meta_retarget_cache_prob=0.0,
        meta_alpha_enabled=False,
    )

    action_horizon = data_config.data_action_horizon_override or train_config.model.action_horizon
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, train_config.model)
    max_meta_areas = _infer_max_meta_areas(data_config)
    action_stride = _infer_action_stride(data_config)
    delta_action_masks = _infer_delta_action_masks(data_config)
    delta_action_masks_json = [mask.astype(bool).tolist() for mask in delta_action_masks]

    generator_config = _retarget.MetaRetargetGeneratorConfig(
        position_noise_max_m=position_noise_max_m,
        direction_noise_max_deg=direction_noise_max_deg,
        future_near_mode_prob=future_near_mode_prob,
        future_near_window_frames=future_near_window_frames,
        future_near_position_noise_max_m=future_near_position_noise_max_m,
        future_near_direction_noise_max_deg=future_near_direction_noise_max_deg,
        future_near_transition_steps=future_near_transition_steps,
        correction_forward_exclusion_angle_deg=correction_forward_exclusion_angle_deg,
        correction_min_position_offset_m=correction_min_position_offset_m,
        correction_min_direction_offset_deg=correction_min_direction_offset_deg,
        correction_max_sample_attempts=correction_max_sample_attempts,
        approach_joint_step_rad=approach_joint_step_rad,
        max_approach_steps=max_approach_steps,
        ik_max_iters=ik_max_iters,
        ik_tolerance=ik_tolerance,
        ik_backend=ik_backend,
        accept_max_position_error_m=accept_max_position_error_m,
        accept_max_direction_error_rad=accept_max_direction_error_rad,
        accept_max_step_joint_delta_rad=accept_max_step_joint_delta_rad,
        accept_max_abs_action_value=accept_max_abs_action_value,
        accept_max_camera_rotvec_norm_rad=accept_max_camera_rotvec_norm_rad,
    )

    metadata = {
        "config_name": config_name,
        "repo_id": data_config.repo_id,
        "retarget_prob": retarget_prob,
        "variants_per_selected_chunk": variants_per_selected_chunk,
        "seed": seed,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "max_chunks": max_chunks,
        "max_attempts_per_variant": max_attempts_per_variant,
        "dataset_len": len(dataset),
        "raw_action_horizon": action_horizon,
        "model_action_horizon": train_config.model.action_horizon,
        "action_stride": action_stride,
        "cache_action_space": "delta" if delta_action_masks else "absolute",
        "delta_action_masks": delta_action_masks_json,
        "retarget_mode_sampling": "per_variant_fixed_before_retries",
        "sample_retarget_scale": sample_retarget_scale,
        "retarget_scale_min": retarget_scale_min,
        "retarget_scale_max": retarget_scale_max,
        "max_meta_areas": max_meta_areas,
        "generator_config": generator_config.to_json_dict(),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    existing_pairs = set() if overwrite else _load_existing_pairs(cache_dir)
    selection_rng = np.random.default_rng(seed)
    submitted = 0
    accepted = 0
    rejected = 0
    skipped_existing = 0
    skipped_probability = 0

    pending: dict[futures.Future, tuple[int, int]] = {}
    batch_jobs: list[dict[str, Any]] = []

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

    total = len(dataset) if max_chunks is None else min(max_chunks, len(dataset))
    if ik_backend == "jax":
        batch_future: futures.Future | None = None

        def drain_batch_future() -> None:
            nonlocal batch_future
            if batch_future is None:
                return
            for record in batch_future.result():
                handle_record(record)
            batch_future = None

        def submit_batch_async(executor: futures.Executor) -> None:
            nonlocal batch_future
            if not batch_jobs:
                return
            drain_batch_future()
            jobs = list(batch_jobs)
            batch_jobs.clear()
            batch_future = executor.submit(
                _generate_variant_batch,
                jobs=jobs,
                cache_dir=str(cache_dir),
                generator_config=generator_config.to_json_dict(),
                delta_action_masks=delta_action_masks_json,
                max_attempts=max_attempts_per_variant,
                pad_to_batch_size=batch_size,
            )

        with futures.ThreadPoolExecutor(max_workers=1) as batch_executor:
            for base_index in tqdm.trange(total, desc="Submitting retarget chunks"):
                if selection_rng.random() > retarget_prob:
                    skipped_probability += variants_per_selected_chunk
                    continue

                raw_sample = dataset[base_index]
                canonical = _canonicalize_sample(
                    raw_sample,
                    data_config,
                    max_meta_areas=max_meta_areas,
                    action_stride=action_stride,
                )

                for variant_id in range(variants_per_selected_chunk):
                    pair = (base_index, variant_id)
                    if pair in existing_pairs:
                        skipped_existing += 1
                        continue
                    variant_seed = int(seed + base_index * 10007 + variant_id * 101)
                    retarget_mode = "future_near" if selection_rng.random() < future_near_mode_prob else "correction"
                    retarget_scale = (
                        float(selection_rng.uniform(retarget_scale_min, retarget_scale_max))
                        if sample_retarget_scale
                        else None
                    )
                    batch_jobs.append(
                        {
                            "base_index": base_index,
                            "variant_id": variant_id,
                            "seed": variant_seed,
                            "sample": _with_retarget_mode(canonical, retarget_mode),
                            "retarget_scale": retarget_scale,
                        }
                    )
                    submitted += 1
                    if len(batch_jobs) >= batch_size:
                        submit_batch_async(batch_executor)

            submit_batch_async(batch_executor)
            drain_batch_future()
    else:
        with futures.ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
            for base_index in tqdm.trange(total, desc="Submitting retarget chunks"):
                if selection_rng.random() > retarget_prob:
                    skipped_probability += variants_per_selected_chunk
                    continue

                raw_sample = dataset[base_index]
                canonical = _canonicalize_sample(
                    raw_sample,
                    data_config,
                    max_meta_areas=max_meta_areas,
                    action_stride=action_stride,
                )

                for variant_id in range(variants_per_selected_chunk):
                    pair = (base_index, variant_id)
                    if pair in existing_pairs:
                        skipped_existing += 1
                        continue
                    variant_seed = int(seed + base_index * 10007 + variant_id * 101)
                    retarget_mode = "future_near" if selection_rng.random() < future_near_mode_prob else "correction"
                    retarget_scale = (
                        float(selection_rng.uniform(retarget_scale_min, retarget_scale_max))
                        if sample_retarget_scale
                        else None
                    )
                    pending[
                        executor.submit(
                            _generate_variant_worker,
                            base_index=base_index,
                            variant_id=variant_id,
                            seed=variant_seed,
                            sample=_with_retarget_mode(canonical, retarget_mode),
                            cache_dir=str(cache_dir),
                            generator_config=generator_config.to_json_dict(),
                            delta_action_masks=delta_action_masks_json,
                            max_attempts=max_attempts_per_variant,
                            retarget_scale=retarget_scale,
                        )
                    ] = pair
                    submitted += 1
                    if len(pending) >= max(1, num_workers * 4):
                        drain_one()

            while pending:
                drain_one()

    summary = {
        "submitted": submitted,
        "accepted": accepted,
        "rejected": rejected,
        "skipped_existing": skipped_existing,
        "skipped_probability": skipped_probability,
        "manifest_path": str(manifest_path),
        "failure_path": str(failure_path),
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
