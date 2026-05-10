# openpi — Geometry Meta-Action Fork

This repository is a fork of [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi).
The original documentation is preserved in [README_official.md](README_official.md).

This fork adds a **geometry meta-action** auxiliary head to the π₀.₅ model that jointly predicts
structured per-area meta targets alongside the standard flow-matching robot actions. It supports
the legacy 6D pose interface and the newer 12D structured affordance interface.

---

## Overview of Changes

| File | Change |
|---|---|
| `src/openpi/models/pi0_meta.py` | New `Pi0Meta` model with `MetaActionHead` |
| `src/openpi/models/pi0_config.py` | New config flags to enable the meta model |
| `src/openpi/models/model.py` | Extended `Observation` with `meta_area_poses/types/masks` and optional 12D dim masks |
| `src/openpi/policies/xtrainer_meta_policy.py` | Data transforms for the meta pipeline |
| `src/openpi/training/config.py` | New legacy and structured meta data configs, including 12D structured config |
| `src/openpi/policies/xtrainer_meta_retarget.py` | Offline geometry retarget augmentation for 6D and 12D meta-aware x-trainer chunks |
| `scripts/build_xtrainer_meta_retarget_cache.py` | GPU-batched retarget-cache builder |

---

## Model Architecture (`Pi0Meta`)

`Pi0Meta` extends the π₀.₅ dual-expert transformer with an auxiliary cross-attention head that
predicts geometry meta-actions. It only supports `pi05=True` models.

### Token Sequence Layout

The full token sequence fed into the dual-stream Gemma backbone is:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              PREFIX STREAM  (PaliGemma width)                        │
│  [image tokens × N_cams] | [text prompt tokens] | [meta context tokens × M] |        │
│  [meta special tokens × S]                                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              SUFFIX STREAM  (action-expert width)                    │
│  [noisy action tokens × action_horizon]                                              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Segment details:**

| Segment | Width | Length | Content |
|---|---|---|---|
| Image tokens | PaliGemma | `256 × N_cams` | SigLIP-encoded frames, one block per camera |
| Text prompt tokens | PaliGemma | `max_token_len` | Tokenized task instruction |
| Meta context tokens | PaliGemma | `max_meta_areas` (M) | Per-area pose+type embeddings (see below) |
| Meta special tokens | PaliGemma | `num_meta_special_tokens` (S=4) | Learned task-level context slots |
| Noisy action tokens | Action-expert | `action_horizon` | Flow-matching input `x_t` projected to action-expert width |

All prefix tokens are non-autoregressive (`ar_mask=False`). The first action token is autoregressive
(`ar_mask=True`) and each subsequent action token attends to all preceding ones.

### Meta Context Token Construction

Each of the `M` meta-area slots produces one context token. Given the observation fields
`meta_area_poses ∈ [B, M, 6]`, `meta_area_types ∈ [B, M]`, `meta_area_masks ∈ [B, M]`:

```
slot_token[i]     = meta_slot_embedding(i)               # learned per-slot bias
default_token[i]  = meta_default_embedding(0) + slot_token[i]

provided_token[i] = meta_pose_out(swish(meta_pose_in(pose6d[i])))
                  + meta_type_embedding(type[i])
                  + slot_token[i]

meta_context_token[i] = LayerNorm(
    where(mask[i], provided_token[i], default_token[i])
)
```

When `mask[i]=False` (area not observed), the slot falls back to the default learnable token.

---

## Structured 12D Meta Interface

The new structured interface uses:

```text
pose12d = [position3, shape_matrix6, approach3]
shape_matrix6 = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
```

`shape_matrix6` is the independent entries of a symmetric direction-distribution matrix. It can
represent point/line/surface-like geometry without relying on a discrete type at model input time.
The `type` field is still stored for bookkeeping and retarget logic. `approach3` is an oriented task
direction; labels with no approach set `approach3=[0,0,0]` and mark dims 9:12 inactive in
`dim_mask12`.

LeRobot fields for the 12D path:

```text
observation.meta_areas.pose12d      [M, 12]
observation.meta_areas.dim_mask12   [M, 12]
observation.meta_areas.type         [M]
observation.meta_areas.mask         [M]
action.meta_targets.pose12d         [H, M, 12]
action.meta_targets.dim_mask12      [H, M, 12]
action.meta_targets.mask            [H, M]
```

Normalization rule:

- `observation.meta_areas.pose12d` is not normalized as a separate input field.
- `action.meta_targets.pose12d` stays as an explicit raw target field; it is not copied into `actions`.
- `compute_norm_stats.py` still only computes stats for `state` and `actions`.
- Runtime `meta_actions[:,0,:]` is therefore a raw 12D prediction and must not be unnormalized with action stats.

12D action layout:

```text
dims  0–13   robot joint/gripper action (14D)
dims  14–19  unused legacy meta slot    (zeroed, no supervision)
dims  20–25  camera pose/action channel (6D)
dims  26–31  reserved / padding
```

The 12D config is `pi05_xtrainer_meta_aux_structured_12d_delta`. It sets
`meta_area_pose_dim=12`, `meta_action_dim=12`, and `action_dim=32`.

Retarget cache supports the same structured fields. For 12D line/surface tools, if `approach3` is
active, retargeting aligns the shape matrix and approach direction together instead of only
interpolating a single direction vector along a great circle. The first 12D implementation uses the
numpy IK path for semantic correctness; requesting `--ik-backend jax` on 12D data falls back to numpy.

---

## Training Data Flow

```
Dataset sample
    │
    ▼
XTrainerMetaInputs.__call__
    ├── Parse cam_high / cam_left_wrist / cam_right_wrist images → uint8 HWC
    ├── Parse state vector (raw_state, shape [D])
    ├── If meta_areas present in sample:
    │       copy pose6d / type / mask → meta_area_{poses,types,masks} [M,6], [M], [M]
    │   Else if derive_meta_from_state_if_missing and D >= 20:
    │       meta_area_poses[0] = raw_state[14:20]   # 6D pose from state slice
    │       meta_area_masks[0] = True
    └── Zero-out state[14:20] so the backbone never sees the raw pose directly
    │
    ▼
ModelTransformFactory (tokenizer + image norm)
    │
    ▼
Observation struct
    images:              {cam: [B, H, W, 3] float32 in [-1,1]}
    state:               [B, D]   (meta slice zeroed)
    tokenized_prompt:    [B, L]
    meta_area_poses:     [B, M, 6]
    meta_area_types:     [B, M]
    meta_area_masks:     [B, M]
    │
    ▼
Pi0Meta.compute_loss(rng, observation, actions)
```

Inside `compute_loss`:

1. **Backbone action masking** — only dimensions `[0, meta_action_start_dim)` and
   `[meta_action_start_dim + meta_action_dim, meta_action_start_dim + meta_action_dim + 6)` are
   passed to the backbone. The meta-action slice `[meta_action_start_dim : +meta_action_dim]`
   (default dims 14–19, 6D pose) is masked out of the noisy input so the backbone cannot cheat.

2. **Meta dropout** — at training time, each meta mask bit is independently zeroed with probability
   `meta_dropout_prob` (default 0.25) to make the model robust to missing area annotations.

3. **Flow-matching noise** — a noisy sample `x_t = t·noise + (1-t)·actions` is constructed with
   `t ~ Beta(1.5, 1) × 0.999 + 0.001`.

4. **Forward pass** — prefix and suffix token sequences are concatenated and passed through the
   dual-stream Gemma backbone.

5. **Action output** — the last `action_horizon` suffix tokens are projected back to action space:
   `v_t = action_out_proj(suffix_out[:, -action_horizon:])`.

6. **Meta-action decoding** — the `MetaActionHead` takes the last M+S prefix output tokens and
   the last `action_horizon` suffix output tokens to produce `meta_pred ∈ [B, action_horizon, M, 6]`.

---

## Loss Calculation

The total per-step loss is:

```
L = L_base + meta_loss_weight × L_meta
```

### Base flow-matching loss

```
u_t = noise - actions                         # target velocity field
v_t = model prediction of u_t

backbone_mask = [1 if i < meta_action_start_dim
                 or meta_action_start_dim+meta_action_dim <= i < +6
                 else 0  for i in range(action_dim)]

squared_error = (v_t - u_t)² * backbone_mask
L_base = sum(squared_error, dim=-1) / sum(backbone_mask)   # shape [B, action_horizon]
```

### Meta-action auxiliary loss

```
meta_target[b, t, 0, :] = actions[b, t, meta_action_start_dim : +meta_action_dim]
                         # the ground-truth 6D pose, only at slot 0

meta_loss_mask[b, i] = meta_supervision_masks[b, i]  AND  (i == 0)
                       # only supervise slot 0 when it has a valid annotation

meta_loss = mean((meta_pred - meta_target)², dim=-1)    # [B, ah, M]
L_meta = sum(meta_loss * meta_loss_mask[:, None, :], dim=-1)
         / max(sum(meta_loss_mask, dim=-1, keepdim=True), 1)
                                                          # [B, action_horizon]
```

`meta_loss_weight` defaults to `1.0` and is set in `Pi0Config`.

---

## Meta-Action Head (`MetaActionHead`)

The head produces a per-timestep, per-slot prediction of the 6D pose delta:

```
action_features  = LayerNorm(action_tokens)            # [B, ah, W]
meta_features    = LayerNorm(meta_tokens)              # [B, M, W]
special_features = LayerNorm(special_tokens)           # [B, S, W]

queries[b, t, m, :] = action_query_proj(action_features[b,t])
                     + meta_query_proj(meta_features[b,m])   # [B, ah, M, W]

memory[b, t, m, :, :] = concat(meta_features[b,m], special_features[b,:])   # [B, ah, M, 1+S, W]

attended = CrossAttention(queries, memory)               # [B, ah, M, W]

fused = concat(action_tokens, meta_tokens_expanded, attended)  # [B, ah, M, 3W]
fused = swish(out_in(fused))                             # [B, ah, M, W]
output = out_out(fused)                                  # [B, ah, M, meta_action_dim=6]
```

At inference time the meta-action at slot 0 is the predicted 6D pose of the target area for that
action horizon step.

---

## Inference Data Flow

```
XTrainerMetaInputs (same as training, without actions)
    │
    ▼
Policy.infer → Pi0Meta.sample_actions_with_aux(rng, observation)
    │
    ├── embed_prefix(observation)  →  run once, cache KV
    │
    └── Flow-matching denoising loop (num_steps=10):
            x_1 = Gaussian noise (backbone-masked)
            for t from 1.0 → 0.0:
                embed_suffix(observation, x_t, t)
                suffix_out = backbone(suffix | cached prefix KV)
                v_t = action_out_proj(suffix_out[-action_horizon:])
                x_{t+dt} = x_t + dt * v_t
    │
    ├── Final meta-action decode using cached prefix_out + last suffix_out
    │       meta_actions ∈ [B, action_horizon, M, 6]
    │
    └── Returns {"actions": x_0 [B, ah, action_dim],
                 "meta_actions": [B, ah, M, 6]}

XTrainerMetaOutputs
    ├── actions:      [B, ah, action_dim]
    └── meta_actions: [B, ah, M, 6]  (if present)
```

---

## Training Configs

All configs are defined in `src/openpi/training/config.py`. The meta-aware configs are:

| Config name | Model | action_dim | Delta actions | Meta head | Notes |
|---|---|---|---|---|---|
| `pi0_xtrainer_meta` | π₀ | 32 | No | No | Baseline, no pi05 |
| `pi05_xtrainer_meta` | π₀.₅ | 32 | No | No | Standard 32D finetuning |
| `pi05_xtrainer_meta_delta` | π₀.₅ | 32 | Yes | No | Delta joint actions |
| `pi05_xtrainer_meta_aux` | π₀.₅ | 32 | No | Yes | Full meta-action model |
| `pi05_xtrainer_meta_aux_delta` | π₀.₅ | 32 | Yes | Yes | Meta + delta actions |
| `pi05_xtrainer_meta_aux_structured_delta` | π₀.₅ | 32 | Yes | Yes | Explicit structured 6D fields |
| `pi05_xtrainer_meta_aux_structured_12d_delta` | π₀.₅ | 32 | Yes | Yes | Explicit structured 12D fields |
| `pi05_xtrainer_meta_aux_low_mem_finetune` | π₀.₅ LoRA | 32 | No | Yes | Low-memory LoRA, batch 8 |

The `*_aux*` configs use `Pi0Meta` (`meta_model=True`) with `meta_dropout_prob=0.25` and
`max_meta_areas=1`. All pi05 configs load from the public `gs://openpi-assets/checkpoints/pi05_base`
checkpoint.

### Action dimension layout (32D)

```
dims  0–5    left arm joint positions  (6D)
dims  6      left gripper              (1D)
dims  7–12   right arm joint positions (6D)
dims  13     right gripper             (1D)
dims  14–19  meta geometry pose        (6D)  ← meta_action_start_dim=14, meta_action_dim=6
dims  20–25  camera/extra DoF          (6D)  ← included in backbone mask
dims  26–31  (reserved / padding)
```

For `pi05_xtrainer_meta_aux_structured_12d_delta`, dims 14–19 are unused and masked out of the
backbone loss, dims 20–25 remain the camera channel, and the 12D target is supervised only through
`action.meta_targets.pose12d`.

---

## How to Run Training

Training follows the same flow as the upstream openpi full fine-tuning. Requirements: an NVIDIA GPU
with ≥70 GB VRAM (A100/H100) for full fine-tuning, or ≥22.5 GB for LoRA.

### 1. Install dependencies

```bash
pip install uv
uv sync
```

### 2. Prepare your dataset

Your dataset must be in [LeRobot](https://github.com/huggingface/lerobot) format. The expected
observation keys are:

```
observation.images.top          # top/overhead camera
observation.images.left_wrist   # left wrist camera
observation.images.right_wrist  # right wrist camera
observation.state               # robot state vector, shape [D >= 20]
action                          # action vector, shape [32]
```

If your dataset includes explicit geometry annotations, add a `meta_areas` field with:
```
meta_areas.pose6d   [M, 6]   # 6D pose of each functional area
meta_areas.type     [M]      # integer type label per area
meta_areas.mask     [M]      # bool: True if slot is populated
```

If `meta_areas` is absent, `XTrainerMetaInputs` will automatically derive the first meta slot from
`state[14:20]`.

### 3. Build the optional meta-retarget cache

The meta-aware x-trainer data configs can train from a precomputed geometry retarget cache. The
cache generator retargets the input meta area in raw, unnormalized coordinates, treats the retargeted
meta area as rigidly attached to the right wrist J6, and solves right-arm IK so the retargeted meta
area follows a mode-specific meta-action trajectory. Line and surface areas constrain only their
direction vector, with no roll constraint; point areas constrain position only.

Retarget variants are sampled from two modes. `future_near` places the new meta area near a randomly
selected frame from the local future trajectory window, then follows the subsequent meta trajectory
directly. It uses a shifted virtual trajectory that linearly decays back to the original trajectory
over 8 steps by default, avoiding the earlier behavior where an already-ahead retarget first learned to move
backward to frame 0. `correction` keeps the original qpos-interpolated approach stage, but samples
larger off-trajectory perturbations only: the position offset is forbidden inside the forward
70-degree cone estimated from the first local meta-action frames, and the retarget must differ by
either more than 2 cm in position or more than 13 degrees in direction while staying within 4 cm and
35 degrees.
The mode is sampled once per cache variant according to `--future-near-mode-prob` and stays fixed
across retry attempts.

The cache builder follows the data interface selected by `--config-name`. Legacy meta configs such
as `pi05_xtrainer_meta_aux` and `pi05_xtrainer_meta_aux_delta` repack only `state` and `actions`;
the builder then derives the first meta area from the legacy state slice. Structured configs such as
`pi05_xtrainer_meta_aux_structured_delta` repack `observation.meta_areas.*` and
`action.meta_targets.*`, so the builder uses those structured fields directly. In both cases, IK and
retargeting are computed in absolute, unnormalized coordinates after meta extraction.

Cache action space also follows the training config. Non-delta configs write absolute retargeted
actions. Delta configs still solve IK in absolute space, but the cache writer applies the same
`DeltaActions` mask before saving each accepted variant; this keeps cached variants compatible with
the dataloader, which wraps the cache after `data_transforms` and before normalization. Rebuild old
delta caches generated before this behavior, because those files may contain absolute actions in a
delta training pipeline. Training will warn when a delta config points to a cache that lacks the new
`cache_action_space` metadata or records a mismatched action space.

By default, the cache builder uses GPU JAX IK, creates one random variant per selected chunk
(`V=1`), and selects chunks with probability `0.5`:

```bash
python scripts/build_xtrainer_meta_retarget_cache.py \
    --config-name pi05_xtrainer_meta_aux \
    --output-dir /path/to/retarget_cache \
    --overwrite
```

For 12D structured data:

```bash
python scripts/build_xtrainer_meta_retarget_cache.py \
    --config-name pi05_xtrainer_meta_aux_structured_12d_delta \
    --output-dir /path/to/retarget_cache_12d \
    --overwrite
```

Useful flags:

| Flag | Default | Meaning |
|---|---:|---|
| `--retarget-prob` | `0.5` | Probability that a dataset chunk is selected for cache generation |
| `--variants-per-selected-chunk` | `1` | Number of random retarget variants generated for each selected chunk |
| `--ik-backend` | `jax` | IK backend; `jax` uses batched GPU execution, `numpy` uses the CPU path |
| `--batch-size` | `128` | JAX batch size for cache generation |
| `--future-near-mode-prob` | `0.5` | Per-variant probability of the future-near mode; `1 - p` uses correction mode |
| `--future-near-window-frames` | `6` | Local trajectory window used by future-near mode, including frame 0 input meta |
| `--future-near-position-noise-max-m` | `0.008` | Max position noise around the selected future frame |
| `--future-near-direction-noise-max-deg` | `7.0` | Max direction noise around the selected future frame |
| `--future-near-transition-steps` | `8` | Linear decay length from shifted virtual trajectory back to original trajectory |
| `--position-noise-max-m` | `0.04` | Max correction-mode xyz perturbation in meters |
| `--direction-noise-max-deg` | `35.0` | Max correction-mode direction perturbation for line/surface meta areas |
| `--correction-forward-exclusion-angle-deg` | `70.0` | Reject correction offsets inside this forward cone around the fitted local motion direction |
| `--correction-min-position-offset-m` | `0.02` | Correction mode requires this much position offset unless direction offset is large enough |
| `--correction-min-direction-offset-deg` | `13.0` | Correction mode requires this much direction offset unless position offset is large enough |
| `--approach-joint-step-rad` | `0.04` | Dynamic approach length is `ceil(max(|Δq_right|) / this)` |
| `--accept-max-camera-rotvec-norm-rad` | `3.143` | Reject cached variants whose recomputed camera rotvec is outside the canonical near-π range |
| `--accept-max-abs-action-value` | `1e4` | Reject catastrophic non-camera action values before cache write |

The builder writes accepted retargeted chunks to `variants/` and records them in `manifest.jsonl`.
If IK or validation fails, it retries random perturbations up to `--max-attempts-per-variant`; failed
records go to `failures.jsonl` and are not sampled during training. Validation also rejects
non-finite or clearly out-of-range retarget actions; camera rotvecs are recomputed with a
near-180-degree-stable SO(3) conversion and are bounded by
`--accept-max-camera-rotvec-norm-rad` before cache write.

Training-time sampling is independent from cache generation. The base `DataConfig` keeps retarget
disabled by default, so old/non-meta training configs do not use this path. The x-trainer
meta-aware configs default to `meta_retarget_cache_prob=0.5`; when
`data.meta_retarget_cache_dir` is set, each access to a cached chunk independently returns a
retarget variant with probability `0.5`, otherwise it returns the original chunk. With the default
cache coverage `--retarget-prob=0.5`, the global expected retarget training ratio is about
`0.5 × 0.5 = 0.25`. If a chunk has multiple cached variants, one variant is sampled randomly each
time.

### 4. Compute norm stats

```bash
python scripts/compute_norm_stats.py --config-name pi05_xtrainer_meta_aux \
    --overrides data.repo_id=/path/to/your/dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/retarget_cache
```

For 12D structured data:

```bash
python scripts/compute_norm_stats.py --config-name pi05_xtrainer_meta_aux_structured_12d_delta \
    --overrides data.repo_id=/path/to/your/structured_12d_lerobot_dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/retarget_cache_12d
```

This writes normalisation statistics to `assets/<repo_id>/norm_stats.json`.

### 5. Run training

```bash
# Full fine-tuning with meta-action head (recommended, requires A100/H100)
python scripts/train.py pi05_xtrainer_meta_aux \
    --exp-name my_run \
    --overrides data.repo_id=/path/to/your/dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/retarget_cache

# With delta actions
python scripts/train.py pi05_xtrainer_meta_aux_delta \
    --exp-name my_run_delta \
    --overrides data.repo_id=/path/to/your/dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/retarget_cache

# Low-memory LoRA variant (RTX 4090 / 24 GB)
python scripts/train.py pi05_xtrainer_meta_aux_low_mem_finetune \
    --exp-name my_run_lora \
    --overrides data.repo_id=/path/to/your/dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/retarget_cache
```

12D structured training:

```bash
python scripts/train.py pi05_xtrainer_meta_aux_structured_12d_delta \
    --exp-name my_run_12d \
    --overrides data.repo_id=/path/to/your/structured_12d_lerobot_dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/retarget_cache_12d
```

Training checkpoints are saved under `checkpoints/<exp-name>/`.

### 6. Serve the policy

```bash
python scripts/serve_policy.py \
    --config-name pi05_xtrainer_meta_aux \
    --checkpoint checkpoints/my_run/<step>
```

The server returns both `actions` (shape `[action_horizon, 32]`) and `meta_actions`
(shape `[action_horizon, M, meta_action_dim]`) when using a `*_aux` config. For the 12D config,
`meta_action_dim=12`; slot-0 runtime predictions are raw 12D values and are not action-normalized.

---

## Key Configuration Parameters (`Pi0Config`)

| Parameter | Default | Description |
|---|---|---|
| `meta_model` | `False` | Enable `Pi0Meta` instead of `Pi0` |
| `max_meta_areas` | `3` | Number of geometry slots M |
| `meta_area_type_vocab_size` | `3` | Number of discrete area-type labels |
| `num_meta_special_tokens` | `4` | Learned global context tokens S |
| `meta_loss_weight` | `1.0` | Weight of meta loss relative to base loss |
| `meta_action_start_dim` | `14` | First action dimension belonging to meta pose |
| `meta_action_dim` | `6` | Dimensionality of each meta target; set to `12` for structured 12D |
| `meta_area_pose_dim` | `6` | Dimensionality of input meta-area pose; set to `12` for `pose12d` |
| `meta_dropout_prob` | `0.0` | Probability of masking a meta slot during training |
