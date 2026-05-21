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
observation.tool_instance_hash      [1] optional but required by beta same-tool sampling
observation.source_type_id          [1] optional lineage id: 0=origin, 1=retarget, 2=imagine
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

The beta config is `pi05_xtrainer_meta_aux_structured_12d_delta_beta`.
It keeps the same raw 12D fields but interprets them through a chunk-pair meta-token interface:

```text
chunk1 condition path:
  condition observation tokens + condition meta/query tokens (+ optional reference-action tokens)
      -> refined chunk1 meta tokens

chunk2 execution path:
  current observation tokens + refined chunk1 meta tokens + special tokens -> action suffix

meta-action head:
  action tokens + refined chunk1 meta tokens + special tokens -> meta_actions
```

`execution_meta_areas` is still produced by the beta dataloader for target construction and masks,
but the beta2 model no longer builds a separate chunk2 execution-meta token segment from it.
Therefore:

- `meta_areas` means the chunk1 condition when meta-area conditioning is selected.
- `reference_actions` means the chunk1 demonstration action condition when reference conditioning is selected.
- Reference-action and obs-only conditioning use learnable/default meta query slots instead of explicit
  condition meta-area input.
- The refined chunk1 meta tokens are injected into chunk2 execution directly. Runtime meta-area inference
  uses the current observation as both chunk1 condition observation and chunk2 execution observation.
- `execution_meta_areas` still defines the semantic target frame for supervised `meta_action_targets`, but
  those fields are not visible to the action suffix as an independent token segment.

The beta prefix token order is:

```text
chunk1 condition encoder:
  condition_obs -> condition_meta/query -> reference_action

chunk2 execution encoder:
  current_obs -> refined_chunk1_meta -> special
```

Attention is intentionally restricted:

- In the chunk1 encoder, refined meta/query tokens can read condition observation and the selected
  condition source. Meta-area and reference-action contrastive views are encoded as two independent
  passes, so their condition tokens do not directly attend each other.
- Reference-action tokens attend to condition observation and themselves.
- In the chunk2 encoder, refined chunk1 meta tokens stay as the tool-operation representation; special
  tokens read current observation plus refined meta tokens.
- The action suffix reads current observation, refined meta tokens, and special tokens.
- The meta-action head reads action tokens, refined meta tokens, and special tokens.

Beta training can additionally compute a small cosine contrastive loss between the two chunk1
condition views. For non-obs-only samples, the dataloader keeps `contrastive_meta_areas` and
`contrastive_reference_actions` from the same chunk1/source sample, including pair-cache and
imagine chunk-cache samples. The model encodes these as two independent condition passes,
`condition obs + meta_area -> refined_meta_tokens` and
`condition obs + reference_actions + meta queries -> refined_meta_tokens`, then applies token-wise
normalized squared L2 alignment over the refined meta tokens from every LLM layer. Layer weights grow
quadratically with depth, `w_l=((l+1)/L)^2`, and are normalized to mean 1, so later layers dominate
without changing the overall loss scale when model depth changes. Only the sampled selected condition
path is injected into chunk2 execution; the contrastive branch is training-only. The weight is
controlled by `model.meta_contrastive_loss_weight` and defaults to `0.0`; the current beta configs set
it to `0.4`.
Obs-only samples carry zero contrastive masks and do not contribute to this loss. When tuning this
weight, inspect the unweighted/raw action, meta-action, and contrastive loss magnitudes together; if
raw contrastive loss is around `0.7-1.2`, `0.4` contributes roughly `0.28-0.48` before any task-loss
scaling, so reduce it if action or meta-action loss regresses.

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
| `pi05_xtrainer_meta_aux_structured_12d_delta_beta` | π₀.₅ | 32 | Yes | Yes | Beta latent chunk-pair 12D training |
| `pi05_xtrainer_meta_aux_structured_12d_delta_beta_black_ring_hookNewUpper30_60_stick10_9type_12D_classified_stride3` | π₀.₅ | 32 | Yes | Yes | Dataset-specific beta config for classified black-ring 12D data |
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

For `pi05_xtrainer_meta_aux_structured_12d_delta_beta`, the dataloader samples a chunk1/chunk2
relationship before normalization:

```text
self same chunk:             0.12
same episode different chunk:0.08
same tool different episode: 0.50
retarget conditioned:        0.30
```

Condition type is sampled separately. Non-retarget samples reserve 10% for obs-only conditioning.
Retarget-conditioned samples default to 0% obs-only conditioning, so they must provide either
meta-area or reference-action condition tokens. If chunk1/source has `observation.source_type_id != 0`,
obs-only conditioning is disabled and the meta/reference probabilities are re-normalized, because
dropping the condition would discard the imagined source.
The beta wrapper requires `observation.tool_instance_hash` for same-tool sampling. If the dataset
lacks it, the structured input transform falls back to hash 0, which keeps training runnable but
disables meaningful same-tool grouping.

The beta model is a two-pass model. The chunk1/source pass uses the existing Pi0.5 observation
prefix format with condition image plus `condition_tokenized_prompt`, where that prompt is generated
from the source `condition_prompt` when available, otherwise the current prompt, and normalized
`condition_state`. The chunk2/execution pass uses the normal Pi0.5 observation prefix with current
image plus `tokenized_prompt`, generated from the current normalized `state`, and then appends latent
tokens to generate actions. `condition_image`, `condition_image_mask`, `condition_state`,
`condition_tokenized_prompt`, and
`condition_tokenized_prompt_mask` are required for the beta latent path; missing condition observation
fields are treated as data errors instead of falling back to the current execution observation. The
beta code does not add a new continuous state projection on top of the existing Pi0.5 state-in-prompt
encoding.

Reference-action conditioning only encodes the first 14 qpos action dimensions. During training, the
source `actions` have already passed through the normal delta-action transform, so dims `0:6` and
`7:13` are relative to the same source state copied into `condition_state`; gripper dims `6` and `13`
remain absolute. The runtime HDF5 reference loader follows the same rule: it samples the reference
trajectory, subtracts the reference start-frame qpos for dims `0:6` and `7:13`, returns 14D
`reference_actions`, and supplies that start-frame qpos as `condition_state`.

`meta_control.imagination_alpha` is no longer injected into beta special tokens. Current beta training
and inference force this value to `0.0` and keep the field only as a compatibility/debug value.

Pair retarget has two modes. In the default same-tool setting
(`data.meta_beta_pair_retarget_same_tool_only=true`), retarget-conditioned training first samples the
condition type: current configs use `meta_area=0.85`, `reference_action=0.15`, and `obs_only=0.0`.
For meta-area conditioning, `data.meta_beta_imagine_cache_condition_prob` controls the direct
chunk-cache shortcut. Current configs set it to `0.90`, so the direct shortcut probability inside
relation-3 is `0.85 * 0.90`. The direct shortcut samples a same-tool accepted record from the
chunk-level imagine retarget cache at `data.meta_retarget_cache_dir`; that cached retargeted chunk is
used as both chunk1 and chunk2.

The remaining retarget-conditioned samples use the pair retarget path: chunk2/target must be an origin
sample, chunk1/source must be a non-origin imagine/retarget sample from the same `tool_instance_id`.
This pair path can consume a beta pair cache, a ready online result, or the bounded sync generator. It
does not allow same-tool origin-origin pair retarget, because origin-origin pairs are not semantically
retarget-conditioned and would corrupt the intended relation ratio. If the selected direct chunk-cache
shortcut has no same-tool accepted record, training raises an error so the chunk cache can be rebuilt
instead of silently changing the sample distribution.

For same-tool beta pair caches, `source_index` is the base/origin row used by a chunk-level retarget
record, not a raw imagine row in the LeRobot dataset. The manifest must also contain
`source_retarget_path`, which points into `data.meta_retarget_cache_dir`; training reconstructs chunk1
by loading that chunk retarget payload and applying it to the base source sample. Pair-cache records
without `source_retarget_path` are treated as legacy same-tool origin-origin caches, skipped with a
warning, and should be deleted/rebuilt. The current beta same-tool source reconstruction is supported
for structured 12D caches only; legacy 6D `meta_action_targets` are not supported in this path.

Cross-tool chunk1/chunk2 pair retargets are intentionally disabled by default because asymmetric IK
success rates can bias the distribution toward tools that are easy to retarget into. Use
`meta_beta_pair_retarget_same_tool_only=false` only for explicit cross-tool ablations. Any beta pair
cache built with same-tool origin-origin pairs must be discarded; it no longer represents the intended
training distribution.

The online producer can run in two worker groups. `data.meta_beta_online_worker_group="dataloader"`
keeps the older per-DataLoader-worker thread producer. `data.meta_beta_online_worker_group="process"`
starts a separate multiprocessing retarget worker group from the main process and shares request/result
queues with the DataLoader workers. Process mode is experimental: it can isolate IK from DataLoader work,
but it pays IPC and dataset-copy overhead and should only be used after an A/B test shows higher online
retarget throughput. In default same-tool beta retarget, process-mode requests include
`source_retarget_path`; the process worker reconstructs chunk1/source from `data.meta_retarget_cache_dir`
before running pair retarget, so it does not fall back to origin-origin pairs. Producer polling is
demand-driven: it happens when a retarget-conditioned sample is requested, not at the start of every
`__getitem__`.

`data.meta_beta_online_num_workers` controls producer threads in dataloader mode or producer processes in
process mode. `data.meta_beta_online_queue_size` bounds local ready results, `data.meta_beta_online_max_pending`
caps expensive submissions per demand poll, and `data.meta_beta_online_request_queue_size` /
`data.meta_beta_online_result_queue_size` bound the shared process-mode queues. The beta pair-cache
fallback is built by `scripts/build_xtrainer_beta_pair_retarget_cache.py`; in same-tool mode it stores
origin-target plus imagine-source records, never origin-origin pairs.

The dataset-specific beta config
`pi05_xtrainer_meta_aux_structured_12d_delta_beta_black_ring_hookNewUpper30_60_stick10_9type_12D_classified_stride3`
sets both `model.max_meta_areas=1` and `data.max_meta_areas=1`. This is required because the beta
refined meta-token slot count is `max_meta_areas`; if the model keeps the default `max_meta_areas=3`
while the data emits one slot, the model creates extra prefix tokens and the token mask shapes no
longer match.

That config also sets `data.meta_retarget_cache_prob=1.0`. In the beta dataloader, relation sampling
already controls how often retarget-conditioned pairs are attempted through
`data.meta_beta_retarget_conditioned_prob=0.15`; leaving the old `0.2` value would add a second random
gate and reduce the effective attempt rate to `0.15 * 0.20 = 0.03`.

The default beta relation mix is now `self_same_chunk=0.12`,
`same_episode_diff_chunk=0.08`, `same_tool_diff_episode=0.65`, and
`retarget_conditioned=0.15`. Retarget-conditioned samples use a separate condition ratio:
`meta_area=0.85`, `reference_action=0.15`, and `obs_only=0.0`; this does not change the
non-retarget condition ratios. For `self_same_chunk`, reference-action conditioning is capped at
`0.20`; the removed reference probability is assigned to obs-only conditioning, yielding the default
self condition mix `meta_area=0.45`, `reference_action=0.20`, `obs_only=0.35`.

For throughput, `TrainConfig.data_loader_prefetch_factor` defaults to `4` when `num_workers > 0`.
Increasing `num_workers`, building a beta pair cache, enabling the online producer, and using
`train_data_prefetch_buffer` are the main levers for keeping GPU utilization high.

For throughput diagnosis, set:

```bash
export OPENPI_TRAIN_TIMING_INTERVAL=20
export OPENPI_TRAIN_TIMING_BLOCK_UNTIL_READY=1  # optional; diagnostic only
export OPENPI_TRAIN_DATA_PREFETCH_BUFFER=1      # optional; overrides TrainConfig
```

This logs `TRAIN_TIMING` with average `data_wait_s`, `step_dispatch_s`, optional device block time,
log-reduce time, and checkpoint-trigger time. `OPENPI_TRAIN_DATA_PREFETCH_BUFFER=1` starts one
main-process background data prefetch thread so the next batch can be fetched while the current
JAX step is running. For beta data-loader source diagnostics, set:

```bash
--overrides data.meta_beta_debug_stats_enabled=true \
--overrides data.meta_beta_debug_stats_interval=1000
```

This logs per-worker `BETA_DATALOADER_STATS`, including relation/condition counts, retarget source
counts (`online`, `cache`, `sync`, `origin_fallback`, `gated`), online queue lengths, online
accepted/rejected counts, cache hits/misses, and average getitem/cache/online timings.

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
| `--approach-joint-step-rad` | `0.03` | Dynamic approach length is `ceil(max(|Δq_right|) / this)` |
| `--accept-max-camera-rotvec-norm-rad` | `3.143` | Reject cached variants whose recomputed camera rotvec is outside the canonical near-π range |
| `--accept-max-abs-action-value` | `1e4` | Reject catastrophic non-camera action values before cache write |
| `--retarget-algorithm` | config `data.meta_retarget_algorithm` | Retarget planner: `legacy_structured_min_rotation` or `wrist_pose_v2` |

The builder writes accepted retargeted chunks to `variants/` and records them in `manifest.jsonl`.
If IK or validation fails, it retries random perturbations up to `--max-attempts-per-variant`; failed
records go to `failures.jsonl` and are not sampled during training. Validation also rejects
non-finite or clearly out-of-range retarget actions; camera rotvecs are recomputed with a
near-180-degree-stable SO(3) conversion and are bounded by
`--accept-max-camera-rotvec-norm-rad` before cache write.

`data.meta_retarget_algorithm` controls the planner used by chunk-cache generation, beta pair-cache
generation, and training-time online/sync beta pair retargeting. The default
`legacy_structured_min_rotation` keeps the pre-alpha/beta structured retarget behavior: correction
first aligns to the original trajectory start, future-near follows the old smooth shifted-trajectory
decay, and IK directly minimizes the meta feature residual. The only intentional legacy fix is 12D
shape sign handling: shape axes remain undirected, so when shape and approach can both be aligned,
the planner chooses the lower-angle rotation instead of an arbitrary near-180-degree equivalent.
Use `--retarget-algorithm wrist_pose_v2` or `--overrides data.meta_retarget_algorithm=wrist_pose_v2`
only for explicit new-planner ablations. Cache metadata records the algorithm; training warns, but
does not reject, if a configured algorithm differs from an existing cache. Caches generated before
this switch do not contain a `retarget_algorithm` field; they are treated as backward-compatible and
will also emit only this warning, not a hard error.

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

Beta latent 12D structured training:

```bash
python scripts/train.py pi05_xtrainer_meta_aux_structured_12d_delta_beta \
    --exp-name my_beta_run_12d \
    --overrides data.repo_id=/path/to/your/classified_structured_12d_lerobot_dataset
```

Optional beta pair cache generation:

```bash
python scripts/build_xtrainer_beta_pair_retarget_cache.py \
    --config-name pi05_xtrainer_meta_aux_structured_12d_delta_beta \
    --output-dir /path/to/beta_pair_cache \
    --source-retarget-cache-dir /path/to/chunk_retarget_cache_12d \
    --variants-per-target 2 \
    --num-workers 8 \
    --max-attempts-per-pair 4
```

The beta pair-cache builder uses the same `data.meta_retarget_algorithm` default as training. In
default same-tool mode it samples origin targets and same-tool imagine sources reconstructed from the
chunk-level retarget cache. Pass `--source-retarget-cache-dir` explicitly, or let the builder use
`data.meta_retarget_cache_dir`; as a local pipeline fallback it also checks
`<beta_pair_cache_parent>/retarget_cache`. Rebuild both the chunk-level retarget cache and beta pair cache
when switching algorithms or after changing the same-tool origin/imagine pairing rule; any old same-tool
origin-origin beta pair cache should be discarded.

Default same-tool beta training:

```bash
python scripts/train.py pi05_xtrainer_meta_aux_structured_12d_delta_beta \
    --exp-name my_beta_run_12d \
    --overrides data.repo_id=/path/to/your/classified_structured_12d_lerobot_dataset \
    --overrides data.meta_retarget_cache_dir=/path/to/chunk_retarget_cache_12d \
    --overrides data.meta_beta_pair_cache_dir=/path/to/beta_pair_cache \
    --overrides data.meta_beta_pair_retarget_same_tool_only=true \
    --overrides data.meta_beta_online_async_enabled=true \
    --overrides num_workers=12 \
    --overrides train_data_prefetch_buffer=8
```

Recommended same-tool throughput settings for the classified black-ring stride3 dataset:

```text
default same-tool:
  num_workers=12
  data_loader_prefetch_factor=4
  train_data_prefetch_buffer=8
  data.meta_retarget_cache_dir=/path/to/chunk_retarget_cache_12d
  data.meta_beta_pair_cache_dir=/path/to/beta_pair_cache
  data.meta_beta_pair_retarget_same_tool_only=true
  data.meta_beta_online_async_enabled=true

cache-first:
  num_workers=12
  data_loader_prefetch_factor=4
  train_data_prefetch_buffer=8
  data.meta_beta_pair_cache_dir=/path/to/beta_pair_cache
  data.meta_beta_pair_retarget_same_tool_only=true
  data.meta_beta_online_async_enabled=true
  data.meta_beta_online_prefer_prob=0.0

throttled-online:
  num_workers=12
  data_loader_prefetch_factor=4
  train_data_prefetch_buffer=8
  data.meta_beta_pair_cache_dir=/path/to/beta_pair_cache
  data.meta_beta_pair_retarget_same_tool_only=true
  data.meta_beta_online_async_enabled=true
  data.meta_beta_online_worker_group=dataloader
  data.meta_beta_online_num_workers=1
  data.meta_beta_online_queue_size=16
  data.meta_beta_online_max_pending=4
  data.meta_beta_online_submit_prob=1.0

experimental separate-online-workgroup:
  num_workers=12
  data_loader_prefetch_factor=4
  train_data_prefetch_buffer=8
  data.meta_beta_pair_cache_dir=/path/to/beta_pair_cache
  data.meta_beta_pair_retarget_same_tool_only=true
  data.meta_beta_online_async_enabled=true
  data.meta_beta_online_worker_group=process
  data.meta_beta_online_num_workers=16
  data.meta_beta_online_queue_size=64
  data.meta_beta_online_max_pending=4
  data.meta_beta_online_submit_prob=1.0
  data.meta_beta_online_request_queue_size=256
  data.meta_beta_online_result_queue_size=256
```

For each run, compare average step time after warmup, GPU-utilization pattern, `TRAIN_TIMING`
`data_wait_s`, `BETA_DATALOADER_STATS` retarget source counts, and loss/grad-norm sanity.

Classified black-ring stride3 beta training:

```bash
python scripts/train.py pi05_xtrainer_meta_aux_structured_12d_delta_beta_black_ring_hookNewUpper30_60_stick10_9type_12D_classified_stride3 \
    --exp-name black_ring_beta_classified_stride3
```

For beta training, prefer datasets that include `observation.tool_instance_hash` and
`observation.source_type_id`, for example a classified copy under
`datasets_lerobot_structured/dataset_black_ring_12D_classified/`.

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
| `meta_beta_model` | `False` | Switch PI0.5 meta creation to the beta chunk-pair meta-token model |
| `num_meta_latent_tokens` | `4` | Deprecated beta2 no-op retained for old config compatibility; refined meta-token count is `max_meta_areas` |
| `reference_action_group_size` | `5` | Number of action steps compressed into one reference-action token |
