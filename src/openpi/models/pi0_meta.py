import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _expand_attention_mask(mask: at.Bool[at.Array, "b q k"], num_heads: int) -> at.Bool[at.Array, "b h q k"]:
    return einops.repeat(mask, "b q k -> b h q k", h=num_heads)


class MetaActionHead(nnx.Module):
    def __init__(self, width: int, num_heads: int, meta_action_dim: int, rngs: nnx.Rngs):
        self.action_norm = nnx.LayerNorm(width, rngs=rngs)
        self.meta_norm = nnx.LayerNorm(width, rngs=rngs)
        self.special_norm = nnx.LayerNorm(width, rngs=rngs)
        self.action_query_proj = nnx.Linear(width, width, rngs=rngs)
        self.meta_query_proj = nnx.Linear(width, width, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=width,
            qkv_features=width,
            out_features=width,
            deterministic=True,
            decode=False,
            rngs=rngs,
        )
        self.out_in = nnx.Linear(width * 3, width, rngs=rngs)
        self.out_out = nnx.Linear(width, meta_action_dim, rngs=rngs)

    def __call__(
        self,
        action_tokens: at.Float[at.Array, "b ah emb"],
        meta_tokens: at.Float[at.Array, "b m emb"],
        special_tokens: at.Float[at.Array, "b s emb"],
    ) -> at.Float[at.Array, "b ah m md"]:
        batch_size, action_horizon, width = action_tokens.shape
        meta_count = meta_tokens.shape[1]
        special_count = special_tokens.shape[1]

        action_features = self.action_norm(action_tokens)
        meta_features = self.meta_norm(meta_tokens)
        special_features = self.special_norm(special_tokens)

        action_queries = self.action_query_proj(action_features)[:, :, None, :]
        meta_queries = self.meta_query_proj(meta_features)[:, None, :, :]
        queries = action_queries + meta_queries

        slot_memory = einops.repeat(meta_features, "b m emb -> b ah m one emb", ah=action_horizon, one=1)
        special_memory = einops.repeat(special_features, "b s emb -> b ah m s emb", ah=action_horizon, m=meta_count)
        memory = jnp.concatenate([slot_memory, special_memory], axis=3)

        flat_queries = queries.reshape(batch_size * action_horizon * meta_count, 1, width)
        flat_memory = memory.reshape(batch_size * action_horizon * meta_count, 1 + special_count, width)
        context_mask = jnp.ones(
            (batch_size * action_horizon * meta_count, self.cross_attn.num_heads, 1, 1 + special_count),
            dtype=jnp.bool_,
        )
        attended = self.cross_attn(flat_queries, flat_memory, mask=context_mask, decode=False)
        attended = attended.reshape(batch_size, action_horizon, meta_count, width)

        action_context = einops.repeat(action_tokens, "b ah emb -> b ah m emb", m=meta_count)
        meta_context = einops.repeat(meta_tokens, "b m emb -> b ah m emb", ah=action_horizon)
        fused = jnp.concatenate([action_context, meta_context, attended], axis=-1)
        fused = self.out_in(fused)
        fused = nnx.swish(fused)
        return self.out_out(fused)


class Pi0Meta(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        if not config.pi05:
            raise ValueError("Pi0Meta only supports PI0.5-style models.")

        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = True
        self.max_meta_areas = config.max_meta_areas
        self.num_meta_special_tokens = config.num_meta_special_tokens
        self.meta_area_type_vocab_size = config.meta_area_type_vocab_size
        self.action_loss_weight = config.action_loss_weight
        self.meta_loss_weight = config.meta_loss_weight
        self.meta_action_start_dim = config.meta_action_start_dim
        self.meta_action_dim = config.meta_action_dim
        self.meta_actions_in_action_slice = config.meta_actions_in_action_slice
        self.meta_area_pose_dim = config.meta_area_pose_dim
        self.meta_dropout_prob = config.meta_dropout_prob
        self.use_meta_control_alpha = config.use_meta_control_alpha
        self.meta_loss_alpha_power = config.meta_loss_alpha_power
        self.meta_stop_backbone_grad = config.meta_stop_backbone_grad

        backbone_action_mask = [0.0] * config.action_dim
        for i in range(self.meta_action_start_dim):
            backbone_action_mask[i] = 1.0
        camera_start = (
            self.meta_action_start_dim + self.meta_action_dim
            if self.meta_actions_in_action_slice
            else self.meta_action_start_dim + 6
        )
        camera_end = min(camera_start + 6, config.action_dim)
        for i in range(camera_start, camera_end):
            backbone_action_mask[i] = 1.0
        self.backbone_action_mask_values = tuple(backbone_action_mask)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.prefix_token_width = paligemma_config.width
        self.action_token_width = action_expert_config.width
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.meta_special_embedding = nnx.Embed(
            num_embeddings=self.num_meta_special_tokens,
            features=paligemma_config.width,
            rngs=rngs,
        )
        self.meta_alpha_in = (
            nnx.Linear(1, paligemma_config.width, rngs=rngs) if self.use_meta_control_alpha else None
        )
        self.meta_default_embedding = nnx.Embed(num_embeddings=1, features=paligemma_config.width, rngs=rngs)
        self.meta_slot_embedding = nnx.Embed(num_embeddings=self.max_meta_areas, features=paligemma_config.width, rngs=rngs)
        self.meta_type_embedding = nnx.Embed(
            num_embeddings=self.meta_area_type_vocab_size,
            features=paligemma_config.width,
            rngs=rngs,
        )
        self.meta_pose_in = nnx.Linear(self.meta_area_pose_dim, paligemma_config.width, rngs=rngs)
        self.meta_pose_out = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
        self.meta_context_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
        self.meta_context_decode_proj = nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
        self.meta_special_decode_proj = nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
        self.meta_head = MetaActionHead(
            width=action_expert_config.width,
            num_heads=action_expert_config.num_heads,
            meta_action_dim=self.meta_action_dim,
            rngs=rngs,
        )

        self.deterministic = True

    def _mask_backbone_channels(self, values: at.Float[at.Array, "... ad"]) -> at.Float[at.Array, "... ad"]:
        mask = jnp.asarray(self.backbone_action_mask_values, dtype=values.dtype)
        return values * mask

    def _prepare_observation(self, observation: _model.Observation) -> _model.Observation:
        masked_state = self._mask_backbone_channels(observation.state)
        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=masked_state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
            meta_area_poses=observation.meta_area_poses,
            meta_area_dim_masks=observation.meta_area_dim_masks,
            meta_area_types=observation.meta_area_types,
            meta_area_masks=observation.meta_area_masks,
            meta_action_target_poses=observation.meta_action_target_poses,
            meta_action_target_dim_masks=observation.meta_action_target_dim_masks,
            meta_action_target_masks=observation.meta_action_target_masks,
            meta_control_alpha=observation.meta_control_alpha,
        )

    def _apply_meta_dropout(
        self, rng: at.KeyArrayLike | None, meta_masks: at.Bool[at.Array, "b m"], *, train: bool
    ) -> at.Bool[at.Array, "b m"]:
        if not train or self.meta_dropout_prob <= 0.0 or rng is None:
            return meta_masks
        keep_mask = jax.random.bernoulli(rng, 1.0 - self.meta_dropout_prob, meta_masks.shape)
        return jnp.logical_and(meta_masks, keep_mask)

    def _build_special_tokens(
        self, observation: _model.Observation
    ) -> at.Float[at.Array, "b s emb"]:
        batch_size = observation.state.shape[0]
        special_ids = jnp.arange(self.num_meta_special_tokens, dtype=jnp.int32)
        special_tokens = self.meta_special_embedding(special_ids)[None, :, :]
        special_tokens = jnp.broadcast_to(
            special_tokens, (batch_size, self.num_meta_special_tokens, special_tokens.shape[-1])
        )
        if self.meta_alpha_in is not None:
            if observation.meta_control_alpha is None:
                alpha = jnp.ones((batch_size,), dtype=observation.state.dtype)
            else:
                alpha = jnp.asarray(observation.meta_control_alpha, dtype=observation.state.dtype).reshape(batch_size)
            alpha_token = self.meta_alpha_in(alpha[:, None])
            special_tokens = special_tokens.at[:, 0, :].add(alpha_token)
        return special_tokens

    def _build_meta_context_tokens(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b m emb"], at.Bool[at.Array, "b m"]]:
        batch_size = observation.state.shape[0]
        if observation.meta_area_poses is None or observation.meta_area_types is None or observation.meta_area_masks is None:
            meta_area_poses = jnp.zeros((batch_size, self.max_meta_areas, self.meta_area_pose_dim), dtype=observation.state.dtype)
            meta_area_types = jnp.zeros((batch_size, self.max_meta_areas), dtype=jnp.int32)
            meta_area_masks = jnp.zeros((batch_size, self.max_meta_areas), dtype=jnp.bool_)
        else:
            meta_area_poses = observation.meta_area_poses
            meta_area_types = observation.meta_area_types
            meta_area_masks = observation.meta_area_masks

        slot_ids = jnp.arange(self.max_meta_areas, dtype=jnp.int32)
        slot_tokens = self.meta_slot_embedding(slot_ids)[None, :, :]
        slot_tokens = jnp.broadcast_to(slot_tokens, (batch_size, self.max_meta_areas, slot_tokens.shape[-1]))

        default_tokens = self.meta_default_embedding(jnp.zeros((batch_size, self.max_meta_areas), dtype=jnp.int32))
        default_tokens = default_tokens + slot_tokens

        provided_tokens = self.meta_pose_in(meta_area_poses)
        provided_tokens = nnx.swish(provided_tokens)
        provided_tokens = self.meta_pose_out(provided_tokens)
        provided_tokens = provided_tokens + self.meta_type_embedding(meta_area_types) + slot_tokens

        meta_tokens = jnp.where(meta_area_masks[..., None], provided_tokens, default_tokens)
        meta_tokens = self.meta_context_norm(meta_tokens)
        return meta_tokens, meta_area_masks

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        meta_context_tokens, meta_masks = self._build_meta_context_tokens(obs)
        tokens.append(meta_context_tokens)
        input_mask.append(jnp.ones(meta_context_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * meta_context_tokens.shape[1]

        special_tokens = self._build_special_tokens(obs)
        tokens.append(special_tokens)
        input_mask.append(jnp.ones((obs.state.shape[0], self.num_meta_special_tokens), dtype=jnp.bool_))
        ar_mask += [False] * self.num_meta_special_tokens

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        action_tokens = self.action_in_proj(self._mask_backbone_channels(noisy_actions))
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = nnx.swish(time_emb)
        tokens.append(action_tokens)
        input_mask.append(jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _decode_meta_actions(
        self,
        prefix_out: at.Float[at.Array, "b p emb"],
        suffix_out: at.Float[at.Array, "b s emb"],
    ) -> at.Float[at.Array, "b ah m md"]:
        special_tokens = self.meta_special_decode_proj(prefix_out[:, -self.num_meta_special_tokens :])
        meta_tokens = self.meta_context_decode_proj(
            prefix_out[:, -(self.num_meta_special_tokens + self.max_meta_areas) : -self.num_meta_special_tokens]
        )
        action_tokens = suffix_out[:, -self.action_horizon :]
        return self.meta_head(action_tokens, meta_tokens, special_tokens)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, meta_dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        observation = self._prepare_observation(observation)

        batch_shape = actions.shape[:-2]
        noise = self._mask_backbone_channels(jax.random.normal(noise_rng, actions.shape))
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        meta_supervision_masks = (
            jnp.zeros((observation.state.shape[0], self.max_meta_areas), dtype=jnp.bool_)
            if observation.meta_area_masks is None
            else observation.meta_area_masks
        )
        dropped_meta_masks = self._apply_meta_dropout(meta_dropout_rng, meta_supervision_masks, train=train)
        observation_for_meta = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
            meta_area_poses=observation.meta_area_poses,
            meta_area_dim_masks=observation.meta_area_dim_masks,
            meta_area_types=observation.meta_area_types,
            meta_area_masks=dropped_meta_masks,
            meta_action_target_poses=observation.meta_action_target_poses,
            meta_action_target_dim_masks=observation.meta_action_target_dim_masks,
            meta_action_target_masks=observation.meta_action_target_masks,
            meta_control_alpha=observation.meta_control_alpha,
        )

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation_for_meta)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation_for_meta, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        action_out = suffix_out[:, -self.action_horizon :]
        v_t = self._mask_backbone_channels(self.action_out_proj(action_out))
        backbone_action_mask = jnp.asarray(self.backbone_action_mask_values, dtype=v_t.dtype)
        squared_error = jnp.square(v_t - u_t) * backbone_action_mask[None, None, :]
        base_loss = jnp.sum(squared_error, axis=-1) / jnp.maximum(jnp.sum(backbone_action_mask), 1.0)

        # Optionally stop gradients from the meta loss flowing back into the backbone.
        # This protects pre-trained backbone weights from being corrupted by a randomly
        # initialised meta head during the early phase of fine-tuning.
        prefix_for_meta = jax.lax.stop_gradient(prefix_out) if self.meta_stop_backbone_grad else prefix_out
        suffix_for_meta = jax.lax.stop_gradient(suffix_out) if self.meta_stop_backbone_grad else suffix_out
        meta_pred = self._decode_meta_actions(prefix_for_meta, suffix_for_meta)
        if not self.meta_actions_in_action_slice:
            if observation.meta_action_target_poses is None:
                meta_target = jnp.zeros_like(meta_pred)
                meta_loss_mask = jnp.zeros(meta_pred.shape[:3], dtype=jnp.bool_)
                meta_dim_weights = jnp.ones(meta_pred.shape, dtype=meta_pred.dtype)
            else:
                target_poses = observation.meta_action_target_poses[..., : self.meta_action_dim]
                if target_poses.ndim == 3:
                    target_poses = target_poses[:, None, :, :]
                if target_poses.shape[1] == 1 and meta_pred.shape[1] > 1:
                    target_poses = jnp.broadcast_to(
                        target_poses,
                        (target_poses.shape[0], meta_pred.shape[1], target_poses.shape[2], target_poses.shape[3]),
                    )
                horizon = min(meta_pred.shape[1], target_poses.shape[1])
                slots = min(meta_pred.shape[2], target_poses.shape[2])
                meta_target = jnp.zeros_like(meta_pred)
                meta_target = meta_target.at[:, :horizon, :slots, :].set(target_poses[:, :horizon, :slots, :])

                if observation.meta_action_target_masks is None:
                    target_masks = (
                        jnp.ones(meta_pred.shape[:3], dtype=jnp.bool_)
                        if observation.meta_area_masks is None
                        else jnp.broadcast_to(observation.meta_area_masks[:, None, :], meta_pred.shape[:3])
                    )
                else:
                    target_masks = observation.meta_action_target_masks
                    if target_masks.ndim == 2:
                        target_masks = target_masks[:, None, :]
                    if target_masks.shape[1] == 1 and meta_pred.shape[1] > 1:
                        target_masks = jnp.broadcast_to(target_masks, meta_pred.shape[:3])
                meta_loss_mask = jnp.zeros(meta_pred.shape[:3], dtype=jnp.bool_)
                meta_loss_mask = meta_loss_mask.at[:, :horizon, :slots].set(target_masks[:, :horizon, :slots])

                if observation.meta_action_target_dim_masks is not None:
                    target_dim_masks = observation.meta_action_target_dim_masks[..., : self.meta_action_dim]
                    if target_dim_masks.ndim == 3:
                        target_dim_masks = target_dim_masks[:, None, :, :]
                    if target_dim_masks.shape[1] == 1 and meta_pred.shape[1] > 1:
                        target_dim_masks = jnp.broadcast_to(
                            target_dim_masks,
                            (
                                target_dim_masks.shape[0],
                                meta_pred.shape[1],
                                target_dim_masks.shape[2],
                                target_dim_masks.shape[3],
                            ),
                        )
                elif observation.meta_area_dim_masks is not None:
                    target_dim_masks = jnp.broadcast_to(
                        observation.meta_area_dim_masks[:, None, :, : self.meta_action_dim],
                        meta_pred.shape,
                    )
                else:
                    target_dim_masks = jnp.ones(meta_pred.shape, dtype=jnp.bool_)
                meta_dim_weights = jnp.zeros_like(meta_pred, dtype=meta_pred.dtype)
                meta_dim_weights = meta_dim_weights.at[:, :horizon, :slots, :].set(
                    target_dim_masks[:, :horizon, :slots, :].astype(meta_pred.dtype)
                )
        else:
            meta_target = jnp.zeros_like(meta_pred)
            meta_target = meta_target.at[:, :, 0, :].set(
                actions[:, :, self.meta_action_start_dim : self.meta_action_start_dim + self.meta_action_dim]
            )
            meta_loss_mask = jnp.zeros(meta_pred.shape[:3], dtype=jnp.bool_).at[:, :, 0].set(True)
            meta_loss_mask = jnp.logical_and(meta_loss_mask, meta_supervision_masks[:, None, :])
            if observation.meta_area_dim_masks is None:
                meta_dim_masks = jnp.ones(meta_pred.shape[0:1] + meta_pred.shape[2:], dtype=jnp.bool_)
            else:
                meta_dim_masks = observation.meta_area_dim_masks[..., : self.meta_action_dim]
            meta_dim_weights = meta_dim_masks[:, None, :, :].astype(meta_pred.dtype)
        meta_dim_denom = jnp.maximum(jnp.sum(meta_dim_weights, axis=-1), 1.0)
        meta_loss = jnp.sum(jnp.square(meta_pred - meta_target) * meta_dim_weights, axis=-1) / meta_dim_denom
        denom = jnp.maximum(jnp.sum(meta_loss_mask, axis=-1), 1)
        meta_loss = jnp.sum(meta_loss * meta_loss_mask, axis=-1) / denom

        if self.use_meta_control_alpha and self.meta_loss_alpha_power > 0.0:
            if observation.meta_control_alpha is None:
                alpha = jnp.ones((observation.state.shape[0],), dtype=meta_loss.dtype)
            else:
                alpha = jnp.asarray(observation.meta_control_alpha, dtype=meta_loss.dtype).reshape(
                    observation.state.shape[0]
                )
            alpha = jnp.clip(alpha, 0.0, 1.0)
            meta_loss_scale = jnp.power(alpha, self.meta_loss_alpha_power)[:, None]
        else:
            meta_loss_scale = 1.0

        return self.action_loss_weight * base_loss + self.meta_loss_weight * meta_loss_scale * meta_loss

    def sample_actions_with_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> dict[str, _model.Actions | at.Array]:
        observation = _model.preprocess_observation(None, observation, train=False)
        observation = self._prepare_observation(observation)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        noise = self._mask_backbone_channels(noise)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_outputs, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
        prefix_out = prefix_outputs[0] if isinstance(prefix_outputs, (tuple, list)) else prefix_outputs
        assert prefix_out is not None

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self._mask_backbone_channels(self.action_out_proj(suffix_out[:, -self.action_horizon :]))
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        zero_time = jnp.zeros((batch_size,), dtype=x_0.dtype)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_0, zero_time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        meta_actions = self._decode_meta_actions(prefix_out, suffix_out)
        return {"actions": self._mask_backbone_channels(x_0), "meta_actions": meta_actions}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        return self.sample_actions_with_aux(rng, observation, num_steps=num_steps, noise=noise)["actions"]  # type: ignore[return-value]
