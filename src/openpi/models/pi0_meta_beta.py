import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_meta import Pi0Meta
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class _BetaPrefix:
    tokens: at.Float[at.Array, "b p emb"]
    input_mask: at.Bool[at.Array, "b p"]
    attn_mask: at.Bool[at.Array, "b p p"]
    action_visible_mask: at.Bool[at.Array, "b p"]


@dataclasses.dataclass(frozen=True)
class _ConditionPrefix:
    tokens: at.Float[at.Array, "b p emb"]
    input_mask: at.Bool[at.Array, "b p"]
    attn_mask: at.Bool[at.Array, "b p p"]
    latent_start: int
    latent_end: int


class Pi0MetaBeta(Pi0Meta):
    """Beta structured-meta model with latent tokens.

    Raw meta-area and reference-action condition tokens are prefix-only inputs.
    The action expert is masked so it only attends to observation, latent, and
    special tokens. This keeps the latent bottleneck explicit.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        if config.action_horizon % config.reference_action_group_size != 0:
            raise ValueError(
                "Pi0MetaBeta requires action_horizon to be divisible by reference_action_group_size; "
                f"got horizon={config.action_horizon}, group={config.reference_action_group_size}"
            )
        self.num_meta_latent_tokens = int(config.num_meta_latent_tokens)
        self.reference_action_group_size = int(config.reference_action_group_size)
        self.reference_action_dim = int(config.reference_action_dim)
        if self.reference_action_dim <= 0 or self.reference_action_dim > config.action_dim:
            raise ValueError(
                "Pi0MetaBeta requires 0 < reference_action_dim <= action_dim; "
                f"got reference_action_dim={self.reference_action_dim}, action_dim={config.action_dim}"
            )
        self.num_reference_action_tokens = config.action_horizon // self.reference_action_group_size

        width = self.prefix_token_width
        self.reference_action_in = nnx.Linear(
            self.reference_action_group_size * self.reference_action_dim,
            width,
            rngs=rngs,
        )
        self.reference_action_out = nnx.Linear(width, width, rngs=rngs)
        self.reference_position_embedding = nnx.Embed(
            num_embeddings=self.num_reference_action_tokens,
            features=width,
            rngs=rngs,
        )
        self.latent_embedding = nnx.Embed(
            num_embeddings=self.num_meta_latent_tokens,
            features=width,
            rngs=rngs,
        )
        self.latent_decode_proj = nnx.Linear(width, self.action_token_width, rngs=rngs)
        self.meta_imagination_in = nnx.Linear(1, width, rngs=rngs)

    def _build_observation_tokens(
        self,
        obs: _model.Observation,
        *,
        images: dict[str, at.Float[at.Array, "b h w c"]] | None = None,
        image_masks: dict[str, at.Bool[at.Array, "b"]] | None = None,
        tokenized_prompt: at.Int[at.Array, "b l"] | None = None,
        tokenized_prompt_mask: at.Bool[at.Array, "b l"] | None = None,
    ) -> tuple[at.Float[at.Array, "b o emb"], at.Bool[at.Array, "b o"]]:
        images = obs.images if images is None else images
        image_masks = obs.image_masks if image_masks is None else image_masks
        tokenized_prompt = obs.tokenized_prompt if tokenized_prompt is None else tokenized_prompt
        tokenized_prompt_mask = obs.tokenized_prompt_mask if tokenized_prompt_mask is None else tokenized_prompt_mask
        tokens = []
        input_mask = []
        for name in images:
            image_tokens, _ = self.PaliGemma.img(images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(image_masks[name], "b -> b s", s=image_tokens.shape[1]))

        if tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(tokenized_prompt_mask)

        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1)

    @staticmethod
    def _require_condition_observation(
        obs: _model.Observation,
    ) -> tuple[
        dict[str, at.Float[at.Array, "b h w c"]],
        dict[str, at.Bool[at.Array, "b"]],
        at.Int[at.Array, "b l"],
        at.Bool[at.Array, "b l"],
    ]:
        missing = []
        if obs.condition_images is None:
            missing.append("condition_image")
        if obs.condition_image_masks is None:
            missing.append("condition_image_mask")
        if obs.condition_state is None:
            missing.append("condition_state")
        if obs.condition_tokenized_prompt is None:
            missing.append("condition_tokenized_prompt")
        if obs.condition_tokenized_prompt_mask is None:
            missing.append("condition_tokenized_prompt_mask")
        if missing:
            raise ValueError(
                "Pi0MetaBeta requires chunk1 condition observation fields for latent encoding; "
                f"missing: {', '.join(missing)}"
            )
        return (
            obs.condition_images,
            obs.condition_image_masks,
            obs.condition_tokenized_prompt,
            obs.condition_tokenized_prompt_mask,
        )

    def _build_reference_action_tokens(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b r emb"], at.Bool[at.Array, "b r"]]:
        batch_size = observation.state.shape[0]
        if observation.reference_actions is None:
            reference_actions = jnp.zeros(
                (batch_size, self.action_horizon, self.reference_action_dim),
                dtype=observation.state.dtype,
            )
            ref_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
        else:
            reference_actions = observation.reference_actions[..., : self.reference_action_dim]
            if reference_actions.shape[-1] != self.reference_action_dim:
                raise ValueError(
                    "reference_actions last dimension is smaller than reference_action_dim: "
                    f"{reference_actions.shape[-1]} < {self.reference_action_dim}"
                )
            ref_mask = (
                jnp.ones((batch_size,), dtype=jnp.bool_)
                if observation.reference_action_mask is None
                else observation.reference_action_mask.astype(jnp.bool_)
            )

        grouped = reference_actions.reshape(
            batch_size,
            self.num_reference_action_tokens,
            self.reference_action_group_size * self.reference_action_dim,
        )
        ref_tokens = self.reference_action_in(grouped)
        ref_tokens = nnx.swish(ref_tokens)
        ref_tokens = self.reference_action_out(ref_tokens)
        pos_ids = jnp.arange(self.num_reference_action_tokens, dtype=jnp.int32)
        ref_tokens = ref_tokens + self.reference_position_embedding(pos_ids)[None, :, :]
        return ref_tokens, einops.repeat(ref_mask, "b -> b r", r=self.num_reference_action_tokens)

    def _build_beta_special_tokens(self, observation: _model.Observation) -> at.Float[at.Array, "b s emb"]:
        return self._build_special_tokens(observation)

    def _build_execution_meta_context_tokens(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b m emb"], at.Bool[at.Array, "b m"]]:
        if observation.execution_meta_area_poses is None:
            return self._build_meta_context_tokens(observation)
        execution_observation = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            condition_images=observation.condition_images,
            condition_image_masks=observation.condition_image_masks,
            condition_state=observation.condition_state,
            condition_tokenized_prompt=observation.condition_tokenized_prompt,
            condition_tokenized_prompt_mask=observation.condition_tokenized_prompt_mask,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
            meta_area_poses=observation.execution_meta_area_poses,
            meta_area_dim_masks=observation.execution_meta_area_dim_masks,
            meta_area_types=observation.execution_meta_area_types,
            meta_area_masks=observation.execution_meta_area_masks,
            execution_meta_area_poses=observation.execution_meta_area_poses,
            execution_meta_area_dim_masks=observation.execution_meta_area_dim_masks,
            execution_meta_area_types=observation.execution_meta_area_types,
            execution_meta_area_masks=observation.execution_meta_area_masks,
            meta_action_target_poses=observation.meta_action_target_poses,
            meta_action_target_dim_masks=observation.meta_action_target_dim_masks,
            meta_action_target_masks=observation.meta_action_target_masks,
            meta_control_alpha=observation.meta_control_alpha,
            reference_actions=observation.reference_actions,
            reference_action_mask=observation.reference_action_mask,
            meta_imagination_alpha=observation.meta_imagination_alpha,
        )
        return self._build_meta_context_tokens(execution_observation)

    def _build_condition_prefix(self, obs: _model.Observation) -> _ConditionPrefix:
        condition_images, condition_masks, condition_prompt, condition_prompt_mask = self._require_condition_observation(
            obs
        )
        obs_tokens, obs_mask = self._build_observation_tokens(
            obs,
            images=condition_images,
            image_masks=condition_masks,
            tokenized_prompt=condition_prompt,
            tokenized_prompt_mask=condition_prompt_mask,
        )
        meta_tokens, meta_mask = self._build_meta_context_tokens(obs)
        ref_tokens, ref_mask = self._build_reference_action_tokens(obs)

        latent_ids = jnp.arange(self.num_meta_latent_tokens, dtype=jnp.int32)
        latent_tokens = self.latent_embedding(latent_ids)[None, :, :]
        latent_tokens = jnp.broadcast_to(latent_tokens, (obs.state.shape[0], self.num_meta_latent_tokens, latent_tokens.shape[-1]))
        latent_mask = jnp.ones(latent_tokens.shape[:2], dtype=jnp.bool_)

        tokens = jnp.concatenate([obs_tokens, meta_tokens, ref_tokens, latent_tokens], axis=1)
        input_mask = jnp.concatenate([obs_mask, meta_mask, ref_mask, latent_mask], axis=1)

        obs_len = obs_tokens.shape[1]
        meta_len = meta_tokens.shape[1]
        ref_len = ref_tokens.shape[1]
        latent_len = latent_tokens.shape[1]
        segments = jnp.concatenate(
            [
                jnp.zeros((obs_len,), dtype=jnp.int32),
                jnp.ones((meta_len,), dtype=jnp.int32),
                jnp.full((ref_len,), 2, dtype=jnp.int32),
                jnp.full((latent_len,), 3, dtype=jnp.int32),
            ],
            axis=0,
        )
        q_seg = segments[:, None]
        k_seg = segments[None, :]
        allowed = jnp.logical_or(
            jnp.logical_and(q_seg == 0, k_seg == 0),
            jnp.logical_or(
                jnp.logical_and(q_seg == 1, jnp.logical_or(k_seg == 0, k_seg == 1)),
                jnp.logical_or(
                    jnp.logical_and(q_seg == 2, jnp.logical_or(k_seg == 0, k_seg == 2)),
                    jnp.logical_and(q_seg == 3, k_seg <= 3),
                ),
            ),
        )
        valid = jnp.logical_and(input_mask[:, :, None], input_mask[:, None, :])
        attn_mask = jnp.logical_and(allowed[None, :, :], valid)
        latent_start = obs_len + meta_len + ref_len
        latent_end = latent_start + latent_len
        return _ConditionPrefix(
            tokens=tokens,
            input_mask=input_mask,
            attn_mask=attn_mask,
            latent_start=latent_start,
            latent_end=latent_end,
        )

    def _encode_condition_latents(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b latent emb"]:
        condition_prefix = self._build_condition_prefix(obs)
        condition_positions = jnp.cumsum(condition_prefix.input_mask, axis=1) - 1
        condition_outputs, _ = self.PaliGemma.llm(
            [condition_prefix.tokens, None],
            mask=condition_prefix.attn_mask,
            positions=condition_positions,
        )
        condition_out = condition_outputs[0] if isinstance(condition_outputs, tuple | list) else condition_outputs
        assert condition_out is not None
        return condition_out[:, condition_prefix.latent_start : condition_prefix.latent_end]

    def _build_execution_prefix(
        self,
        obs: _model.Observation,
        latent_tokens: at.Float[at.Array, "b latent emb"],
    ) -> _BetaPrefix:
        obs_tokens, obs_mask = self._build_observation_tokens(obs)
        latent_mask = jnp.ones(latent_tokens.shape[:2], dtype=jnp.bool_)
        special_tokens = self._build_beta_special_tokens(obs)
        special_mask = jnp.ones((obs.state.shape[0], self.num_meta_special_tokens), dtype=jnp.bool_)
        execution_meta_tokens, execution_meta_mask = self._build_execution_meta_context_tokens(obs)

        tokens = jnp.concatenate([obs_tokens, latent_tokens, special_tokens, execution_meta_tokens], axis=1)
        input_mask = jnp.concatenate([obs_mask, latent_mask, special_mask, execution_meta_mask], axis=1)

        obs_len = obs_tokens.shape[1]
        latent_len = latent_tokens.shape[1]
        special_len = special_tokens.shape[1]
        execution_meta_len = execution_meta_tokens.shape[1]
        segments = jnp.concatenate(
            [
                jnp.zeros((obs_len,), dtype=jnp.int32),
                jnp.full((latent_len,), 3, dtype=jnp.int32),
                jnp.full((special_len,), 4, dtype=jnp.int32),
                jnp.full((execution_meta_len,), 5, dtype=jnp.int32),
            ],
            axis=0,
        )
        q_seg = segments[:, None]
        k_seg = segments[None, :]
        allowed = jnp.logical_or(
            jnp.logical_and(q_seg == 0, k_seg == 0),
            jnp.logical_or(
                jnp.logical_and(q_seg == 3, k_seg == 3),
                jnp.logical_or(
                    jnp.logical_and(q_seg == 4, jnp.logical_or(k_seg == 0, jnp.logical_or(k_seg == 3, k_seg == 4))),
                    jnp.logical_and(q_seg == 5, jnp.logical_or(k_seg == 0, k_seg == 5)),
                ),
            ),
        )
        valid = jnp.logical_and(input_mask[:, :, None], input_mask[:, None, :])
        attn_mask = jnp.logical_and(allowed[None, :, :], valid)
        action_visible_segments = jnp.logical_or(segments == 0, jnp.logical_or(segments == 3, segments == 4))
        action_visible_mask = jnp.logical_and(input_mask, action_visible_segments[None, :])
        return _BetaPrefix(tokens=tokens, input_mask=input_mask, attn_mask=attn_mask, action_visible_mask=action_visible_mask)

    def _decode_meta_actions(
        self,
        prefix_out: at.Float[at.Array, "b p emb"],
        suffix_out: at.Float[at.Array, "b s emb"],
    ) -> at.Float[at.Array, "b ah m md"]:
        execution_meta_tokens = prefix_out[:, -self.max_meta_areas :]
        special_end = -self.max_meta_areas
        special_start = special_end - self.num_meta_special_tokens
        special_tokens = self.meta_special_decode_proj(prefix_out[:, special_start:special_end])
        latent_start = special_start - self.num_meta_latent_tokens
        latent_tokens = self.latent_decode_proj(prefix_out[:, latent_start:special_start])
        special_tokens = jnp.concatenate([latent_tokens, special_tokens], axis=1)
        meta_tokens = self.meta_context_decode_proj(execution_meta_tokens)
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

        condition_meta_masks = (
            jnp.zeros((observation.state.shape[0], self.max_meta_areas), dtype=jnp.bool_)
            if observation.meta_area_masks is None
            else observation.meta_area_masks
        )
        meta_supervision_masks = (
            condition_meta_masks
            if observation.execution_meta_area_masks is None
            else observation.execution_meta_area_masks
        )
        dropped_meta_masks = self._apply_meta_dropout(meta_dropout_rng, condition_meta_masks, train=train)
        observation_for_meta = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            condition_images=observation.condition_images,
            condition_image_masks=observation.condition_image_masks,
            condition_state=observation.condition_state,
            condition_tokenized_prompt=observation.condition_tokenized_prompt,
            condition_tokenized_prompt_mask=observation.condition_tokenized_prompt_mask,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
            meta_area_poses=observation.meta_area_poses,
            meta_area_dim_masks=observation.meta_area_dim_masks,
            meta_area_types=observation.meta_area_types,
            meta_area_masks=dropped_meta_masks,
            execution_meta_area_poses=observation.execution_meta_area_poses,
            execution_meta_area_dim_masks=observation.execution_meta_area_dim_masks,
            execution_meta_area_types=observation.execution_meta_area_types,
            execution_meta_area_masks=observation.execution_meta_area_masks,
            meta_action_target_poses=observation.meta_action_target_poses,
            meta_action_target_dim_masks=observation.meta_action_target_dim_masks,
            meta_action_target_masks=observation.meta_action_target_masks,
            meta_control_alpha=observation.meta_control_alpha,
            reference_actions=observation.reference_actions,
            reference_action_mask=observation.reference_action_mask,
            meta_imagination_alpha=observation.meta_imagination_alpha,
        )
        condition_latents = self._encode_condition_latents(observation_for_meta)
        beta_prefix = self._build_execution_prefix(observation_for_meta, condition_latents)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation_for_meta, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(beta_prefix.action_visible_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        suffix_full_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        input_mask = jnp.concatenate([beta_prefix.input_mask, suffix_mask], axis=1)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        prefix_rows = jnp.concatenate(
            [
                beta_prefix.attn_mask,
                jnp.zeros(
                    (beta_prefix.attn_mask.shape[0], beta_prefix.attn_mask.shape[1], suffix_tokens.shape[1]),
                    dtype=jnp.bool_,
                ),
            ],
            axis=-1,
        )
        full_attn_mask = jnp.concatenate([prefix_rows, suffix_full_mask], axis=1)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [beta_prefix.tokens, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        action_out = suffix_out[:, -self.action_horizon :]
        v_t = self._mask_backbone_channels(self.action_out_proj(action_out))
        backbone_action_mask = jnp.asarray(self.backbone_action_mask_values, dtype=v_t.dtype)
        squared_error = jnp.square(v_t - u_t) * backbone_action_mask[None, None, :]
        base_loss = jnp.sum(squared_error, axis=-1) / jnp.maximum(jnp.sum(backbone_action_mask), 1.0)

        prefix_for_meta = jax.lax.stop_gradient(prefix_out) if self.meta_stop_backbone_grad else prefix_out
        suffix_for_meta = jax.lax.stop_gradient(suffix_out) if self.meta_stop_backbone_grad else suffix_out
        meta_pred = self._decode_meta_actions(prefix_for_meta, suffix_for_meta)
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
            meta_target = jnp.zeros_like(meta_pred).at[:, :horizon, :slots, :].set(target_poses[:, :horizon, :slots, :])
            if observation.meta_action_target_masks is None:
                target_masks = jnp.broadcast_to(meta_supervision_masks[:, None, :], meta_pred.shape[:3])
            else:
                target_masks = observation.meta_action_target_masks
                if target_masks.ndim == 2:
                    target_masks = target_masks[:, None, :]
                if target_masks.shape[1] == 1 and meta_pred.shape[1] > 1:
                    target_masks = jnp.broadcast_to(target_masks, meta_pred.shape[:3])
            meta_loss_mask = jnp.zeros(meta_pred.shape[:3], dtype=jnp.bool_).at[:, :horizon, :slots].set(
                target_masks[:, :horizon, :slots]
            )
            if observation.meta_action_target_dim_masks is not None:
                target_dim_masks = observation.meta_action_target_dim_masks[..., : self.meta_action_dim]
                if target_dim_masks.ndim == 3:
                    target_dim_masks = target_dim_masks[:, None, :, :]
                if target_dim_masks.shape[1] == 1 and meta_pred.shape[1] > 1:
                    target_dim_masks = jnp.broadcast_to(target_dim_masks, meta_pred.shape)
            elif observation.meta_area_dim_masks is not None:
                target_dim_masks = jnp.broadcast_to(
                    observation.meta_area_dim_masks[:, None, :, : self.meta_action_dim],
                    meta_pred.shape,
                )
            else:
                target_dim_masks = jnp.ones(meta_pred.shape, dtype=jnp.bool_)
            meta_dim_weights = jnp.zeros_like(meta_pred, dtype=meta_pred.dtype).at[:, :horizon, :slots, :].set(
                target_dim_masks[:, :horizon, :slots, :].astype(meta_pred.dtype)
            )
        meta_dim_denom = jnp.maximum(jnp.sum(meta_dim_weights, axis=-1), 1.0)
        meta_loss = jnp.sum(jnp.square(meta_pred - meta_target) * meta_dim_weights, axis=-1) / meta_dim_denom
        denom = jnp.maximum(jnp.sum(meta_loss_mask, axis=-1), 1)
        meta_loss = jnp.sum(meta_loss * meta_loss_mask, axis=-1) / denom
        return self.action_loss_weight * base_loss + self.meta_loss_weight * meta_loss

    @override
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
        condition_latents = self._encode_condition_latents(observation)
        beta_prefix = self._build_execution_prefix(observation, condition_latents)
        prefix_positions = jnp.cumsum(beta_prefix.input_mask, axis=1) - 1
        prefix_outputs, kv_cache = self.PaliGemma.llm(
            [beta_prefix.tokens, None],
            mask=beta_prefix.attn_mask,
            positions=prefix_positions,
        )
        prefix_out = prefix_outputs[0] if isinstance(prefix_outputs, tuple | list) else prefix_outputs
        assert prefix_out is not None

        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        noise = self._mask_backbone_channels(noise)
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(beta_prefix.action_visible_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(beta_prefix.input_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
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
        prefix_to_suffix_mask = einops.repeat(beta_prefix.action_visible_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(beta_prefix.input_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        meta_actions = self._decode_meta_actions(prefix_out, suffix_out)
        return {"actions": self._mask_backbone_channels(x_0), "meta_actions": meta_actions}
