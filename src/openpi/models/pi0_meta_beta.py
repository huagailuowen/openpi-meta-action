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
    meta_start: int
    meta_end: int


class Pi0MetaBeta(Pi0Meta):
    """Beta structured-meta model with chunk1-refined meta tokens.

    Raw meta-area and reference-action condition tokens are prefix-only inputs.
    The condition encoder returns refined meta tokens directly; beta2 does not
    use a separate latent-token bottleneck.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        if config.action_horizon % config.reference_action_group_size != 0:
            raise ValueError(
                "Pi0MetaBeta requires action_horizon to be divisible by reference_action_group_size; "
                f"got horizon={config.action_horizon}, group={config.reference_action_group_size}"
            )
        self.reference_action_group_size = int(config.reference_action_group_size)
        self.reference_action_dim = int(config.reference_action_dim)
        self.meta_contrastive_loss_weight = float(config.meta_contrastive_loss_weight)
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
                "Pi0MetaBeta requires chunk1 condition observation fields for condition meta-token encoding; "
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

    def _condition_observation_for_contrastive(
        self, observation: _model.Observation, *, mode: str
    ) -> _model.Observation:
        batch_size = observation.state.shape[0]
        if mode == "meta":
            if observation.contrastive_meta_area_masks is None:
                meta_masks = jnp.zeros((batch_size, self.max_meta_areas), dtype=jnp.bool_)
            else:
                meta_masks = observation.contrastive_meta_area_masks
            ref_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
            return dataclasses.replace(
                observation,
                meta_area_poses=observation.contrastive_meta_area_poses,
                meta_area_dim_masks=observation.contrastive_meta_area_dim_masks,
                meta_area_types=observation.contrastive_meta_area_types,
                meta_area_masks=meta_masks,
                reference_actions=None,
                reference_action_mask=ref_mask,
            )
        if mode == "reference":
            meta_masks = (
                jnp.zeros((batch_size, self.max_meta_areas), dtype=jnp.bool_)
                if observation.contrastive_meta_area_masks is None
                else jnp.zeros_like(observation.contrastive_meta_area_masks)
            )
            ref_mask = (
                jnp.zeros((batch_size,), dtype=jnp.bool_)
                if observation.contrastive_reference_action_mask is None
                else observation.contrastive_reference_action_mask.astype(jnp.bool_)
            )
            return dataclasses.replace(
                observation,
                meta_area_poses=observation.contrastive_meta_area_poses,
                meta_area_dim_masks=observation.contrastive_meta_area_dim_masks,
                meta_area_types=observation.contrastive_meta_area_types,
                meta_area_masks=meta_masks,
                reference_actions=observation.contrastive_reference_actions,
                reference_action_mask=ref_mask,
            )
        raise ValueError(f"Unknown contrastive condition mode: {mode}")

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
        meta_tokens, _ = self._build_meta_context_tokens(obs)
        # In beta2 the meta slots themselves are the cross-chunk representation.
        # They must stay active even for reference-action or obs-only conditions,
        # where _build_meta_context_tokens returns learnable default/query tokens.
        meta_mask = jnp.ones(meta_tokens.shape[:2], dtype=jnp.bool_)
        ref_tokens, ref_mask = self._build_reference_action_tokens(obs)

        tokens = jnp.concatenate([obs_tokens, meta_tokens, ref_tokens], axis=1)
        input_mask = jnp.concatenate([obs_mask, meta_mask, ref_mask], axis=1)

        obs_len = obs_tokens.shape[1]
        meta_len = meta_tokens.shape[1]
        ref_len = ref_tokens.shape[1]
        segments = jnp.concatenate(
            [
                jnp.zeros((obs_len,), dtype=jnp.int32),
                jnp.ones((meta_len,), dtype=jnp.int32),
                jnp.full((ref_len,), 2, dtype=jnp.int32),
            ],
            axis=0,
        )
        q_seg = segments[:, None]
        k_seg = segments[None, :]
        allowed = jnp.logical_or(
            jnp.logical_and(q_seg == 0, k_seg == 0),
            jnp.logical_or(
                # Refined meta/query tokens may read condition observation,
                # explicit meta tokens, and reference-action tokens.
                jnp.logical_and(q_seg == 1, k_seg <= 2),
                jnp.logical_and(q_seg == 2, jnp.logical_or(k_seg == 0, k_seg == 2)),
            ),
        )
        valid = jnp.logical_and(input_mask[:, :, None], input_mask[:, None, :])
        attn_mask = jnp.logical_and(allowed[None, :, :], valid)
        meta_start = obs_len
        meta_end = meta_start + meta_len
        return _ConditionPrefix(
            tokens=tokens,
            input_mask=input_mask,
            attn_mask=attn_mask,
            meta_start=meta_start,
            meta_end=meta_end,
        )

    def _encode_condition_meta_tokens(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b m emb"]:
        condition_prefix = self._build_condition_prefix(obs)
        condition_positions = jnp.cumsum(condition_prefix.input_mask, axis=1) - 1
        condition_outputs, _ = self.PaliGemma.llm(
            [condition_prefix.tokens, None],
            mask=condition_prefix.attn_mask,
            positions=condition_positions,
        )
        condition_out = condition_outputs[0] if isinstance(condition_outputs, tuple | list) else condition_outputs
        assert condition_out is not None
        return condition_out[:, condition_prefix.meta_start : condition_prefix.meta_end]

    def _encode_condition_meta_token_layers(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "l b m emb"]:
        condition_prefix = self._build_condition_prefix(obs)
        condition_positions = jnp.cumsum(condition_prefix.input_mask, axis=1) - 1
        _, _, layer_outputs = self.PaliGemma.llm(
            [condition_prefix.tokens, None],
            mask=condition_prefix.attn_mask,
            positions=condition_positions,
            method="call_with_intermediates",
        )
        layer_out = layer_outputs[0] if isinstance(layer_outputs, tuple | list) else layer_outputs
        assert layer_out is not None
        return layer_out[:, :, condition_prefix.meta_start : condition_prefix.meta_end]

    def _contrastive_layer_weights(self, num_layers: int, dtype: jnp.dtype) -> at.Float[at.Array, " l"]:
        layer_positions = jnp.arange(1, num_layers + 1, dtype=dtype) / jnp.asarray(num_layers, dtype=dtype)
        weights = jnp.square(layer_positions)
        return weights / jnp.maximum(jnp.mean(weights), 1e-6)

    def _contrastive_meta_token_loss(self, observation: _model.Observation) -> at.Array:
        if self.meta_contrastive_loss_weight <= 0.0:
            return jnp.zeros((observation.state.shape[0],), dtype=observation.state.dtype)
        if (
            observation.contrastive_meta_area_masks is None
            or observation.contrastive_reference_actions is None
            or observation.contrastive_reference_action_mask is None
        ):
            return jnp.zeros((observation.state.shape[0],), dtype=observation.state.dtype)

        meta_obs = self._condition_observation_for_contrastive(observation, mode="meta")
        ref_obs = self._condition_observation_for_contrastive(observation, mode="reference")
        meta_tokens = self._encode_condition_meta_token_layers(meta_obs)
        ref_tokens = self._encode_condition_meta_token_layers(ref_obs)
        meta_tokens = meta_tokens / jnp.maximum(jnp.linalg.norm(meta_tokens, axis=-1, keepdims=True), 1e-6)
        ref_tokens = ref_tokens / jnp.maximum(jnp.linalg.norm(ref_tokens, axis=-1, keepdims=True), 1e-6)
        diff_loss = jnp.sum(jnp.square(meta_tokens - ref_tokens), axis=-1)
        meta_valid = observation.contrastive_meta_area_masks.astype(jnp.bool_)
        ref_valid = observation.contrastive_reference_action_mask.astype(jnp.bool_)
        valid = jnp.logical_and(meta_valid, ref_valid[:, None])
        loss = jnp.where(valid[None, :, :], diff_loss, 0.0)
        denom = jnp.maximum(jnp.sum(valid, axis=-1), 1)
        per_layer_loss = jnp.sum(loss, axis=-1) / denom[None, :]
        weights = self._contrastive_layer_weights(meta_tokens.shape[0], meta_tokens.dtype)
        return jnp.mean(per_layer_loss * weights[:, None], axis=0)

    def _build_execution_prefix(
        self,
        obs: _model.Observation,
        condition_meta_tokens: at.Float[at.Array, "b m emb"],
    ) -> _BetaPrefix:
        obs_tokens, obs_mask = self._build_observation_tokens(obs)
        meta_mask = jnp.ones(condition_meta_tokens.shape[:2], dtype=jnp.bool_)
        special_tokens = self._build_beta_special_tokens(obs)
        special_mask = jnp.ones((obs.state.shape[0], self.num_meta_special_tokens), dtype=jnp.bool_)

        tokens = jnp.concatenate([obs_tokens, condition_meta_tokens, special_tokens], axis=1)
        input_mask = jnp.concatenate([obs_mask, meta_mask, special_mask], axis=1)

        obs_len = obs_tokens.shape[1]
        meta_len = condition_meta_tokens.shape[1]
        special_len = special_tokens.shape[1]
        segments = jnp.concatenate(
            [
                jnp.zeros((obs_len,), dtype=jnp.int32),
                jnp.ones((meta_len,), dtype=jnp.int32),
                jnp.full((special_len,), 2, dtype=jnp.int32),
            ],
            axis=0,
        )
        q_seg = segments[:, None]
        k_seg = segments[None, :]
        allowed = jnp.logical_or(
            jnp.logical_and(q_seg == 0, k_seg == 0),
            jnp.logical_or(
                # Keep chunk1 meta tokens as the tool representation; chunk2
                # special tokens integrate current observation plus those tokens.
                jnp.logical_and(q_seg == 1, k_seg == 1),
                jnp.logical_and(q_seg == 2, k_seg <= 2),
            ),
        )
        valid = jnp.logical_and(input_mask[:, :, None], input_mask[:, None, :])
        attn_mask = jnp.logical_and(allowed[None, :, :], valid)
        action_visible_segments = segments <= 2
        action_visible_mask = jnp.logical_and(input_mask, action_visible_segments[None, :])
        return _BetaPrefix(tokens=tokens, input_mask=input_mask, attn_mask=attn_mask, action_visible_mask=action_visible_mask)

    def _decode_meta_actions(
        self,
        prefix_out: at.Float[at.Array, "b p emb"],
        suffix_out: at.Float[at.Array, "b s emb"],
    ) -> at.Float[at.Array, "b ah m md"]:
        special_start = prefix_out.shape[1] - self.num_meta_special_tokens
        meta_start = special_start - self.max_meta_areas
        special_tokens = self.meta_special_decode_proj(prefix_out[:, special_start:])
        meta_tokens = self.meta_context_decode_proj(prefix_out[:, meta_start:special_start])
        action_tokens = suffix_out[:, -self.action_horizon :]
        return self.meta_head(action_tokens, meta_tokens, special_tokens)

    def compute_loss_terms(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> dict[str, at.Array]:
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
            contrastive_meta_area_poses=observation.contrastive_meta_area_poses,
            contrastive_meta_area_dim_masks=observation.contrastive_meta_area_dim_masks,
            contrastive_meta_area_types=observation.contrastive_meta_area_types,
            contrastive_meta_area_masks=observation.contrastive_meta_area_masks,
            contrastive_reference_actions=observation.contrastive_reference_actions,
            contrastive_reference_action_mask=observation.contrastive_reference_action_mask,
            meta_imagination_alpha=observation.meta_imagination_alpha,
        )
        contrastive_loss = self._contrastive_meta_token_loss(observation_for_meta)
        condition_meta_tokens = self._encode_condition_meta_tokens(observation_for_meta)
        beta_prefix = self._build_execution_prefix(observation_for_meta, condition_meta_tokens)
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
        action_component = self.action_loss_weight * base_loss
        meta_component = self.meta_loss_weight * meta_loss
        contrastive_component = self.meta_contrastive_loss_weight * contrastive_loss[:, None]
        return {
            "loss": action_component + meta_component + contrastive_component,
            "action_loss": action_component,
            "meta_loss": meta_component,
            "contrastive_loss": contrastive_component,
        }

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        return self.compute_loss_terms(rng, observation, actions, train=train)["loss"]

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
        condition_meta_tokens = self._encode_condition_meta_tokens(observation)
        beta_prefix = self._build_execution_prefix(observation, condition_meta_tokens)
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
