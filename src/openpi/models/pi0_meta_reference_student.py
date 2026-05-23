import dataclasses

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_meta import Pi0Meta
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class _ExternalMetaPrefix:
    tokens: at.Float[at.Array, "b p emb"]
    input_mask: at.Bool[at.Array, "b p"]
    attn_mask: at.Bool[at.Array, "b p p"]


class Pi0MetaReferenceStudent(Pi0Meta):
    """Reference-action student that feeds generated meta tokens into frozen Pi0Meta.

    The inherited modules form the old structured execution model. The new
    ``reference_*`` modules encode chunk1/reference observation/action plus the
    current chunk2 qpos anchor into tokens that replace the old 12D meta-area MLP
    output. Training should freeze every non-``reference_*`` parameter.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        if config.action_horizon % config.reference_action_group_size != 0:
            raise ValueError(
                "Pi0MetaReferenceStudent requires action_horizon to be divisible by "
                "reference_action_group_size; "
                f"got horizon={config.action_horizon}, group={config.reference_action_group_size}"
            )
        self.reference_action_group_size = int(config.reference_action_group_size)
        self.reference_action_dim = int(config.reference_action_dim)
        self.reference_current_state_dim = int(config.reference_current_state_dim)
        self.meta_contrastive_loss_weight = float(config.meta_contrastive_loss_weight)
        if self.reference_action_dim <= 0 or self.reference_action_dim > config.action_dim:
            raise ValueError(
                "Pi0MetaReferenceStudent requires 0 < reference_action_dim <= action_dim; "
                f"got reference_action_dim={self.reference_action_dim}, action_dim={config.action_dim}"
            )
        if self.reference_current_state_dim <= 0 or self.reference_current_state_dim > config.action_dim:
            raise ValueError(
                "Pi0MetaReferenceStudent requires 0 < reference_current_state_dim <= action_dim; "
                f"got reference_current_state_dim={self.reference_current_state_dim}, action_dim={config.action_dim}"
            )
        self.num_reference_action_tokens = config.action_horizon // self.reference_action_group_size

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        reference_llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        reference_llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        reference_img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        reference_img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.reference_PaliGemma = nnx.Dict(llm=reference_llm, img=reference_img)

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
        self.reference_state_in = nnx.Linear(config.action_dim, width, rngs=rngs)
        self.reference_state_out = nnx.Linear(width, width, rngs=rngs)
        self.reference_current_state_in = nnx.Linear(self.reference_current_state_dim, width, rngs=rngs)
        self.reference_current_state_out = nnx.Linear(width, width, rngs=rngs)
        self.reference_meta_query_embedding = nnx.Embed(
            num_embeddings=self.max_meta_areas,
            features=width,
            rngs=rngs,
        )
        self.reference_query_norm = nnx.LayerNorm(width, rngs=rngs)

    def _build_reference_observation_tokens(
        self,
        obs: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b o emb"], at.Bool[at.Array, "b o"]]:
        if obs.condition_images is None or obs.condition_image_masks is None:
            raise ValueError("Reference-student encoding requires condition images and masks.")
        tokens = []
        input_mask = []
        for name in obs.condition_images:
            image_tokens, _ = self.reference_PaliGemma.img(obs.condition_images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.condition_image_masks[name], "b -> b s", s=image_tokens.shape[1]))

        if obs.condition_tokenized_prompt is not None:
            tokenized_inputs = self.reference_PaliGemma.llm(obs.condition_tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.condition_tokenized_prompt_mask)

        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1)

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

    def _build_reference_state_token(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b one emb"], at.Bool[at.Array, "b one"]]:
        if observation.condition_state is None:
            raise ValueError("Reference-student encoding requires 32D condition_state.")
        condition_state = self._mask_backbone_channels(observation.condition_state)
        if condition_state.shape[-1] != self.action_dim:
            raise ValueError(
                "Reference-student condition_state must match the executor state/action dimension; "
                f"got {condition_state.shape[-1]}, expected {self.action_dim}."
            )
        token = self.reference_state_in(condition_state)
        token = nnx.swish(token)
        token = self.reference_state_out(token)[:, None, :]
        mask = jnp.ones((observation.state.shape[0], 1), dtype=jnp.bool_)
        return token, mask

    def _build_current_state_token(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b one emb"], at.Bool[at.Array, "b one"]]:
        state14 = observation.state[..., : self.reference_current_state_dim]
        token = self.reference_current_state_in(state14)
        token = nnx.swish(token)
        token = self.reference_current_state_out(token)[:, None, :]
        mask = jnp.ones((observation.state.shape[0], 1), dtype=jnp.bool_)
        return token, mask

    def _encode_reference_meta_tokens(self, observation: _model.Observation) -> at.Float[at.Array, "b m emb"]:
        obs_tokens, obs_mask = self._build_reference_observation_tokens(observation)
        reference_state_token, reference_state_mask = self._build_reference_state_token(observation)
        ref_tokens, ref_mask = self._build_reference_action_tokens(observation)
        current_state_token, current_state_mask = self._build_current_state_token(observation)
        query_ids = jnp.arange(self.max_meta_areas, dtype=jnp.int32)
        query_tokens = self.reference_meta_query_embedding(query_ids)[None, :, :]
        query_tokens = jnp.broadcast_to(
            query_tokens,
            (observation.state.shape[0], self.max_meta_areas, query_tokens.shape[-1]),
        )
        query_mask = jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_)

        tokens = jnp.concatenate(
            [obs_tokens, reference_state_token, ref_tokens, current_state_token, query_tokens],
            axis=1,
        )
        input_mask = jnp.concatenate(
            [obs_mask, reference_state_mask, ref_mask, current_state_mask, query_mask],
            axis=1,
        )
        attn_mask = jnp.logical_and(input_mask[:, :, None], input_mask[:, None, :])
        positions = jnp.cumsum(input_mask, axis=1) - 1
        outputs, _ = self.reference_PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
        out = outputs[0] if isinstance(outputs, tuple | list) else outputs
        assert out is not None
        return self.reference_query_norm(out[:, -self.max_meta_areas :])

    def _condition_meta_tokens(self, observation: _model.Observation) -> at.Float[at.Array, "b m emb"]:
        if observation.condition_images is None:
            old_meta_tokens, _ = self._build_meta_context_tokens(observation)
            return old_meta_tokens
        # Reference-action and obs-only paths both use the query output from the
        # reference encoder. For obs-only samples the reference-action tokens are
        # present but fully masked inside _build_reference_action_tokens.
        return self._encode_reference_meta_tokens(observation)

    def _contrastive_meta_token_loss(
        self,
        observation: _model.Observation,
        student_tokens: at.Float[at.Array, "b m emb"],
    ) -> at.Array:
        if self.meta_contrastive_loss_weight <= 0.0:
            return jnp.zeros((observation.state.shape[0],), dtype=observation.state.dtype)
        if observation.execution_meta_area_poses is not None:
            teacher_obs = dataclasses.replace(
                observation,
                meta_area_poses=observation.execution_meta_area_poses,
                meta_area_dim_masks=observation.execution_meta_area_dim_masks,
                meta_area_types=observation.execution_meta_area_types,
                meta_area_masks=observation.execution_meta_area_masks,
            )
        else:
            teacher_obs = observation
        teacher_tokens, teacher_masks = self._build_meta_context_tokens(teacher_obs)
        teacher_tokens = jax.lax.stop_gradient(teacher_tokens)
        student_norm = student_tokens / jnp.maximum(jnp.linalg.norm(student_tokens, axis=-1, keepdims=True), 1e-6)
        teacher_norm = teacher_tokens / jnp.maximum(jnp.linalg.norm(teacher_tokens, axis=-1, keepdims=True), 1e-6)
        cosine_loss = 1.0 - jnp.sum(student_norm * teacher_norm, axis=-1)
        valid = teacher_masks.astype(jnp.bool_)
        loss = jnp.sum(jnp.where(valid, cosine_loss, 0.0), axis=-1) / jnp.maximum(jnp.sum(valid, axis=-1), 1)
        return loss

    def _build_external_meta_prefix(
        self,
        obs: _model.Observation,
        meta_tokens: at.Float[at.Array, "b m emb"],
    ) -> _ExternalMetaPrefix:
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
        tokens.append(meta_tokens)
        input_mask.append(jnp.ones(meta_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * meta_tokens.shape[1]
        special_tokens = self._build_special_tokens(obs)
        tokens.append(special_tokens)
        input_mask.append(jnp.ones((obs.state.shape[0], self.num_meta_special_tokens), dtype=jnp.bool_))
        ar_mask += [False] * self.num_meta_special_tokens
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        attn_mask = make_attn_mask(input_mask, jnp.asarray(ar_mask))
        return _ExternalMetaPrefix(tokens=tokens, input_mask=input_mask, attn_mask=attn_mask)

    def _meta_loss(
        self,
        observation: _model.Observation,
        meta_pred: at.Float[at.Array, "b ah m md"],
    ) -> at.Float[at.Array, "b ah"]:
        if observation.execution_meta_area_masks is None:
            meta_supervision_masks = (
                jnp.zeros((observation.state.shape[0], self.max_meta_areas), dtype=jnp.bool_)
                if observation.meta_area_masks is None
                else observation.meta_area_masks
            )
        else:
            meta_supervision_masks = observation.execution_meta_area_masks
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
            elif observation.execution_meta_area_dim_masks is not None:
                target_dim_masks = jnp.broadcast_to(
                    observation.execution_meta_area_dim_masks[:, None, :, : self.meta_action_dim],
                    meta_pred.shape,
                )
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
        return jnp.sum(meta_loss * meta_loss_mask, axis=-1) / denom

    def compute_loss_terms(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> dict[str, at.Array]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        observation = self._prepare_observation(observation)

        batch_shape = actions.shape[:-2]
        noise = self._mask_backbone_channels(jax.random.normal(noise_rng, actions.shape))
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        meta_tokens = self._condition_meta_tokens(observation)
        contrastive_loss = self._contrastive_meta_token_loss(observation, meta_tokens)
        prefix = self._build_external_meta_prefix(observation, meta_tokens)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix.input_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        suffix_full_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        input_mask = jnp.concatenate([prefix.input_mask, suffix_mask], axis=1)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        prefix_rows = jnp.concatenate(
            [
                prefix.attn_mask,
                jnp.zeros((prefix.attn_mask.shape[0], prefix.attn_mask.shape[1], suffix_tokens.shape[1]), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        full_attn_mask = jnp.concatenate([prefix_rows, suffix_full_mask], axis=1)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix.tokens, suffix_tokens],
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
        meta_loss = self._meta_loss(observation, meta_pred)
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
        meta_tokens = self._condition_meta_tokens(observation)
        prefix = self._build_external_meta_prefix(observation, meta_tokens)
        prefix_positions = jnp.cumsum(prefix.input_mask, axis=1) - 1
        prefix_outputs, kv_cache = self.PaliGemma.llm(
            [prefix.tokens, None],
            mask=prefix.attn_mask,
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
            prefix_to_suffix_mask = einops.repeat(prefix.input_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix.input_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
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
        prefix_to_suffix_mask = einops.repeat(prefix.input_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix.input_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
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
