import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi05_meta_model_builds():
    config = _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    assert model.max_meta_areas == 3


def test_pi05_meta_inputs_spec_contains_meta_areas():
    config = _pi0_config.Pi0Config(pi05=True, meta_model=True, paligemma_variant="dummy", action_expert_variant="dummy")
    observation_spec, _ = config.inputs_spec()
    assert observation_spec.meta_area_poses is not None
    assert observation_spec.meta_area_types is not None
    assert observation_spec.meta_area_masks is not None


def test_pi05_meta_12d_inputs_spec_contains_dim_masks():
    config = _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        meta_actions_in_action_slice=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    observation_spec, _ = config.inputs_spec()
    assert observation_spec.meta_area_poses is not None
    assert observation_spec.meta_area_poses.shape[-1] == 12
    assert observation_spec.meta_area_dim_masks is not None
    assert observation_spec.meta_area_dim_masks.shape[-1] == 12
    assert observation_spec.meta_action_target_poses is not None
    assert observation_spec.meta_action_target_poses.shape[-1] == 12
    model = nnx.eval_shape(config.create, jax.random.key(0))
    mask = model.backbone_action_mask_values
    assert mask[:14] == (1.0,) * 14
    assert mask[14:20] == (0.0,) * 6
    assert mask[20:26] == (1.0,) * 6
    assert mask[26:32] == (0.0,) * 6


def test_pi05_meta_beta_inputs_spec_contains_execution_meta_and_reference():
    config = _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_beta_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    observation_spec, _ = config.inputs_spec()
    assert observation_spec.condition_images is not None
    assert observation_spec.condition_image_masks is not None
    assert observation_spec.condition_state is not None
    assert observation_spec.condition_tokenized_prompt is not None
    assert observation_spec.condition_tokenized_prompt_mask is not None
    assert observation_spec.execution_meta_area_poses is not None
    assert observation_spec.execution_meta_area_poses.shape[-1] == 12
    assert observation_spec.execution_meta_area_dim_masks is not None
    assert observation_spec.reference_actions is not None
    assert observation_spec.reference_actions.shape[1:] == (50, 14)
    assert observation_spec.contrastive_meta_area_poses is not None
    assert observation_spec.contrastive_meta_area_poses.shape[-1] == 12
    assert observation_spec.contrastive_reference_actions is not None
    assert observation_spec.contrastive_reference_actions.shape[1:] == (50, 14)
    assert observation_spec.meta_imagination_alpha is not None


def test_preprocess_observation_preserves_beta_fields():
    obs = _model.Observation(
        images={
            "base_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        },
        state=jnp.zeros((1, 32), dtype=jnp.float32),
        condition_images={
            "base_0_rgb": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
        },
        condition_image_masks={
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        },
        condition_state=jnp.ones((1, 32), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, 8), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, 8), dtype=jnp.bool_),
        tokenized_prompt=jnp.zeros((1, 8), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 8), dtype=jnp.bool_),
        meta_area_poses=jnp.zeros((1, 1, 12), dtype=jnp.float32),
        meta_area_dim_masks=jnp.ones((1, 1, 12), dtype=jnp.bool_),
        meta_area_types=jnp.zeros((1, 1), dtype=jnp.int32),
        meta_area_masks=jnp.ones((1, 1), dtype=jnp.bool_),
        execution_meta_area_poses=jnp.ones((1, 1, 12), dtype=jnp.float32),
        execution_meta_area_dim_masks=jnp.ones((1, 1, 12), dtype=jnp.bool_),
        execution_meta_area_types=jnp.ones((1, 1), dtype=jnp.int32),
        execution_meta_area_masks=jnp.ones((1, 1), dtype=jnp.bool_),
        reference_actions=jnp.ones((1, 50, 14), dtype=jnp.float32),
        reference_action_mask=jnp.ones((1,), dtype=jnp.bool_),
        contrastive_meta_area_poses=jnp.ones((1, 1, 12), dtype=jnp.float32) * 2.0,
        contrastive_meta_area_dim_masks=jnp.ones((1, 1, 12), dtype=jnp.bool_),
        contrastive_meta_area_types=jnp.ones((1, 1), dtype=jnp.int32),
        contrastive_meta_area_masks=jnp.ones((1, 1), dtype=jnp.bool_),
        contrastive_reference_actions=jnp.ones((1, 50, 14), dtype=jnp.float32) * 3.0,
        contrastive_reference_action_mask=jnp.ones((1,), dtype=jnp.bool_),
        meta_imagination_alpha=jnp.asarray([0.5], dtype=jnp.float32),
    )
    out = _model.preprocess_observation(None, obs, train=False)

    assert out.reference_actions is not None
    assert out.reference_action_mask is not None
    assert out.meta_imagination_alpha is not None
    assert out.condition_images is not None
    assert out.condition_state is not None
    assert out.condition_tokenized_prompt is not None
    assert out.condition_tokenized_prompt_mask is not None
    assert out.execution_meta_area_poses is not None
    assert jnp.all(out.execution_meta_area_poses == 1.0)
    assert out.contrastive_meta_area_poses is not None
    assert jnp.all(out.contrastive_meta_area_poses == 2.0)
    assert out.contrastive_reference_actions is not None
    assert jnp.all(out.contrastive_reference_actions == 3.0)


def _make_dummy_beta_config() -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_beta_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        action_dim=32,
        action_horizon=10,
        max_token_len=8,
        max_meta_areas=1,
        num_meta_latent_tokens=2,
        reference_action_group_size=5,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )


def _make_dummy_beta_observation(
    config: _pi0_config.Pi0Config,
    *,
    condition_state: jax.Array | None = None,
    condition_tokenized_prompt: jax.Array | None = None,
    condition_tokenized_prompt_mask: jax.Array | None = None,
) -> _model.Observation:
    image = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    images = {
        "base_0_rgb": image,
        "left_wrist_0_rgb": image,
        "right_wrist_0_rgb": image,
    }
    masks = {key: jnp.ones((1,), dtype=jnp.bool_) for key in images}
    return _model.Observation(
        images=images,
        image_masks=masks,
        state=jnp.zeros((1, config.action_dim), dtype=jnp.float32),
        condition_images={key: jnp.ones_like(value) for key, value in images.items()},
        condition_image_masks=masks,
        condition_state=condition_state,
        condition_tokenized_prompt=condition_tokenized_prompt,
        condition_tokenized_prompt_mask=condition_tokenized_prompt_mask,
        tokenized_prompt=jnp.zeros((1, config.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
        meta_area_poses=jnp.zeros((1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.float32),
        meta_area_dim_masks=jnp.ones((1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.bool_),
        meta_area_types=jnp.zeros((1, config.max_meta_areas), dtype=jnp.int32),
        meta_area_masks=jnp.ones((1, config.max_meta_areas), dtype=jnp.bool_),
        execution_meta_area_poses=jnp.zeros((1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.float32),
        execution_meta_area_dim_masks=jnp.ones((1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.bool_),
        execution_meta_area_types=jnp.zeros((1, config.max_meta_areas), dtype=jnp.int32),
        execution_meta_area_masks=jnp.ones((1, config.max_meta_areas), dtype=jnp.bool_),
        reference_actions=jnp.zeros((1, config.action_horizon, config.reference_action_dim), dtype=jnp.float32),
        reference_action_mask=jnp.ones((1,), dtype=jnp.bool_),
        meta_imagination_alpha=jnp.zeros((1,), dtype=jnp.float32),
    )


def test_pi05_meta_beta_condition_prefix_requires_condition_state():
    config = _make_dummy_beta_config()
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(config, condition_state=None)
    with pytest.raises(ValueError, match="condition_state"):
        model._build_condition_prefix(obs)


def test_pi05_meta_beta_condition_prefix_requires_condition_tokenized_prompt():
    config = _make_dummy_beta_config()
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="condition_tokenized_prompt"):
        model._build_condition_prefix(obs)


def test_pi05_meta_beta_condition_prefix_uses_condition_pi05_obs_tokens():
    config = _make_dummy_beta_config()
    model = config.create(jax.random.key(0))
    condition_tokens = jnp.ones((1, config.max_token_len), dtype=jnp.int32)
    condition_token_mask = jnp.ones((1, config.max_token_len), dtype=jnp.bool_)
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=condition_tokens,
        condition_tokenized_prompt_mask=condition_token_mask,
    )
    obs_tokens, _ = model._build_observation_tokens(
        obs,
        images=obs.condition_images,
        image_masks=obs.condition_image_masks,
        tokenized_prompt=obs.condition_tokenized_prompt,
        tokenized_prompt_mask=obs.condition_tokenized_prompt_mask,
    )
    execution_tokens, _ = model._build_observation_tokens(obs)
    prefix = model._build_condition_prefix(obs)
    expected_latent_start = obs_tokens.shape[1] + config.max_meta_areas + model.num_reference_action_tokens
    assert prefix.latent_start == expected_latent_start
    assert not jnp.allclose(obs_tokens[:, -config.max_token_len :], execution_tokens[:, -config.max_token_len :])
