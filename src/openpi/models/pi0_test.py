import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
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


def test_pi05_raw14_action_loss_masks_padded_dims():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_loss_dim=14,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    assert model.action_loss_mask_values[:14] == (1.0,) * 14
    assert model.action_loss_mask_values[14:] == (0.0,) * 18


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
    assert observation_spec.condition_state.shape[-1] == 32
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


def test_pi05_meta_reference_student_inputs_spec_and_builds():
    config = _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_reference_student_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        max_meta_areas=3,
        reference_meta_output_slots=1,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    observation_spec, _ = config.inputs_spec()
    assert observation_spec.condition_images is not None
    assert observation_spec.condition_state is not None
    assert observation_spec.condition_state.shape[-1] == 32
    assert observation_spec.execution_meta_area_poses is not None
    assert observation_spec.reference_actions is not None
    assert observation_spec.reference_actions.shape[1:] == (50, 14)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    assert model.max_meta_areas == 3
    assert model.reference_meta_replace_slots == 1
    assert hasattr(model, "reference_PaliGemma")


def test_pi05_meta_reference_student_freezes_only_old_executor():
    config = _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_reference_student_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        max_meta_areas=3,
        reference_meta_output_slots=1,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    state = _get_frozen_state(config)
    assert len(state) > 0
    assert all("reference_" not in "/".join(str(part) for part in path) for path in state)


def _make_dummy_reference_student_config(
    *,
    max_meta_areas: int = 3,
    reference_meta_output_slots: int = 1,
    meta_reference_teacher_student_learning: bool = False,
) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_reference_student_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        action_dim=32,
        action_horizon=10,
        max_token_len=8,
        max_meta_areas=max_meta_areas,
        reference_meta_output_slots=reference_meta_output_slots,
        reference_action_group_size=5,
        reference_action_dim=14,
        reference_current_state_dim=14,
        meta_reference_teacher_student_learning=meta_reference_teacher_student_learning,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )


def test_pi05_meta_reference_student_compute_loss_terms_smoke():
    config = _make_dummy_reference_student_config()
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )
    actions = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)

    terms = model.compute_loss_terms(jax.random.key(1), obs, actions, train=False)

    assert set(terms) == {"loss", "action_loss", "meta_loss", "contrastive_loss"}
    assert terms["loss"].shape == (1, config.action_horizon)
    assert jnp.all(jnp.isfinite(terms["loss"]))


def test_pi05_meta_reference_student_teacher_student_loss_smoke():
    config = _make_dummy_reference_student_config(meta_reference_teacher_student_learning=True)
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )
    actions = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)

    terms = model.compute_loss_terms(jax.random.key(1), obs, actions, train=False)

    assert set(terms) == {"loss", "action_loss", "meta_loss", "contrastive_loss"}
    assert terms["loss"].shape == (1, config.action_horizon)
    assert jnp.all(jnp.isfinite(terms["loss"]))


def test_pi05_meta_reference_student_requires_tokenized_condition_prompt_state():
    config = _make_dummy_reference_student_config()
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=None,
        condition_tokenized_prompt_mask=None,
    )

    with pytest.raises(ValueError, match="tokenized prompt"):
        model._encode_reference_meta_tokens(obs)


def test_pi05_meta_reference_student_replaces_only_leading_meta_slot():
    config = _make_dummy_reference_student_config(max_meta_areas=3, reference_meta_output_slots=1)
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )

    old_meta_tokens, _ = model._build_meta_context_tokens(obs)
    encoded_meta_tokens = model._encode_reference_meta_tokens(obs)
    conditioned_meta_tokens = model._condition_meta_tokens(obs)

    assert encoded_meta_tokens.shape[1] == 1
    assert conditioned_meta_tokens.shape == old_meta_tokens.shape
    np.testing.assert_allclose(np.asarray(conditioned_meta_tokens[:, :1]), np.asarray(encoded_meta_tokens), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(conditioned_meta_tokens[:, 1:]), np.asarray(old_meta_tokens[:, 1:]), rtol=1e-5)


def test_pi05_meta_reference_student_clips_requested_replacement_slots():
    config = _make_dummy_reference_student_config(max_meta_areas=2, reference_meta_output_slots=5)
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )

    encoded_meta_tokens = model._encode_reference_meta_tokens(obs)
    conditioned_meta_tokens = model._condition_meta_tokens(obs)

    assert model.reference_meta_replace_slots == 2
    assert encoded_meta_tokens.shape[1] == 2
    assert conditioned_meta_tokens.shape[1] == 2
    np.testing.assert_allclose(np.asarray(conditioned_meta_tokens), np.asarray(encoded_meta_tokens), rtol=1e-5)


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


def test_pi05_meta_beta_prepare_observation_preserves_contrastive_fields():
    config = _make_dummy_beta_config()
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(config)
    obs = dataclasses.replace(
        obs,
        contrastive_meta_area_poses=jnp.ones(
            (1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.float32
        )
        * 2.0,
        contrastive_meta_area_dim_masks=jnp.ones(
            (1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.bool_
        ),
        contrastive_meta_area_types=jnp.ones((1, config.max_meta_areas), dtype=jnp.int32),
        contrastive_meta_area_masks=jnp.ones((1, config.max_meta_areas), dtype=jnp.bool_),
        contrastive_reference_actions=jnp.ones(
            (1, config.action_horizon, config.reference_action_dim), dtype=jnp.float32
        )
        * 3.0,
        contrastive_reference_action_mask=jnp.ones((1,), dtype=jnp.bool_),
    )

    prepared = model._prepare_observation(obs)

    assert prepared.contrastive_meta_area_poses is not None
    assert jnp.all(prepared.contrastive_meta_area_poses == 2.0)
    assert prepared.contrastive_meta_area_dim_masks is not None
    assert prepared.contrastive_meta_area_types is not None
    assert prepared.contrastive_meta_area_masks is not None
    assert prepared.contrastive_reference_actions is not None
    assert jnp.all(prepared.contrastive_reference_actions == 3.0)
    assert prepared.contrastive_reference_action_mask is not None


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
    assert prefix.meta_start == obs_tokens.shape[1]
    assert prefix.meta_end == obs_tokens.shape[1] + config.max_meta_areas
    assert not jnp.allclose(obs_tokens[:, -config.max_token_len :], execution_tokens[:, -config.max_token_len :])


def test_pi05_meta_beta_condition_tokens_ignore_num_meta_latent_tokens():
    config = _pi0_config.Pi0Config(
        pi05=True,
        meta_model=True,
        meta_beta_model=True,
        meta_area_pose_dim=12,
        meta_action_dim=12,
        action_dim=32,
        action_horizon=10,
        max_token_len=8,
        max_meta_areas=1,
        num_meta_latent_tokens=7,
        reference_action_group_size=5,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )

    meta_tokens = model._encode_condition_meta_tokens(obs)

    assert meta_tokens.shape[1] == config.max_meta_areas


def test_pi05_meta_beta_execution_prefix_uses_refined_condition_meta_tokens():
    config = _make_dummy_beta_config()
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )

    obs_tokens, _ = model._build_observation_tokens(obs)
    condition_meta_tokens = model._encode_condition_meta_tokens(obs)
    prefix = model._build_execution_prefix(obs, condition_meta_tokens)

    expected_len = obs_tokens.shape[1] + config.max_meta_areas + model.num_meta_special_tokens
    assert prefix.tokens.shape[1] == expected_len
    assert prefix.action_visible_mask.shape[1] == expected_len


def test_pi05_meta_beta_contrastive_layer_weights_are_quadratic():
    config = _make_dummy_beta_config()
    model = config.create(jax.random.key(0))

    weights = model._contrastive_layer_weights(4, jnp.float32)
    expected = jnp.square(jnp.arange(1, 5, dtype=jnp.float32) / 4.0)
    expected = expected / jnp.mean(expected)

    assert jnp.allclose(weights, expected)
    assert jnp.allclose(jnp.mean(weights), 1.0)


def test_pi05_meta_beta_layerwise_contrastive_loss_is_finite():
    config = dataclasses.replace(_make_dummy_beta_config(), meta_contrastive_loss_weight=0.4)
    model = config.create(jax.random.key(0))
    obs = _make_dummy_beta_observation(
        config,
        condition_state=jnp.ones((1, config.action_dim), dtype=jnp.float32),
        condition_tokenized_prompt=jnp.ones((1, config.max_token_len), dtype=jnp.int32),
        condition_tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=jnp.bool_),
    )
    obs = dataclasses.replace(
        obs,
        contrastive_meta_area_poses=jnp.ones(
            (1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.float32
        ),
        contrastive_meta_area_dim_masks=jnp.ones(
            (1, config.max_meta_areas, config.meta_area_pose_dim), dtype=jnp.bool_
        ),
        contrastive_meta_area_types=jnp.zeros((1, config.max_meta_areas), dtype=jnp.int32),
        contrastive_meta_area_masks=jnp.ones((1, config.max_meta_areas), dtype=jnp.bool_),
        contrastive_reference_actions=jnp.ones(
            (1, config.action_horizon, config.reference_action_dim), dtype=jnp.float32
        ),
        contrastive_reference_action_mask=jnp.ones((1,), dtype=jnp.bool_),
    )

    loss = model._contrastive_meta_token_loss(obs)

    assert loss.shape == (1,)
    assert jnp.all(jnp.isfinite(loss))
