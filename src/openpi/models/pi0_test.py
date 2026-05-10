import flax.nnx as nnx
import jax

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
