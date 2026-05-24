import os

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax.numpy as jnp
import numpy as np

from openpi.shared import nnx_utils
from openpi.training import checkpoints


def test_ema_inference_params_preserve_frozen_leaves_bitwise():
    params = nnx.State(
        {
            "frozen_executor": {
                "kernel": nnx.Param(jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16)),
            },
            "reference_encoder": {
                "kernel": nnx.Param(jnp.asarray([3.0, 4.0], dtype=jnp.float32)),
            },
        }
    )
    ema_params = nnx.State(
        {
            # Simulates accumulated bf16 EMA drift on a frozen leaf. This must
            # never be exported into inference params for reference-student runs.
            "frozen_executor": {
                "kernel": nnx.Param(jnp.asarray([0.9921875, 1.9921875], dtype=jnp.bfloat16)),
            },
            "reference_encoder": {
                "kernel": nnx.Param(jnp.asarray([30.0, 40.0], dtype=jnp.float32)),
            },
        }
    )

    exported = checkpoints._ema_inference_params(  # noqa: SLF001
        params,
        ema_params,
        nnx_utils.PathRegex("reference_encoder/.*"),
    ).to_pure_dict()

    np.testing.assert_array_equal(
        np.asarray(exported["frozen_executor"]["kernel"]),
        np.asarray(params.to_pure_dict()["frozen_executor"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(exported["reference_encoder"]["kernel"]),
        np.asarray(ema_params.to_pure_dict()["reference_encoder"]["kernel"]),
    )
