"""Utils for loading Mamba params."""

import functools
from typing import Any, Mapping, Optional

import torch
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
import orbax.checkpoint

Params = Mapping[str, Any]


def layer_exists(layer_num: str, params: Params) -> bool:
    """Check if a layer exists in the parameters."""
    return f"layers_{layer_num}" in params.keys()


def load_params_pytorch(path: str) -> Params:
    """Loads parameters from a checkpoint path."""
    pytorch_state_dict = torch.load("path_to_pytorch_weights.pth", map_location="cpu")
    jax_params = pytorch_to_jax_weights(pytorch_state_dict)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return jax_params


def pytorch_to_jax_weights(pytorch_state_dict):
    jax_params = {}

    for key, value in pytorch_state_dict.items():
        # Convert PyTorch tensor to numpy array
        np_value = value.cpu().numpy()

        # Handle different layer types
        if "embedding" in key:
            jax_params["embedding"] = {"embedding": np_value}
        elif "layers" in key:
            layer_num, rest = key.split(".", 2)[1:]
            if not layer_exists(layer_num, jax_params):
                jax_params[f"layers_{layer_num}"] = {}
            if "mixer" in rest:
                mixer_part = rest.split(".", 1)[1]
                if mixer_part.startswith("in_proj"):
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    jax_params[f"layers_{layer_num}"]["mixer"]["in_proj"] = {
                        "kernel": np_value.T
                    }
                elif mixer_part.startswith("conv1d"):
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    if "weight" in mixer_part:
                        jax_params[f"layers_{layer_num}"]["mixer"]["conv1d"] = {
                            "kernel": np_value.transpose(2, 1, 0)
                        }
                    elif "bias" in mixer_part:
                        jax_params[f"layers_{layer_num}"]["mixer"]["conv1d"] = (
                            jax_params[f"layers_{layer_num}"]["mixer"].get("conv1d", {})
                        )
                        jax_params[f"layers_{layer_num}"]["mixer"]["conv1d"][
                            "bias"
                        ] = np_value
                elif mixer_part.startswith("x_proj"):
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    jax_params[f"layers_{layer_num}"]["mixer"]["x_proj"] = {
                        "kernel": np_value.T
                    }
                elif mixer_part.startswith("dt_proj"):
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    if "weight" in mixer_part:
                        jax_params[f"layers_{layer_num}"]["mixer"]["dt_proj"] = {
                            "kernel": np_value.T
                        }
                    elif "bias" in mixer_part:
                        jax_params[f"layers_{layer_num}"]["mixer"]["dt_proj"] = (
                            jax_params[f"layers_{layer_num}"]["mixer"].get(
                                "dt_proj", {}
                            )
                        )
                        jax_params[f"layers_{layer_num}"]["mixer"]["dt_proj"][
                            "bias"
                        ] = np_value
                elif mixer_part.startswith("out_proj"):
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    jax_params[f"layers_{layer_num}"]["mixer"]["out_proj"] = {
                        "kernel": np_value.T
                    }
                elif mixer_part == "A_log":
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    jax_params[f"layers_{layer_num}"]["mixer"]["A_log"] = np_value
                elif mixer_part == "D":
                    jax_params[f"layers_{layer_num}"]["mixer"] = jax_params[
                        f"layers_{layer_num}"
                    ].get("mixer", {})
                    jax_params[f"layers_{layer_num}"]["mixer"]["D"] = np_value
            elif "norm" in rest:
                jax_params[f"layers_{layer_num}"]["norm"] = {"weight": np_value}
        elif key.startswith("norm_f"):
            jax_params["norm_f"] = {"weight": np_value}
        # elif key.startswith("lm_head"):
        #     jax_params["lm_head"] = {"kernel": np_value.T}

    # Convert numpy arrays to jax arrays
    jax_params = jax.tree_map(jnp.array, jax_params)

    # Round to 4 decimal places
    jax_params = jax.tree_map(lambda x: jnp.round(x, 4), jax_params)

    return freeze(jax_params)
