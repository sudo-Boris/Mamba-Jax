"""Simple, minimal implementation of Mamba in one file of Jax.

Code inspired by https://github.com/johnma2006/mamba-minimal/blob/master/model.py

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""

from __future__ import annotations
import math
import json
from typing import Union

from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from dataclasses import dataclass
from einops import rearrange, repeat, einsum

from model_pytorch import Mamba as MambaPytorch


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba(nn.Module):
    args: ModelArgs

    def setup(self):
        self.embedding = Embedder(self.args.vocab_size, self.args.d_model)
        # self.embedding = nn.Embed(self.args.vocab_size, self.args.d_model)

        # Mamba layers with layer names like "layers_0", "layers_1", etc.
        self.layers = [
            ResidualBlock(self.args, name=f"layers_{i}")
            for i in range(self.args.n_layer)
        ]

        # self.norm_f = RMSNorm()
        self.norm_f = RMSNorm(self.args.d_model)
        self.lm_head = self.embedding

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """
        Args:
            input_ids (int array): shape (b, l)

        Returns:
            logits: shape (b, l, vocab_size)
        """

        # Ensure input_ids is a batch
        if input_ids.ndim == 1:
            input_ids = jnp.expand_dims(input_ids, axis=0)

        x = self.embedding.encode(input_ids)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.norm_f(x)

        logits = self.embedding.decode(x)

        return logits


class Embedder(nn.Module):
    """Embedder module. From https://github.com/google-deepmind/gemma/blob/main/gemma/modules.py"""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def _encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x,)]
        # x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def encode(self, x: jax.Array) -> jax.Array:
        encode = jax.vmap(self._encode, in_axes=(0))
        return encode(x)

    def _decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)

    def decode(self, x: jax.Array) -> jax.Array:
        decode = jax.vmap(self._decode, in_axes=(0))
        return decode(x)


class ResidualBlock(nn.Module):
    """Simple block wrapping Mamba block with normalization and residual connection."""

    args: ModelArgs

    def setup(self):
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)

    def __call__(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)
        """
        x_norm = self.norm(x)
        output = self.mixer(x_norm) + x
        return output


class MambaBlock(nn.Module):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""

    args: ModelArgs

    def setup(self):
        args = self.args

        self.in_proj = nn.Dense(args.d_inner * 2, use_bias=args.bias)

        self.conv1d = nn.Conv(
            features=args.d_inner,
            kernel_size=(args.d_conv,),
            feature_group_count=args.d_inner,
            padding=(args.d_conv - 1),
            use_bias=args.conv_bias,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Dense(args.dt_rank + args.d_state * 2, use_bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Dense(args.d_inner, use_bias=True)

        A = jnp.repeat(jnp.arange(1, args.d_state + 1)[None, :], args.d_inner, axis=0)
        self.A_log = self.param(
            "A_log", lambda _, shape: jnp.log(A), (args.d_inner, args.d_state)
        )
        self.D = self.param("D", nn.initializers.ones, (args.d_inner,))
        self.out_proj = nn.Dense(args.d_model, use_bias=args.bias)

    def __call__(self, x):
        b, l, d = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        x, res = jnp.split(x_and_res, [self.args.d_inner], axis=-1)
        x = self.conv1d(x)[:, :l, :]

        x = jax.nn.silu(x)

        y = self.ssm(x)

        y = y * jax.nn.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        d_in, n = self.A_log.shape

        A = -jnp.exp(self.A_log)
        D = self.D

        x_dbl = self.x_proj(x)

        delta, B, C = jnp.split(
            x_dbl,
            [self.args.dt_rank, self.args.dt_rank + n],
            axis=-1,
        )
        delta = jax.nn.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """

        b, l, d_in = u.shape
        n = A.shape[1]

        deltaA = jnp.exp(jnp.einsum("bld,dn->bldn", delta, A))
        deltaB_u = jnp.einsum("bld,bln,bld->bldn", delta, B, u)

        def scan_fn(carry, inputs):
            deltaA_t, deltaB_u_t, C_t = inputs
            x = carry

            x = deltaA_t * x + deltaB_u_t
            y = jnp.einsum("bdn,bn->bd", x, C_t)
            return x, y

            # deltaA, deltaB_u, C = inputs
            # x, i = carry
            # deltaA[:, i], deltaB_u[:, i], C[:, i, :]
            # deltaA_t, deltaB_u_t = deltaA[:, i], deltaB_u[:, i]
            # C_t = C[:, i, :]

            # x = deltaA_t * x + deltaB_u_t
            # y = jnp.einsum("bdn,bn->bd", x, C_t)
            # return (x, i + 1), y

        x = jnp.zeros((b, d_in, n))
        # _, ys = jax.lax.scan(scan_fn, (x, 0), (deltaA, deltaB_u, C))
        ys = []
        for i in range(l):
            x, y = scan_fn(x, (deltaA[:, i], deltaB_u[:, i], C[:, i, :]))
            ys.append(y)
        ys = jnp.stack(ys, axis=1)  # shape (b, l, d_in)

        y = ys + u * D

        return y


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        output = (
            x
            * jnp.reciprocal(
                jnp.sqrt(self.eps + jnp.mean(x**2, axis=-1, keepdims=True))
            )
            * self.weight
        )
        return output

    def setup(self):
        self.weight = self.param(
            "weight", lambda rng, shape: jnp.ones(shape), (self.d_model,)
        )


def load_model_from_pytorch(
    pretrained_model_name: str = "state-spaces/mamba-370m",
    tokenizer: str = "EleutherAI/gpt-neox-20b",
):
    """Load a pretrained model from HuggingFace and convert it to a Flax model.

    Args:
        pretrained_model_name (str): The name of the model to load. One of:
            'state-spaces/mamba-2.8b-slimpj'
            'state-spaces/mamba-2.8b'
            'state-spaces/mamba-1.4b'
            'state-spaces/mamba-790m'
            'state-spaces/mamba-370m'
            'state-spaces/mamba-130m'
        tokenizer (str): The name of the tokenizer to use. One of:
            'EleutherAI/gpt-neox-20b'

    Returns:
        model: The flax model
        weights: The pretrained weights for the model
        tokenizer: The tokenizer to use with the model
    """
    modelPytorch = MambaPytorch.from_pretrained(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    modelArgsJax = ModelArgs(
        d_model=modelPytorch.args.d_model,
        n_layer=modelPytorch.args.n_layer,
        vocab_size=modelPytorch.args.vocab_size,
    )

    modelJax = Mamba(modelArgsJax)

    from params import pytorch_to_jax_weights

    weights = pytorch_to_jax_weights(modelPytorch.state_dict())

    return modelJax, weights, tokenizer


def get_mamba_model(
    finetune: bool = None,
    model_config: ModelArgs = None,
    b: int = 1,
    t: int = 128,
    rng: jax.random.PRNGKey = None,
):

    assert rng is not None, "Must provide a random key for initialization."

    if finetune is not None:
        model, params, tokenizer = load_model_from_pytorch()
    else:
        # Initialize model
        modelArgsJax = ModelArgs(
            d_model=1024,
            n_layer=48,
            vocab_size=50280,
        )
        model = Mamba(modelArgsJax)

        # Initialize parameters
        param_key, dropout_key = jax.random.split(rng, 2)
        model_rng = {"params": param_key, "dropout": dropout_key}
        example_input = jnp.ones((b, t), jnp.int32)
        params = model.init(model_rng, example_input)["params"]

    return model, params
