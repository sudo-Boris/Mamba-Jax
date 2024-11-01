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

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    n_layer: int = 12
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16


class CausalSelfAttention(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):

        assert len(x.shape) == 3

        b, l, d = x.shape

        # q = nn.Dense(self.config.n_embd)(x)
        # k = nn.Dense(self.config.n_embd)(x)
        # v = nn.Dense(self.config.n_embd)(x)
        # # q*k / sqrt(dim) -> softmax -> @v
        # q = jnp.reshape(q, (b, l, d // self.config.n_head, self.config.n_head))
        # k = jnp.reshape(k, (b, l, d // self.config.n_head, self.config.n_head))
        # v = jnp.reshape(v, (b, l, d // self.config.n_head, self.config.n_head))
        # norm = jnp.sqrt(list(jnp.shape(k))[-1])
        # attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / norm
        # mask = jnp.tril(attn)
        # attn = jnp.where(mask[:, :, :l, :l], attn, float("-inf"))
        # probs = jax.nn.softmax(attn, axis=-1)
        # y = jnp.matmul(probs, v)
        # y = jnp.reshape(y, (b, l, d))
        # y = nn.Dense(self.config.n_embd)(y)

        q = nn.Dense(
            self.config.n_embd,
            dtype=self.config.dtype,
            # kernel_init=self.config.kernel_init,
            # bias_init=self.config.bias_init,
        )(x)
        k = nn.Dense(
            self.config.n_embd,
            dtype=self.config.dtype,
            # kernel_init=self.config.kernel_init,
            # bias_init=self.config.bias_init,
        )(x)
        v = nn.Dense(
            self.config.n_embd,
            dtype=self.config.dtype,
            # kernel_init=self.config.kernel_init,
            # bias_init=self.config.bias_init,
        )(x)
        # q*k / sqrt(dim) -> softmax -> @v
        q = jnp.reshape(q, (b, l, d // self.config.n_head, self.config.n_head)).astype(
            jnp.float32
        )
        k = jnp.reshape(k, (b, l, d // self.config.n_head, self.config.n_head)).astype(
            jnp.float32
        )
        v = jnp.reshape(v, (b, l, d // self.config.n_head, self.config.n_head))
        norm = jnp.sqrt(list(jnp.shape(k))[-1])
        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / norm
        mask = jnp.tril(attn)
        attn = jnp.where(mask[:, :, :l, :l], attn, float("-inf")).astype(jnp.float32)
        probs = jax.nn.softmax(attn, axis=-1).astype(self.config.dtype)
        y = jnp.matmul(probs, v)
        y = jnp.reshape(y, (b, l, d))
        y = nn.Dense(self.config.n_embd)(y).astype(self.config.dtype)

        return y


class MLP(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.config.n_embd * 4)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.config.n_embd)(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        return x


class Block(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = x + CausalSelfAttention(self.config)(x)
        x = nn.LayerNorm()(x)
        x = x + MLP(self.config)(x)
        return x


class GPT(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=False):

        B, T = x.shape
        assert T <= self.config.block_size

        pos = jnp.arange(0, T)[None]
        pos_emb = nn.Embed(self.config.block_size, self.config.n_embd)(pos)
        wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        tok_emb = wte(x)
        x = tok_emb + pos_emb

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x)
        x = nn.LayerNorm()(x)
        # logits = nn.Dense(self.config.n_embd, self.config.vocab_size)(x)
        logits = wte.attend(x)  # parameter sharing
        return logits

    def init(self, rng):
        tokens = jnp.zeros((1, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params


def get_gpt_model(
    finetune: bool = None,
    model_config: ModelConfig = None,
    t: int = 1024,
    rng: jax.random.PRNGKey = None,
):

    assert rng is not None, "Must provide a random key for initialization."

    if model_config is None:
        model_config = ModelConfig(block_size=t)

    model = GPT(model_config)
    params = model.init(rng)

    return model, params
