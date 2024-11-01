from dataclasses import dataclass
from typing import Tuple
import time

from absl import logging, flags, app
from flax import linen as nn
import jax
import jax.numpy as jnp
import tiktoken
from flax.core import FrozenDict
from flax.training.train_state import TrainState
import optax


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    n_layer: int = 12
    dropout_rate: float = 0.1


class CausalSelfAttention(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):

        assert len(x.shape) == 3

        b, l, d = x.shape

        q = nn.Dense(self.config.n_embd)(x)
        k = nn.Dense(self.config.n_embd)(x)
        v = nn.Dense(self.config.n_embd)(x)
        # q*k / sqrt(dim) -> softmax -> @v
        q = jnp.reshape(q, (b, l, d // self.config.n_head, self.config.n_head))
        k = jnp.reshape(k, (b, l, d // self.config.n_head, self.config.n_head))
        v = jnp.reshape(v, (b, l, d // self.config.n_head, self.config.n_head))
        norm = jnp.sqrt(list(jnp.shape(k))[-1])
        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / norm
        mask = jnp.tril(attn)
        attn = jnp.where(mask[:, :, :l, :l], attn, float("-inf"))
        probs = jax.nn.softmax(attn, axis=-1)
        y = jnp.matmul(probs, v)
        y = jnp.reshape(y, (b, l, d))
        y = nn.Dense(self.config.n_embd)(y)
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


def count_params_jax(weights):
    p = jax.tree_util.tree_map(
        lambda a: a.size if isinstance(a, jnp.ndarray) else 0, weights
    )
    return jax.tree_util.tree_reduce(lambda a, b: a + b, p)


# config = ModelConfig()
# key = jax.random.PRNGKey(0)
# model = GPT(config)
# params = model.init(key)
# logging.info(count_params_jax(params))


class DataLoader:
    def __init__(self, B, T):
        self.current_position = 0
        self.B = B
        self.T = T

        with open("data/input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = jnp.array(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens in the datasets")
        print(f" 1 epoch = {len(self.tokens)//(B*T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x, y = jnp.reshape(buf[:-1], (B, T)), jnp.reshape(buf[1:], (B, T))
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


def init_train_state(key, config) -> TrainState:
    model = GPT(config)
    params = model.init(key)
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.98, eps=1e-9, weight_decay=1e-1)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return train_state


@jax.jit
def train_step(
    state: TrainState, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, TrainState]:

    def loss_fn(params: FrozenDict) -> jnp.ndarray:

        logits = state.apply_fn(params, x, False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def main(_):
    config = ModelConfig()
    key = jax.random.PRNGKey(0)
    train_state = init_train_state(key, config)
    dataloader = DataLoader(4, 128)

    x, y = dataloader.next_batch()
    train_steps = 50
    for step in range(train_steps):
        t0 = time.time()
        loss, train_state = train_step(train_state, x, y)
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = dataloader.B * dataloader.T
        tokens_per_sec = tokens_processed / dt
        logging.info(
            f"step {step}/{train_steps}  |  loss: {loss:.4f}  |  dt: {dt*1000:.2f}ms  |  tokens/s: {tokens_per_sec:.2f}",
        )


if __name__ == "__main__":
    app.run(main)
