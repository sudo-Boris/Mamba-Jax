""" Training/ finetuning script for the model. Based on https://github.com/google/flax/blob/main/examples/nlp_seq/train.py
"""

import time

from absl import logging, flags, app
from ml_collections import config_flags
import ml_collections
import jax
from jax import numpy as jnp
from jax import random
import optax
from transformers import AutoTokenizer
import tiktoken
from tqdm import tqdm

from model import get_mamba_model
from NanoGPT import get_gpt_model
from utils.train_state import TrainState

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)

flags.DEFINE_string(
    "workdir", default=".", help="Directory to save checkpoints and logs."
)
flags.DEFINE_boolean(
    "cleanup", default=False, help="Delete workdir (only) after successful completion."
)
flags.DEFINE_integer("seed", default=42, help="Random seed.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()

train_config = ml_collections.ConfigDict(
    {
        # "B": 1,
        # "T": 32,
        # "B": 4,
        # "T": 128,
        "B": 16,
        "T": 1024,
        "EPOCHS": 10,
        "LR": 3e-4,
        "BETA1": 0.9,
        "BETA2": 0.99,
        "WORKDIR": ".",
        "FINETUNE": None,
        # "MODEL": "Mamba",
        "MODEL": "NanoGPT",
    }
)

model_config = ml_collections.ConfigDict(
    {
        "d_model": 1024,
        "n_layer": 48,
        "vocab_size": 50280,
    }
)


class DataLoader:
    def __init__(self, B, T, tokenizer):
        self.current_position = 0
        self.B = B
        self.T = T

        with open("data/input.txt", "r") as f:
            text = f.read()

        if tokenizer == "gpt2":
            tokenizer = tiktoken.get_encoding("gpt2")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, clean_up_tokenization_spaces=True
            )

        self.tokens = jnp.array(tokenizer.encode(text))

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x, y = jnp.reshape(buf[:-1], (B, T)), jnp.reshape(buf[1:], (B, T))
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


def main(_):

    rng = jax.random.PRNGKey(flags.FLAGS.seed)

    ################################################################################
    #                                                                              #
    #                                Set up logging                                #
    #                                                                              #
    ################################################################################

    logging.info(
        f"\u001b[33mHello from process {jax.process_index()} holding "
        f"{jax.local_device_count()}/{jax.device_count()} devices and "
        f"writing to workdir '{flags.FLAGS.workdir}'.\u001b[0m"
    )

    def info(s, *a):
        logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    ################################################################################
    #                                                                              #
    #                                Input Pipeline                                #
    #                                                                              #
    ################################################################################

    write_note("Initializing train dataset...")

    batch_size = train_config.B
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must "
            f"be divisible by device number ({jax.device_count()})"
        )
    info(
        "Global batch size %d on %d hosts results in %d local batch size. With "
        "%d dev per host (%d dev total), that's a %d per-device batch size.",
        batch_size,
        jax.process_count(),
        batch_size // jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        batch_size // jax.device_count(),
    )

    if train_config.MODEL == "Mamba":
        dataloader = DataLoader(
            train_config.B, train_config.T, "EleutherAI/gpt-neox-20b"
        )
    elif train_config.MODEL == "NanoGPT":
        dataloader = DataLoader(train_config.B, train_config.T, "gpt2")

    write_note(
        f"Loaded {len(dataloader.tokens):,} tokens in the datasets. With 1 epoch being "
        f"{(len(dataloader.tokens)//(train_config.B*train_config.T)):,} batches. Running for "
        f"{train_config.EPOCHS:,} epochs, i.e. {(len(dataloader.tokens) * train_config.EPOCHS):,} tokens in total."
    )

    # x, y = dataloader.next_batch()
    # print(x.shape, y.shape)

    ################################################################################
    #                                                                              #
    #                      Create (and load) Model & Optimizer                     #
    #                                                                              #
    ################################################################################

    write_note(f"Creating {train_config.MODEL} model...")
    if train_config.FINETUNE is not None:
        write_note(f"Finetuning from {train_config.FINETUNE}...")

    rng, model_rng = random.split(rng)
    if train_config.MODEL == "Mamba":
        model, params = get_mamba_model(
            finetune=train_config.FINETUNE,
            b=train_config.B,
            t=train_config.T,
            rng=model_rng,
        )
    elif train_config.MODEL == "NanoGPT":
        model, params = get_gpt_model(rng=model_rng)

    def count_params_jax(weights):
        p = jax.tree_util.tree_map(
            lambda a: a.size if isinstance(a, jnp.ndarray) else 0, weights
        )
        return jax.tree_util.tree_reduce(lambda a, b: a + b, p)

    write_note(f"Model has {count_params_jax(params):,} parameters.")

    write_note("Creating optimizer...")
    tx = optax.adamw(
        learning_rate=train_config.LR,
        b1=train_config.BETA1,
        b2=train_config.BETA2,
    )
    model_ts = TrainState.create(model, params, tx=tx, rng=rng)

    ################################################################################
    #                                                                              #
    #                                 Update Step                                  #
    #                                                                              #
    ################################################################################

    @jax.jit
    def update_fn(
        model_ts: TrainState,
        rng: jax.random.PRNGKey,
        x: jnp.ndarray,
        y: jnp.ndarray,
        pmap_axis: str = "batch",
    ) -> tuple[TrainState, dict]:

        def loss_fn(params):
            logits = model_ts(x, params=params)
            # logits = model_ts({"params": params}, x)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)
            )
            return loss

        # grads, info = jax.grad(loss_fn)(model_ts.params)
        params, opt = model_ts.params, model_ts.opt_state

        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)
        # grads = jax.lax.pmean(grads, axis_name=pmap_axis)

        # Update parameters - model_ts.apply_gradients(grads) also does this
        updates, new_opt_state = model_ts.tx.update(grads, opt, params)
        new_params = optax.apply_updates(params, updates)
        new_model_ts = model_ts.replace(
            step=model_ts.step + 1, params=new_params, opt_state=new_opt_state
        )

        measurements = {"training_loss": loss}
        # measurements["l2_grads"] = optax.global_norm(grads)
        # measurements["l2_update"] = optax.global_norm(updates)
        # measurements["l2_params"] = optax.global_norm(new_params)

        return new_model_ts, measurements

    ################################################################################
    #                                                                              #
    #                                  Train Loop                                  #
    #                                                                              #
    ################################################################################

    write_note("Starting training loop...")

    x, y = dataloader.next_batch()
    train_steps = 50
    for step in range(train_steps):
        t0 = time.time()
        model_ts, measurements = update_fn(model_ts, rng, x, y)
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = dataloader.B * dataloader.T
        tokens_per_sec = tokens_processed / dt
        write_note(
            f"step {step}/{train_steps}  |  loss: {measurements['training_loss']:.4f}  |  dt: {dt*1000:.2f}ms  |  tokens/s: {tokens_per_sec:.2f}",
        )


if __name__ == "__main__":
    app.run(main)
