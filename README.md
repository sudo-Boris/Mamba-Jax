# Mamba-Jax (Minimal)

Minimal implementation of Mamba in Jax.

## Setup - Local Conda

Setup for ```NVIDIA A100 GPU``` with ```CUDA 12.5```.

```bash
conda create -y -n mamba-jax python=3.11
conda activate mamba-jax
pip install -U "jax[cuda12]" jaxlib==0.4.30 flax==0.6.11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install wandb==0.17.5
```

Sanity check for setup:

```bash
sh sanitycheck.sh
```
