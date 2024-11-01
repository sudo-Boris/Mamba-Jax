# Mamba-Jax (Minimal)

Minimal implementation of Mamba in Jax.

## Setup - Local Conda

Setup for ```NVIDIA A100 GPU``` with ```CUDA 12.5```.

```bash
conda create -y -n mamba-jax python=3.11
conda activate mamba-jax
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U "jax[cuda12]"==0.4.31 jaxlib==0.4.31 flax==0.8.5
pip install transformers
pip install wandb==0.17.5 ipykernel einops "numpy<2.0" "ml-collections"
```

Sanity check for setup:

```bash
sh sanitycheck.sh
```
