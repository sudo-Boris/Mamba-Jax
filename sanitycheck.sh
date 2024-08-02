nvidia-smi
echo "\n- - - - python3 --version"
python3 --version
echo "\n- - - - nvcc --version"
nvcc --version
echo "\n- - - - torch.__version__"
python3 -c "import torch; print(torch.__version__)"
echo "\n- - - - torch.cuda.is_available()"
python3 -c "import torch; print(torch.cuda.is_available())"
echo "\n- - - - jax.__version__"
python3 -c "import jax; print(jax.__version__)"
echo "\n- - - - jax.devices()"
python3 -c "import jax; print(jax.devices())"