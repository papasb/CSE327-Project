#!/bin/bash

# ENV_NAME="spike_driven"
# PYTHON_VERSION="3.9"

# echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
# conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# echo "Activating environment '$ENV_NAME'..."
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate $ENV_NAME

# conda install -c conda-forge cupy=10.3.1 cudatoolkit=11.3 -y
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch -c conda-forge
# conda install timm==0.6.12
# conda install spikingjelly==0.0.0.0.14
# conda install tensorboard
# conda install torchinfo
# conda install numpy==1.23.0

export TORCH_CUDA_ARCH_LIST="8.9"

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"  # Adjust based on `nvidia-smi`
pip uninstall spikingjelly -y
pip install spikingjelly==0.0.0.0.14 --no-cache-dir

echo "All done! Environment '$ENV_NAME' is ready to use."
