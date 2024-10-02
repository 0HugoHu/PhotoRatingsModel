#!/bin/bash

# Create a virtual environment in a folder named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install required dependencies
pip install pillow transformers requests torch

# Freeze dependencies to a requirements.txt file
pip freeze > requirements.txt

# Set the Hugging Face cache directory
export HUGGINGFACE_HUB_CACHE="D:/hf_cache_directory"

echo "Virtual environment setup is complete. To activate it, run:"
echo "source venv/bin/activate"
