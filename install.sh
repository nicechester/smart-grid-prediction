#!/bin/bash
# Create environment
python3.11 -m venv venv_prod
source venv_prod/bin/activate

# Install
pip install -r requirements.txt

# Verify
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
