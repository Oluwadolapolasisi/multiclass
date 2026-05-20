#!/bin/bash
# Set environment variables for Kaggle
# This script can be sourced to set the environment variable in the current shell session
# Usage: source set_env_variables.sh
# NOTE: Ensure that you make a copy of this script, rename the copy as 'set_env_var.sh' 
# and replace the placeholder with your actual APIs token before sourcing it.

# Replace 'xxxxxxxxxxxxxx' with your actual Kaggle API token
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx

# Set LD_LIBRARY_PATH for TensorFlow GPU support if virtual environment is used
if [ -d ".venv" ]; then
    VENV_NVIDIA_PATHS=$(.venv/bin/python3 -c "import os, sys, glob; site_pkgs = next((p for p in sys.path if 'site-packages' in p), None); print(':'.join(glob.glob(os.path.join(site_pkgs, 'nvidia', '*', 'lib'))) if site_pkgs else '')" 2>/dev/null)
    if [ ! -z "$VENV_NVIDIA_PATHS" ]; then
        export LD_LIBRARY_PATH="$VENV_NVIDIA_PATHS:$LD_LIBRARY_PATH"
        echo "TensorFlow GPU libraries configured in LD_LIBRARY_PATH."
    fi
fi
