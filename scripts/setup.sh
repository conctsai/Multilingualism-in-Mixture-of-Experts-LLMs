#!/bin/bash

# Stop execution immediately if any command fails
set -e

# Configuration
ENV_NAME="moe_multilingual"
PYTHON_VERSION="3.11"
SOURCE_FILE="qwen3_moe.py"
REQUIREMENTS_FILE="requirements.txt"

echo "=========================================="
echo "Setting up environment: $ENV_NAME"
echo "=========================================="

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda or Miniconda is installed."
    exit 1
fi

# 2. Create Conda Environment
# Check if environment already exists
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        echo "Creating new environment..."
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    else
        echo "Using existing environment..."
    fi
else
    echo "Creating environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# 3. Activate Environment
# We need to source conda.sh to activate conda environments within a script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "Activated environment: $ENV_NAME"

# 4. Install Dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE
else
    echo "Warning: $REQUIREMENTS_FILE not found in the current directory. Skipping pip install."
fi

# 5. Replace vllm File
echo "Starting file replacement process..."

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found in current directory. Cannot proceed with replacement."
    exit 1
fi

# Dynamically find the vllm installation path in the *current* environment
# This is safer than hardcoding /data/czz/miniconda3/...
echo "Locating vllm installation path..."
VLLM_DIR=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "NOT_FOUND")

if [ "$VLLM_DIR" == "NOT_FOUND" ]; then
    echo "Error: 'vllm' package not found in the environment. Please ensure it is listed in requirements.txt."
    exit 1
fi

# Construct the target path
TARGET_FILE="$VLLM_DIR/model_executor/models/qwen3_moe.py"
echo "Target file located at: $TARGET_FILE"

# Backup the original file
if [ -f "$TARGET_FILE" ]; then
    echo "Backing up original file to ${TARGET_FILE}.bak"
    mv "$TARGET_FILE" "${TARGET_FILE}.bak"
fi

# Copy the new file
cp "$SOURCE_FILE" "$TARGET_FILE"
echo "File replaced successfully!"

# Download datasets
cd datasets
../scripts/hfd.sh facebook/belebele --dataset
../scripts/hfd.sh openlanguagedata/flores_plus --dataset
../scripts/hfd.sh juletxara/mgsm --dataset
../scripts/hfd.sh google/xquad --dataset
../scripts/hfd.sh Qwen/PolyMath --dataset


echo "=========================================="
echo "Setup complete."
echo "To use the environment, run: conda activate $ENV_NAME"
echo "=========================================="