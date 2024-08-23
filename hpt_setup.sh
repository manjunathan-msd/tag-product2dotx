#!/bin/bash

# Function to check if Conda is installed
check_conda_installed() {
    if ! command -v conda &>/dev/null; then
        echo "Conda is not installed. Installing Miniconda..."
        if [ "$(uname)" == "Darwin" ]; then
            # MacOS
            curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
            bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda
            export PATH="$HOME/miniconda/bin:$PATH"
        elif [ "$(uname -s)" == "Linux" ]; then
            # Linux
            curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
            bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
            export PATH="$HOME/miniconda/bin:$PATH"
        else
            echo "Unknown operating system. Please install Conda manually."
            exit 1
        fi
    fi
}

# Install Conda if not installed
check_conda_installed

# Clone the HPT repository
git clone https://github.com/manjunathan-msd/HPT.git

# Change directory to HPT
cd HPT || exit

# Create and activate a new Conda environment
ENV_NAME="hpt_env"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# Install requirements
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Check if HyperGAI directory exists, create if it doesn't
if [ ! -d "HyperGAI" ]; then
    mkdir HyperGAI
fi

# Change directory to HyperGAI
cd HyperGAI || exit

# Install Git LFS if it's not installed
if ! git lfs --version &>/dev/null; then
    echo "Git LFS is not installed. Installing Git LFS..."
    if [ "$(uname)" == "Darwin" ]; then
        # MacOS
        brew install git-lfs
    elif [ "$(uname -s)" == "Linux" ]; then
        # Linux
        sudo apt-get install git-lfs -y
    elif [ "$(uname -s)" == "CYGWIN" ] || [ "$(uname -s)" == "MINGW" ]; then
        # Windows with Git Bash
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs
    else
        echo "Unknown operating system. Please install Git LFS manually."
        exit 1
    fi
fi

# Install Git LFS
git lfs install

# Create a directory for HPT1_5-Air-Llama-3-8B-Instruct-multimodal
mkdir -p "HPT1_5-Air-Llama-3-8B-Instruct-multimodal"

# Change directory to the new folder
cd "HPT1_5-Air-Llama-3-8B-Instruct-multimodal" || exit

# Clone the HPT1_5-Edge repository
git clone https://huggingface.co/HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal .

echo "Downloads and installations are successfully completed."
