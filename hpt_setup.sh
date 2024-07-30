#!/bin/bash


# Clone the HPT repository
git clone https://github.com/HyperGAI/HPT.git

# Change directory to HPT
cd HPT || exit

# Check if HyperGAI directory exists, create if it doesn't
if [ ! -d "HyperGAI" ]; then
    mkdir HyperGAI
fi

# Change directory to HyperGAI
cd HyperGAI || exit

# Install Git LFS
git lfs install

# Clone the HPT1_5-Edge repository
git clone https://huggingface.co/HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal .

echo "Downloads Are Successfully Done"