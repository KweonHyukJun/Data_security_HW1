#!/bin/bash

# Exit immediately if any command fails
set -e

# Run the training script
echo "Starting training..."
python train.py

# Run the inference script
echo "Starting inference..."
python run.py

echo "All tasks completed successfully."
