#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the training script
echo "Running training script..."
python train.py

# Start the uvicorn server
echo "Starting uvicorn server..."
uvicorn app.main:app --reload --port 8000
