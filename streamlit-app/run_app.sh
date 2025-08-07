#!/bin/bash

echo "🚀 Starting Streamlit App (Local Development)..."

# Navigate to the project root and activate virtual environment
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Navigate to streamlit-app directory
cd streamlit-app

# Prepare best model
echo "📦 Preparing best model..."
python prepare-model.py

# Start Streamlit app
echo "🌐 Starting Streamlit server..."
streamlit run app.py --server.port 8501