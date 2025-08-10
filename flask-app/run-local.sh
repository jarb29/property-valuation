#!/bin/bash

echo "🚀 Starting Flask App (Local Development)..."

# Prepare best model
echo "📦 Preparing best model..."
python3 prepare-model.py

# Start Flask app
echo "🌐 Starting Flask server..."
python3 app.py