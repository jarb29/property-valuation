#!/bin/bash

echo "🐳 Building and running Flask Property Valuation App in Docker..."

# Step 1: Prepare best model
echo "📦 Preparing best model for Docker..."
python prepare-model.py

if [ ! -f "models/bestmodel/model.onnx" ]; then
    echo "❌ Model preparation failed!"
    exit 1
fi

echo "✅ Best model prepared successfully"

# Step 2: Build the Docker image
echo "🏗️ Building Docker image..."
docker-compose build --no-cache

# Step 3: Start the container
echo "🚀 Starting container..."
docker-compose up -d

# Show status
echo "📊 Container status:"
docker-compose ps

echo ""
echo "✅ Flask app is running!"
echo "🌐 Access the app at: http://localhost:5002"
echo "🔍 Health check: http://localhost:5002/health"
echo "📊 Model info: http://localhost:5002/model/info"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"