#!/bin/bash

echo "🔍 Debugging Docker container..."

# Check container logs
echo "📝 Container logs:"
docker-compose logs flask-app

echo ""
echo "📁 Checking files inside container:"
docker-compose exec flask-app ls -la /app/

echo ""
echo "📁 Checking bestmodel directory:"
docker-compose exec flask-app ls -la /app/models/bestmodel/ 2>/dev/null || echo "❌ bestmodel directory not found"

echo ""
echo "🐍 Checking Python path:"
docker-compose exec flask-app python -c "import sys; print('Python path:'); [print(p) for p in sys.path]"

echo ""
echo "🔍 Testing model loading:"
docker-compose exec flask-app python -c "
try:
    from models.predictor import PropertyPredictor
    predictor = PropertyPredictor()
    print('✅ Model loaded successfully')
    print(f'📊 Model info: {predictor.get_model_info()}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"