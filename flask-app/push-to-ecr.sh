#!/bin/bash

# Step 1: Build and Push Docker Image to ECR
echo "🐳 Building and pushing Docker image to ECR..."

# AWS Profile Selection
AWS_PROFILE=${1:-default}  # Use first argument or default to 'default'
echo "📋 Using AWS profile: $AWS_PROFILE"

AWS_REGION="us-east-1"
ECR_REPO="property-valuation-api"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"

# Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION --profile $AWS_PROFILE 2>/dev/null || {
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION --profile $AWS_PROFILE
    
    # Set lifecycle policy to keep only 3 most recent images
    echo "🗑️  Setting up lifecycle policy to delete old images..."
    aws ecr put-lifecycle-policy --repository-name $ECR_REPO --region $AWS_REGION --profile $AWS_PROFILE --lifecycle-policy-text '{
        "rules": [
            {
                "rulePriority": 1,
                "description": "Keep only 2 most recent images",
                "selection": {
                    "tagStatus": "any",
                    "countType": "imageCountMoreThan",
                    "countNumber": 2
                },
                "action": {
                    "type": "expire"
                }
            }
        ]
    }'
}

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION --profile $AWS_PROFILE | docker login --username AWS --password-stdin $ECR_URI

# Build Docker image locally (from flask-app directory, context = parent)
echo "🔨 Building Docker image for Flask app..."
cd .. && docker build --platform linux/amd64 -f flask-app/Dockerfile -t $ECR_REPO . && cd flask-app

# Tag for ECR
echo "🏷️  Tagging image..."
docker tag $ECR_REPO:latest $ECR_URI:latest

# Push to ECR
echo "⬆️  Pushing to ECR..."
docker push $ECR_URI:latest

echo "✅ Image pushed to ECR: $ECR_URI:latest"
echo "🚀 Now create App Runner service pointing to this image!"