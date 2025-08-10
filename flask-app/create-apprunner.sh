#!/bin/bash

# Step 2: Create App Runner Service from ECR Image
echo "🚀 Creating App Runner service from ECR image..."

# AWS Profile Selection
AWS_PROFILE=${1:-default}  # Use first argument or default to 'default'
echo "📋 Using AWS profile: $AWS_PROFILE"

AWS_REGION="us-east-1"
ECR_REPO="property-valuation-api"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"

# Create IAM role for App Runner to access ECR
ROLE_NAME="AppRunnerECRAccessRole"
echo "🔑 Creating IAM role for ECR access..."
aws iam get-role --role-name $ROLE_NAME --profile $AWS_PROFILE 2>/dev/null || {
    aws iam create-role --role-name $ROLE_NAME --profile $AWS_PROFILE --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "build.apprunner.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'
    
    aws iam attach-role-policy --role-name $ROLE_NAME --profile $AWS_PROFILE --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
    
    echo "⏳ Waiting for role to be ready..."
    sleep 10
}

ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"

# Create App Runner service
aws apprunner create-service \
  --service-name "property-valuation-api" \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$ECR_URI'",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": {
          "DATA_VERSION": "v3",
          "FLASK_ENV": "production"
        }
      },
      "ImageRepositoryType": "ECR"
    },
    "AuthenticationConfiguration": {
      "AccessRoleArn": "'$ROLE_ARN'"
    },
    "AutoDeploymentsEnabled": true
  }' \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 20,
    "Timeout": 10,
    "HealthyThreshold": 2,
    "UnhealthyThreshold": 3
  }' \
  --region $AWS_REGION \
  --profile $AWS_PROFILE

echo "✅ App Runner service created!"
echo "📱 Check AWS Console for the service URL"
echo "🔗 Or run: aws apprunner list-services --region $AWS_REGION --profile $AWS_PROFILE"