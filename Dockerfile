# Use Python 3.8 slim as the base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data \
    /app/outputs \
    /app/outputs/pipeline/models \
    /app/outputs/pipeline/schema \
    /app/outputs/pipeline/data \
    /app/outputs/pipeline/logs \
    /app/outputs/jupyter/models \
    /app/outputs/jupyter/schema \
    /app/outputs/jupyter/data \
    /app/outputs/jupyter/logs \
    /app/outputs/predictions \
    /app/outputs/predictions/schema

# Copy project files
COPY . .

# Set permissions for the directories
RUN chmod -R 777 /app/data /app/outputs

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "scripts/run_api.py"]
