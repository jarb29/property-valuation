FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY flask-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src directory (needed for config imports)
COPY src ./src

# Copy Flask application with embedded model
COPY flask-app .

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]