# Docker configuration for Network Traffic Anomaly Detection System

# Multi-stage build for production deployment
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tcpdump \
    wireshark-common \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/cicids2017 data/captured models logs

# Expose ports
EXPOSE 8501 9092 8080

# Default command
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development stage
FROM base as development

ENV ENVIRONMENT=development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    jupyter

# Development command
CMD ["python", "setup.py"]

# Production stage
FROM base as production

ENV ENVIRONMENT=production

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Production command
CMD ["python", "src/streaming/pipeline.py"]
