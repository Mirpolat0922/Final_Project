# Dockerfile for inference
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Initialize arguments
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files
COPY inference /app/inference
COPY utils.py /app/
COPY ${CONF_PATH} /app/

# Create necessary directories
RUN mkdir -p data/raw models results/figures

# Default command
CMD ["python3", "inference/run.py"]