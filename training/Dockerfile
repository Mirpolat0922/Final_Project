# Dockerfile for training sentiment analysis model
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Initialize settings_name variable
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files
COPY training /app/training
COPY data_process /app/data_process
COPY utils.py /app/
COPY ${CONF_PATH} /app/

# Create necessary directories
RUN mkdir -p data/raw models/figures

# Download data and train model
RUN python3 data_process/data_generation.py && \
    python3 training/train.py

# Default command
CMD ["python3", "training/train.py"]