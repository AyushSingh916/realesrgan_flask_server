# Use Ubuntu as base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
   python3.10 \
   python3-pip \
   python3-dev \
   build-essential \
   wget \
   && rm -rf /var/lib/apt/lists/*

# Set Python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
   && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
   PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    opencv-contrib-python==4.11.0.86 \
    realesrgan==0.3.0 \
    basicsr==1.4.2 \
    numpy==1.23.5 \
    flask==2.3.2

# Copy project files
COPY . /app

# Create pretrained_models directory
RUN mkdir -p pretrained_models

# Download models 
RUN python update_degradations.py

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]