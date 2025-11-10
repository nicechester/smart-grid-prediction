# Use the official Python 3.11 image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including GDAL for geopandas
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libpng-dev \
    libxml2-dev \
    libxslt-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create necessary directories
RUN mkdir -p /app/src /app/data/downloads /app/data/training /app/data/models /app/templates

# Copy the application files to the container
COPY src/ /app/src/
COPY templates/ /app/templates/

# Copy model and data files
COPY data/ /app/data/

# Set working directory to src for Python imports
WORKDIR /app/src

# Expose the port the app runs on
EXPOSE 8000

CMD ["python", "-u", "app.py"]