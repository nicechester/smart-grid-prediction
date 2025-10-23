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
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# No CMD here - docker-compose.yml specifies commands per service