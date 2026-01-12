#!/bin/bash
# One-command deployment for Smart Grid ML (Pre-trained model)

set -e

echo "========================================"
echo "Smart Grid ML - Cloud Run Deployment"
echo "========================================"
echo ""

# Environment variables are loaded from .gcp-config file
if [ -f ".gcp-config" ]; then
    echo "Loading configuration from .gcp-config..."
    source .gcp-config
    echo "✓ Configuration loaded"
    echo ""
else
    echo "❌ Error: .gcp-config file not found!"
    echo "   Create a .gcp-config file with the following variables:"
    echo "     PROJECT_ID='your-gcp-project-id'"
    echo "     SERVICE_NAME='your-cloud-run-service-name'"
    echo "     REGION='your-gcp-region'"
    echo "   Example:"
    echo "     PROJECT_ID='smart-grid-479417'"
    echo "     SERVICE_NAME='smart-grid'"
    echo "     REGION='us-central1'"
    exit 1
fi
# Configuration
# PROJECT_ID="smart-grid-479417"  # CHANGE THIS!
# SERVICE_NAME="smart-grid"
# REGION="us-central1"  # Free tier eligible region
# IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check prerequisites
echo "Checking prerequisites..."

# 1. Check if model exists
if [ ! -f "data/models/price_model.keras" ]; then
    echo "❌ Error: Model not found at models/price_model.keras"
    echo "   Run: python train.py --epochs 50"
    exit 1
fi

# 2. Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not installed"
    echo "   Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 3. Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

echo "✓ All prerequisites met"
echo ""

# Set project
echo "Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs (first time only)
echo "Enabling Cloud Run API..."
gcloud services enable run.googleapis.com --quiet 2>/dev/null || true
gcloud services enable containerregistry.googleapis.com --quiet 2>/dev/null || true

echo "✓ APIs enabled"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f Dockerfile.cloudrun -t ${IMAGE_NAME} .

echo "✓ Image built"
echo ""

# Configure Docker for GCR
echo "Configuring Docker authentication..."
gcloud auth configure-docker --quiet

# Push to Google Container Registry
echo "Pushing image to GCR (this may take 2-5 minutes)..."
docker push ${IMAGE_NAME}

echo "✓ Image pushed"
echo ""

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --timeout 600 \
  --set-env-vars="EIA_API_KEY='AzFUfTPb16YRdotKhve64uxbg7lRfrBqm9nqfaJ2',NCEI_TOKEN='TDNQRthmEjyQqTHBKZMaQvKvlWUBsbri'"

echo "✓ Deployed!"
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format 'value(status.url)')

echo "========================================"
echo "🎉 Deployment Complete!"
echo "========================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test your endpoints:"
echo ""
echo "1. Health Check:"
echo "   curl ${SERVICE_URL}/health"
echo ""
echo "2. Price Prediction (Los Angeles):"
echo "   curl '${SERVICE_URL}/predict?location_id=los_angeles&location_type=city'"
echo ""
echo "3. Available Locations:"
echo "   curl ${SERVICE_URL}/locations"
echo ""
echo "4. Web UI:"
echo "   Open: ${SERVICE_URL}/"
echo ""
echo "========================================"
echo "Monitoring & Logs:"
echo "========================================"
echo ""
echo "View logs:"
echo "  gcloud run logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "View in console:"
echo "  https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}"
echo ""