#!/bin/bash
# One-command deployment for Smart Grid ML Geo Prediction (Pre-trained model)

set -e

echo "========================================"
echo "Smart Grid ML - Geo Prediction"
echo "Cloud Run Deployment"
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
    echo "     SERVICE_NAME='smart-grid-geo'"
    echo "     REGION='us-central1'"
    exit 1
fi

IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check prerequisites
echo "Checking prerequisites..."

# 1. Check if geo model exists
if [ ! -f "data/models/geo_model.keras" ]; then
    echo "❌ Error: Geo model not found at data/models/geo_model.keras"
    echo "   Run the training pipeline first:"
    echo "     1. docker-compose -f docker-compose-downloader.yml up"
    echo "     2. docker-compose up trainer"
    exit 1
fi

# 2. Check other required model files
if [ ! -f "data/models/geo_scaler.pkl" ] || [ ! -f "data/models/geo_features.json" ]; then
    echo "❌ Error: Missing model artifacts (geo_scaler.pkl or geo_features.json)"
    echo "   Run: docker-compose up trainer"
    exit 1
fi

# 3. Check CAISO node data
if [ ! -f "data/caiso-price-map.json" ]; then
    echo "❌ Error: CAISO price map not found at data/caiso-price-map.json"
    exit 1
fi

# 4. Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not installed"
    echo "   Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 5. Check if Docker is running
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

# Get Google Maps API key from environment or .env file
if [ -z "${GOOGLE_MAPS_API_KEY}" ] && [ -f .env ]; then
  export $(grep GOOGLE_MAPS_API_KEY .env | xargs)
fi

if [ -z "${GOOGLE_MAPS_API_KEY}" ]; then
  echo "⚠️  Warning: GOOGLE_MAPS_API_KEY not set. Map features will not work."
  echo "   Set it with: export GOOGLE_MAPS_API_KEY=your_key"
fi

gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --timeout 600 \
  --set-env-vars="NCEI_TOKEN='TDNQRthmEjyQqTHBKZMaQvKvlWUBsbri',GOOGLE_MAPS_API_KEY='${GOOGLE_MAPS_API_KEY}'"

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
echo "2. Geo Health Check:"
echo "   curl ${SERVICE_URL}/geo/health"
echo ""
echo "3. Price Prediction by Coordinates (Los Angeles):"
echo "   curl '${SERVICE_URL}/predict/geo?latitude=34.0522&longitude=-118.2437'"
echo ""
echo "4. Price Prediction by Coordinates (San Francisco):"
echo "   curl '${SERVICE_URL}/predict/geo?latitude=37.7749&longitude=-122.4194'"
echo ""
echo "5. Find Nearby Nodes:"
echo "   curl '${SERVICE_URL}/nodes/nearby?latitude=34.05&longitude=-118.24&radius_km=25'"
echo ""
echo "6. API Info:"
echo "   curl ${SERVICE_URL}/"
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
