#!/bin/bash
# Complete GCP Project Setup for Smart Grid ML

set -e

echo "========================================"
echo "GCP Project Setup for Smart Grid ML"
echo "========================================"
echo ""

# Get project ID from user
read -p "Enter your GCP Project ID (e.g., smart-grid-ml-123456): " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: Project ID cannot be empty"
    exit 1
fi

echo ""
echo "Using Project ID: $PROJECT_ID"
echo ""

# Set project
echo "1️⃣  Setting active project..."
gcloud config set project $PROJECT_ID

# Verify project exists
if ! gcloud projects describe $PROJECT_ID &>/dev/null; then
    echo "❌ Error: Project '$PROJECT_ID' not found"
    echo "   Create it at: https://console.cloud.google.com/projectcreate"
    exit 1
fi

echo "✓ Project verified"
echo ""

# Enable required APIs
echo "2️⃣  Enabling required APIs (this may take 1-2 minutes)..."

echo "   - Cloud Run API..."
gcloud services enable run.googleapis.com --quiet

echo "   - Container Registry API..."
gcloud services enable containerregistry.googleapis.com --quiet

echo "   - Cloud Build API..."
gcloud services enable cloudbuild.googleapis.com --quiet

echo "✓ APIs enabled"
echo ""

# Set default region
echo "3️⃣  Setting default region..."
gcloud config set run/region us-central1
gcloud config set compute/region us-central1

echo "✓ Region set to us-central1 (Free Tier eligible)"
echo ""

# Configure Docker authentication
echo "4️⃣  Configuring Docker for Google Container Registry..."
gcloud auth configure-docker --quiet

echo "✓ Docker configured"
echo ""

# Create .env file for deployment
echo "5️⃣  Creating deployment configuration..."

cat > .gcp-config << EOF
# GCP Configuration for Smart Grid ML
export PROJECT_ID="$PROJECT_ID"
export REGION="us-central1"
export SERVICE_NAME="smart-grid-ml"
export IMAGE_NAME="gcr.io/$PROJECT_ID/smart-grid-ml"

# Source this file before deploying:
# source .gcp-config
EOF

echo "✓ Configuration saved to .gcp-config"
echo ""

# Summary
echo "========================================"
echo "✅ GCP Project Setup Complete!"
echo "========================================"
echo ""
echo "Project Details:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: us-central1"
echo "  Billing: Check at https://console.cloud.google.com/billing"
echo ""
echo "Next Steps:"
echo ""
echo "1. Load configuration:"
echo "   source .gcp-config"
echo ""
echo "2. Verify billing is enabled:"
echo "   gcloud billing projects describe $PROJECT_ID"
echo ""
echo "3. Deploy your app:"
echo "   ./deploy-to-cloudrun.sh"
echo ""
echo "Useful Commands:"
echo "  - View project info: gcloud config list"
echo "  - View enabled APIs: gcloud services list --enabled"
echo "  - View billing: gcloud billing accounts list"
echo ""
