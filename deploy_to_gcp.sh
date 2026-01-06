#!/bin/bash

# =============================================================================
# Deploy BERT Symptom NER Training to Google Cloud Platform
# =============================================================================
# This script builds and pushes the Docker image, then creates a custom job
# on GCP Vertex AI for training.
#
# Usage:
#   ./deploy_to_gcp.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker Desktop with buildx support
#   - GCP project with billing enabled
#   - Required APIs enabled (artifactregistry, aiplatform)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ID="ai-project-482122"
REGION="us-central1"
REPOSITORY_NAME="bert-symptom-ner"
IMAGE_NAME="train"
IMAGE_TAG="latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Job configuration
JOB_DISPLAY_NAME="bert-ner-train-t4"
MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT=1
REPLICA_COUNT=1

# =============================================================================
# HELPER COMMANDS (COMMENTED OUT - UNCOMMENT AS NEEDED)
# =============================================================================

# --- Create Artifact Registry Repository ---
# gcloud artifacts repositories create ${REPOSITORY_NAME} \
#   --repository-format=docker \
#   --location=${REGION} \
#   --description="Docker images for BERT symptom NER training and inference"

# --- Delete Previous Images (Optional: Clean up old images) ---
# List all images:
# gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}
#
# Delete a specific image by digest:
# gcloud artifacts docker images delete ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}@DIGEST
#
# Delete all images with a specific tag:
# gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME} \
#   --format="value(package)" | xargs -I {} gcloud artifacts docker images delete {} --delete-tags

# --- Delete Previous Custom Jobs (Optional: Clean up old jobs) ---
# List all custom jobs:
# gcloud ai custom-jobs list --region=${REGION} --format="table(name,displayName,state)"
#
# Delete a specific job (replace JOB_ID with actual ID):
# gcloud ai custom-jobs delete projects/PROJECT_NUMBER/locations/${REGION}/customJobs/JOB_ID --region=${REGION}
#
# Delete all jobs in a region (use with caution):
# gcloud ai custom-jobs list --region=${REGION} --format="value(name)" | \
#   xargs -I {} gcloud ai custom-jobs delete {} --region=${REGION}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

echo "============================================================================="
echo "BERT Symptom NER - GCP Deployment Script"
echo "============================================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image URI: ${IMAGE_URI}"
echo "============================================================================="
echo ""

# Step 1: Verify prerequisites
echo "[Step 1] Verifying prerequisites..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Please install it first."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install Docker Desktop."
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Step 2: Authenticate Docker with Artifact Registry
echo "[Step 2] Authenticating Docker with Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
echo "✓ Docker authenticated"
echo ""

# Step 3: Setup Docker buildx (if not already set up)
echo "[Step 3] Setting up Docker buildx..."
if ! docker buildx ls | grep -q "multiarch-builder"; then
    echo "Creating buildx builder..."
    docker buildx create --use --name multiarch-builder
    docker buildx inspect --bootstrap
    echo "✓ Buildx builder created"
else
    echo "✓ Buildx builder already exists"
    docker buildx use multiarch-builder
fi
echo ""

# Step 4: Build and push Docker image
echo "[Step 4] Building and pushing Docker image..."
echo "This may take several minutes..."
docker buildx build --platform linux/amd64 \
  -f Dockerfile.train \
  -t ${IMAGE_URI} \
  --push .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built and pushed successfully"
else
    echo "❌ Error: Failed to build/push Docker image"
    exit 1
fi
echo ""

# Step 5: Verify image was pushed
echo "[Step 5] Verifying image in Artifact Registry..."
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME} \
  --format="table(package,version,create_time)" | head -5
echo ""

# Step 6: Create custom job
echo "[Step 6] Creating custom job..."
echo "Job name: ${JOB_DISPLAY_NAME}"
echo "Machine type: ${MACHINE_TYPE}"
echo "GPU: ${ACCELERATOR_TYPE} x${ACCELERATOR_COUNT}"
echo ""

JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_DISPLAY_NAME} \
  --worker-pool-spec=machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},replica-count=${REPLICA_COUNT},container-image-uri=${IMAGE_URI} \
  2>&1)

if [ $? -eq 0 ]; then
    echo "✓ Custom job created successfully"
    echo ""
    echo "${JOB_OUTPUT}"
    echo ""
    
    # Extract job ID from output (macOS compatible - doesn't use -P flag)
    JOB_ID=$(echo "${JOB_OUTPUT}" | grep -o 'customJobs/[0-9]*' | head -1 | sed 's/customJobs\///')
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    
    if [ ! -z "${JOB_ID}" ]; then
        echo "============================================================================="
        echo "Job created successfully!"
        echo "============================================================================="
        echo "Job ID: ${JOB_ID}"
        echo "Project Number: ${PROJECT_NUMBER}"
        echo ""
        echo "View in GCP Console:"
        echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs/${JOB_ID}?project=${PROJECT_ID}"
        echo "============================================================================="
        echo ""
        
        # Step 7: Stream logs
        echo "[Step 7] Streaming job logs..."
        echo "Waiting for job to start (this may take a few minutes)..."
        echo "Press Ctrl+C to stop streaming (job will continue running)"
        echo ""
        echo "============================================================================="
        
        gcloud ai custom-jobs stream-logs projects/${PROJECT_NUMBER}/locations/${REGION}/customJobs/${JOB_ID}
        
        echo ""
        echo "============================================================================="
        echo "Log streaming ended."
        echo ""
        echo "To check job status:"
        echo "  gcloud ai custom-jobs describe projects/${PROJECT_NUMBER}/locations/${REGION}/customJobs/${JOB_ID} --region=${REGION}"
        echo ""
        echo "To resume streaming logs:"
        echo "  gcloud ai custom-jobs stream-logs projects/${PROJECT_NUMBER}/locations/${REGION}/customJobs/${JOB_ID}"
        echo "============================================================================="
    else
        echo "⚠ Warning: Could not extract job ID. Check the output above."
        echo "You can find the job ID in the GCP Console and stream logs manually."
    fi
else
    echo "❌ Error: Failed to create custom job"
    echo "${JOB_OUTPUT}"
    exit 1
fi

echo ""
echo "✓ Deployment complete!"

