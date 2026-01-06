# Training on GCP with Docker

This guide documents the complete setup process for training BERT-based NER models on Google Cloud Platform using Docker containers and Vertex AI Custom Jobs.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Docker Image Setup](#docker-image-setup)
- [Building and Pushing the Image](#building-and-pushing-the-image)
- [Creating and Running Custom Jobs](#creating-and-running-custom-jobs)
- [Troubleshooting](#troubleshooting)

## Overview

This setup enables you to:

- Build Docker images for GPU training on GCP
- Push images to Google Artifact Registry
- Run training jobs on Vertex AI with NVIDIA T4 GPUs
- Handle architecture compatibility (linux/amd64 for GCP)

**Key Challenge Solved:** When building Docker images on Apple Silicon (M1/M2/M3 Macs), images are built for ARM64 by default, but GCP requires x86_64/amd64 architecture. This guide shows how to build for the correct platform.

## Prerequisites

- Google Cloud Platform account with billing enabled
- Increase quota for 'Custom model training Nvidia T4 GPUs per region (check page IAM & Admin / Quotas & System Limits)
- `gcloud` CLI installed and authenticated
- Docker Desktop installed (with buildx support)
- Project with the following APIs enabled:
  - Vertex AI API
  - Artifact Registry API

## Initial Setup

### Step 1: Configure GCP Settings

Set your GCP region and project configuration:

```bash
# Set the region (us-central1 is recommended for GPU availability)
gcloud config set ai/region us-central1

# Set your quota/billing project
gcloud config set billing/quota_project YOUR_PROJECT_ID

# Verify configuration
gcloud config list
```

**Explanation:** This sets the default region for AI Platform operations and ensures billing is configured correctly.

### Step 2: Enable Required Services

Enable the necessary GCP APIs:

```bash
# Enable Artifact Registry (for storing Docker images)
gcloud services enable artifactregistry.googleapis.com

# Enable AI Platform / Vertex AI (for running custom jobs)
gcloud services enable aiplatform.googleapis.com

# Verify services are enabled
gcloud services list --enabled | grep -E "aiplatform|artifactregistry"
```

**Explanation:** Artifact Registry stores your Docker images, and Vertex AI runs your training jobs. Both must be enabled before proceeding.

### Step 3: Create Docker Repository

Create a repository in Artifact Registry to store your Docker images:

```bash
# Create the repository in your chosen region
gcloud artifacts repositories create bert-symptom-ner \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for BERT symptom NER training and inference"
```

**Explanation:** This creates a private Docker registry where your images will be stored. The repository name (`bert-symptom-ner`) and location (`us-central1`) must match your environment variables.

### Step 4: Set Environment Variables

Set up your environment variables for easy reference:

```bash
# Replace with your actual project ID
export PROJECT_ID=YOUR_PROJECT_ID

# Region must match where you created the repository
export REGION=us-central1

# Image URI format: REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/IMAGE_NAME:TAG
export IMAGE_URI=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/bert-symptom-ner/train:latest
```

**Explanation:**

- `PROJECT_ID`: Your GCP project ID
- `REGION`: Must match the region where you created the Artifact Registry repository
- `IMAGE_URI`: Full path to your Docker image in Artifact Registry

**Important:** All three variables must use the same region (e.g., all `us-central1` or all `southamerica-east1`).

## Docker Image Setup

### Step 5: Authenticate Docker with Artifact Registry

Configure Docker to authenticate with Google Artifact Registry:

```bash
# Authenticate for your region
gcloud auth configure-docker us-central1-docker.pkg.dev
```

**Explanation:** This allows Docker to push/pull images from your private Artifact Registry. You'll need to do this for each region you use.

### Step 6: Set Up Docker Buildx (Multi-Platform Builds)

Configure Docker buildx for building images for different architectures:

```bash
# Check if buildx is available
docker buildx version

# Create a multi-platform builder instance
docker buildx create --use --name multiarch-builder

# Bootstrap the builder (downloads necessary components)
docker buildx inspect --bootstrap
```

**Explanation:** Buildx enables building Docker images for different CPU architectures. This is essential when building on Apple Silicon for GCP (which requires linux/amd64).

**Expected Output:** You should see `linux/amd64` in the list of supported platforms.

### Step 7: Navigate to Project Directory

Ensure you're in the correct directory:

```bash
cd /path/to/your/bert_symptom_ner
```

**Explanation:** The Docker build context needs to be in your project root where `Dockerfile.train` and `requirements.txt` are located.

## Building and Pushing the Image

### Step 8: Build and Push Docker Image

Build the image for linux/amd64 architecture and push it to Artifact Registry:

```bash
# Build for linux/amd64 (required for GCP) and push directly
docker buildx build --platform linux/amd64 \
  -f Dockerfile.train \
  -t $IMAGE_URI \
  --push .
```

**Explanation:**

- `--platform linux/amd64`: Builds for x86_64 architecture (required by GCP)
- `-f Dockerfile.train`: Specifies the Dockerfile to use
- `-t $IMAGE_URI`: Tags the image with your Artifact Registry URI
- `--push .`: Pushes the image directly to the registry after building
- `.`: Build context (current directory)

**Important Notes:**

- This build may take 10-20 minutes on Apple Silicon due to emulation
- The image will be ~4-5GB in size
- All layers will be pushed to Artifact Registry

**What Happens:**

1. Docker pulls the base CUDA image
2. Installs system dependencies
3. Installs PyTorch with CUDA support
4. Installs Python dependencies from `requirements.txt`
5. Copies your project code
6. Pushes all layers to Artifact Registry

### Step 9: Verify Image Push

Confirm the image was successfully pushed:

```bash
# List images in your repository
gcloud artifacts docker images list us-central1-docker.pkg.dev/YOUR_PROJECT_ID/bert-symptom-ner/train
```

**Explanation:** You should see multiple entries:

- **Manifest** (~1-2KB): Image metadata
- **Config layer** (small): Image configuration
- **Image layer** (~4-5GB): Actual image data with all dependencies

This is normal - Docker images are stored as multiple artifacts that combine to form the complete image.

## Creating and Running Custom Jobs

### Step 10: Create Custom Job

Create a Vertex AI Custom Job to run your training:

```bash
# Ensure environment variables are set
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=us-central1
export IMAGE_URI=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/bert-symptom-ner/train:latest

# Create the custom job
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=bert-ner-train-t4 \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=$IMAGE_URI
```

**Explanation:**

- `--region`: Must match your repository region
- `--display-name`: Human-readable name for the job
- `--worker-pool-spec`: Defines the compute resources:
  - `machine-type=n1-standard-8`: 8 vCPUs, 30GB RAM
  - `accelerator-type=NVIDIA_TESLA_T4`: GPU type
  - `accelerator-count=1`: Number of GPUs
  - `replica-count=1`: Number of worker instances
  - `container-image-uri`: Your Docker image URI

**Expected Output:** You'll receive a job ID like:

```
CustomJob [projects/XXXXX/locations/us-central1/customJobs/YYYYY] is submitted successfully.
```

### Step 11: Monitor Job Status

Stream logs to see real-time training progress:

```bash
# Replace JOB_ID with the actual ID from Step 10
gcloud ai custom-jobs stream-logs projects/YOUR_PROJECT_NUMBER/locations/us-central1/customJobs/JOB_ID
```

**Explanation:** This streams logs from your container in real-time. You'll see:

- Provisioning messages
- Container startup
- Your training script output (`7_trainer_gcp.py`)

**Alternative:** Check status without streaming:

```bash
gcloud ai custom-jobs describe projects/YOUR_PROJECT_NUMBER/locations/us-central1/customJobs/JOB_ID
```

**Job Statuses:**

- **Pending**: Waiting for resources (normal, takes 2-5 minutes)
- **Running**: Job is executing
- **Succeeded**: Training completed successfully
- **Failed**: Error occurred (check logs)

## Troubleshooting

### Issue: "exec format error"

**Problem:** Docker image was built for wrong architecture (ARM64 instead of amd64).

**Solution:** Rebuild with `--platform linux/amd64` flag:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.train -t $IMAGE_URI --push .
```

### Issue: "Repository not found"

**Problem:** Repository location doesn't match IMAGE_URI region.

**Solution:** Ensure repository and IMAGE_URI use the same region:

- Repository created in `us-central1` → IMAGE_URI must use `us-central1-docker.pkg.dev`
- Check: `gcloud artifacts repositories list --location=us-central1`

### Issue: "Permission denied" when pushing

**Problem:** Docker not authenticated with Artifact Registry.

**Solution:** Run authentication:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Issue: Job stays in "Pending" for >10 minutes

**Possible Causes:**

- GPU quota exceeded (check quotas in GCP Console)
- Region doesn't have available GPUs
- Billing not enabled

**Solution:**

- Check quotas: GCP Console → IAM & Admin → Quotas
- Try a different region
- Verify billing is enabled

### Issue: Build fails with "no space left on device"

**Problem:** Docker build context or cache is too large.

**Solution:** Clean up Docker:

```bash
docker system prune -a
docker buildx prune
```

## Quick Reference: Complete Command Sequence

```bash
# 1. Setup
gcloud config set ai/region us-central1
gcloud config set billing/quota_project YOUR_PROJECT_ID
gcloud services enable artifactregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com

# 2. Create repository
gcloud artifacts repositories create bert-symptom-ner \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for BERT symptom NER training"

# 3. Set variables
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=us-central1
export IMAGE_URI=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/bert-symptom-ner/train:latest

# 4. Authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# 5. Setup buildx
docker buildx create --use --name multiarch-builder
docker buildx inspect --bootstrap

# 6. Build and push
cd /path/to/bert_symptom_ner
docker buildx build --platform linux/amd64 \
  -f Dockerfile.train \
  -t $IMAGE_URI \
  --push .

# 7. Create job
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=bert-ner-train-t4 \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=$IMAGE_URI

# 8. Stream logs (replace JOB_ID)
gcloud ai custom-jobs stream-logs projects/YOUR_PROJECT_NUMBER/locations/us-central1/customJobs/JOB_ID
```

## Additional Resources

- [Vertex AI Custom Jobs Documentation](https://cloud.google.com/vertex-ai/docs/training/create-custom-job)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [GCP GPU Availability](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)

## Notes

- **Cost Considerations:** GPU instances are billed per minute. Monitor usage in GCP Console.
- **Image Size:** Large images (~5GB) take longer to pull. Consider optimizing Dockerfile layers.
- **Region Selection:** Choose regions with GPU availability and lower latency for your use case.
- **Security:** Artifact Registry repositories are private by default. Ensure proper IAM permissions.

## GCP CLI Commands

```bash

# login to google

gcloud auth login

# set region

gcloud config set ai/region {REGION}

# set quota project

gcloud config set billing/quota_project ai-project-482122

# Check the gcloud configuration

gcloud config list

# Verify enabled services

gcloud services list --enabled

gcloud services list --enabled | grep -E "aiplatform|artifactregistry|cloudbuild"


# Enable Artifact Registry (Docker images)

gcloud services enable artifactregistry.googleapis.com

# Enable Cloud Build (image builds)

gcloud services enable cloudbuild.googleapis.com

# List available accelerator types (GPUs)

gcloud compute accelerator-types list \
  --filter="zone:(southamerica-east1)" \
  --format="table(name, zone)"

# COMMANDS FOR STEP 3

# Create a docker repository in the Artifacts Registry

gcloud artifacts repositories create bert-symptom-ner \
  --repository-format=docker \
  --location=southamerica-east1 \
  --description="Docker images for BERT symptom NER training and inference"


# Authenticate Docker with Artifact Registry. This allows Docker on your Mac to push images.

gcloud auth configure-docker southamerica-east1-docker.pkg.dev

# List current artifact repositories

gcloud artifacts repositories list

# Delete a repository

gcloud artifacts repositories delete bert-symptom-ner \
  --location=us-central1 \
  --quiet
```
# End of Selection
```

```

Summary biobert all parameters hyperparam config 0.
{
  "model_name": "dmis-lab/biobert-base-cased-v1.1",
  "training_time_minutes": 10.12,
  "hyperparameters": {
    "model_name": "dmis-lab/biobert-base-cased-v1.1",
    "dataset_repo": "Rogarcia18/symptoms_ner_v00_biobert",
    "epoch": 30,
    "lr": 0.00003,
    "batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "push_to_hub": false
  },
  "validation_metrics": {
    "f1": 0.0059026069847516,
    "precision": 0.00526315789473684,
    "recall": 0.00671892497200448,
    "accuracy": 0.537882932166302
  },
  "test_metrics": {
    "f1": 0.00233918128654971,
    "precision": 0.00200883889112093,
    "recall": 0.00279955207166853,
    "accuracy": 0.529072883172562
  }
}

  "validation_metrics": {
    "f1": 0.005902606984751598,
    "precision": 0.005263157894736842,
    "recall": 0.006718924972004479,
    "accuracy": 0.537882932166302
  },
  "test_metrics": {
    "f1": 0.002339181286549707,
    "precision": 0.002008838891120932,
    "recall": 0.002799552071668533,
    "accuracy": 0.5290728831725616
  }
}