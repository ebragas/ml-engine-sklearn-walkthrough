#!/usr/bin/env bash
MODEL_NAME="iris_pipeline_v1"

# Train model
echo "Training model..."
python train.py

# Setup a new GCS bucket
echo "Setting up model bucket..."
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION=us-central1

gsutil mb -l $REGION gs://$BUCKET_NAME

# Store exported model in bucket
# gsutil cp ./model.pkl gs://$BUCKET_NAME/$MODEL_NAME/model.pkl

# Test with local prediction
echo "Testing model locally..."
MODEL_DIR="gs://$BUCKET_NAME/$MODEL_NAME/"
INPUT_FILE="input.json"
FRAMEWORK="SCIKIT_LEARN"

gcloud ml-engine local predict --model-dir=$MODEL_DIR \
    --json-instances $INPUT_FILE \
    --framework $FRAMEWORK

# Create model in ml-engine
echo "Creating ML-Engine model..."
gcloud ml-engine models create $MODEL_NAME

VERSION_NAME="v01"

echo "Creating model version..."
gcloud ml-engine versions create $VERSION_NAME \
    --model $MODEL_NAME \
    --origin $MODEL_DIR \
    --runtime-version=1.12 \
    --framework $FRAMEWORK \
    --python-version=3.5

# Testing online prediction
echo "Testing online prediction..."

gcloud ml-engine predict --model $MODEL_NAME \
    --version $VERSION_NAME \
    --json-instances $INPUT_FILE
