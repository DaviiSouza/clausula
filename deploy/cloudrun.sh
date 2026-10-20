#!/bin/sh
# Build in Cloud Build and deploy to Cloud Run: sh deploy/cloudrun.sh
set -e
PROJECT=${PROJECT:-$(gcloud config get-value project)}
REGION=${REGION:-us-central1}
BUILD=$(sh deploy/assemble.sh)
gcloud run deploy clausula \
  --project "$PROJECT" --region "$REGION" --source "$BUILD" \
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 1 \
  --timeout 3600 --session-affinity --allow-unauthenticated --quiet
rm -rf "$BUILD"
