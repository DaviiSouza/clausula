#!/bin/sh
# Copy the files the all-in-one image needs into a temp folder and print its path.
set -e
BUILD=$(mktemp -d)
cp -r src api app .streamlit pyproject.toml requirements.txt "$BUILD"/
cp deploy/standalone/Dockerfile deploy/standalone/start.sh "$BUILD"/
find "$BUILD" \( -name "__pycache__" -o -name "*.egg-info" \) -prune -exec rm -rf {} +
echo "$BUILD"
