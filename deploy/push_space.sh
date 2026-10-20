#!/bin/sh
# Upload to Hugging Face Spaces (needs HF PRO for Docker Spaces): sh deploy/push_space.sh
set -e
SPACE=DaviiiiS/clausula
BUILD=$(sh deploy/assemble.sh)
cp deploy/standalone/SPACE_README.md "$BUILD"/README.md
hf repos create "$SPACE" --type space --sdk docker --exist-ok
hf upload "$SPACE" "$BUILD" . --repo-type space --commit-message "Deploy from GitHub $(git rev-parse --short HEAD)"
rm -rf "$BUILD"
