# Build script for Docker image
# This script helps build the Docker image with proper context

# Build from project root to include all necessary files
cd ..
docker build -f deploy/Dockerfile -t mnist-classifier:latest .
cd deploy

echo "Docker image built successfully!"
echo "Test locally with: make test-local"
