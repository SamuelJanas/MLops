#!/bin/bash
# Helper script for EC2 deployment after instance is created
# Usage: ./setup_ec2.sh PUBLIC_IP

if [ -z "$1" ]; then
    echo "Usage: ./setup_ec2.sh PUBLIC_IP"
    exit 1
fi

PUBLIC_IP=$1
KEY_FILE="mlops-key.pem"

echo "Setting up MNIST Classifier on EC2 instance: $PUBLIC_IP"
echo "============================================================"

# Wait for instance to be ready
echo "Waiting 30 seconds for instance initialization..."
sleep 30

# Upload files
echo "Uploading deployment files..."
scp -i $KEY_FILE -r ../deploy ubuntu@$PUBLIC_IP:/home/ubuntu/
scp -i $KEY_FILE -r ../lightning ubuntu@$PUBLIC_IP:/home/ubuntu/deploy/
scp -i $KEY_FILE -r ../checkpoints ubuntu@$PUBLIC_IP:/home/ubuntu/deploy/

# Build and run on EC2
echo "Building and starting Docker container on EC2..."
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP << 'EOF'
    cd /home/ubuntu/deploy
    sudo docker build -t mnist-classifier .
    sudo docker run -d -p 8000:8000 --name mnist-api mnist-classifier
    echo "Waiting for container to start..."
    sleep 10
    curl http://localhost:8000/health
EOF

echo ""
echo "============================================================"
echo "Deployment complete!"
echo "Test the API: http://$PUBLIC_IP:8000/health"
echo "Run test client: python test_client.py http://$PUBLIC_IP:8000"
echo "============================================================"
