# MNIST Classifier Cloud Deployment

This project deploys a trained MNIST digit classifier to AWS EC2 using FastAPI, Docker, and boto3 for Infrastructure as Code.

## Quick Start

### Prerequisites
- Python 3.10+
- Docker installed locally
- AWS Account with Free Tier access
- AWS CLI configured OR AWS credentials ready

### 1. Setup AWS Credentials

Create `deploy/.env` file (copy from `.env.example`):

```bash
cd deploy
cp .env.example .env
```

Edit `.env` and add your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
INSTANCE_TYPE=t2.micro
KEY_NAME=mlops-key
```

### 2. Install Dependencies

```bash
# For deployment
pip install -r deploy/requirements-deploy.txt

# For testing
pip install -r deploy/requirements-test.txt

# For running the service locally
pip install -r deploy/requirements.txt
```

### 3. Build Docker Image

```bash
make build
```

This builds the Docker image with your MNIST classifier model.

### 4. Test Locally (Recommended)

```bash
make test-local
```

This will:
- Start the Docker container locally on port 8000
- Run automated tests
- Clean up the container

Or test manually:

```bash
# Start container
docker run -d -p 8000:8000 --name mnist-api mnist-classifier:latest

# Test endpoints
curl http://localhost:8000/health

# Test with Python client
python deploy/test_client.py http://localhost:8000

# Stop container
docker stop mnist-api && docker rm mnist-api
```

### 5. Deploy to AWS EC2

```bash
make deploy
```

This will:
1. Create EC2 key pair (saved as `mlops-key.pem`)
2. Create security group with ports 22 (SSH) and 8000 (API) open
3. Launch a t2.micro instance (Free Tier eligible)
4. Install Docker on the instance
5. Display deployment information and next steps

**Important:** Save the `mlops-key.pem` file securely! You need it for SSH access.

### 6. Upload Code and Start Service on EC2

After deployment completes, follow these steps:

```bash
# 1. Upload deployment files to EC2 (replace PUBLIC_IP with your instance IP)
scp -i deploy/mlops-key.pem -r deploy ubuntu@PUBLIC_IP:/home/ubuntu/
scp -i deploy/mlops-key.pem -r lightning ubuntu@PUBLIC_IP:/home/ubuntu/deploy/
scp -i deploy/mlops-key.pem -r checkpoints ubuntu@PUBLIC_IP:/home/ubuntu/deploy/

# 2. SSH into the instance
ssh -i deploy/mlops-key.pem ubuntu@PUBLIC_IP

# 3. Build and run Docker container on EC2
cd /home/ubuntu/deploy
sudo docker build -t mnist-classifier .
sudo docker run -d -p 8000:8000 --name mnist-api mnist-classifier

# 4. Verify it's running
curl http://localhost:8000/health

# Exit SSH
exit
```

### 7. Test Remote Deployment

```bash
# Test from your local machine (replace PUBLIC_IP)
python deploy/test_client.py http://PUBLIC_IP:8000

# Or use curl
curl http://PUBLIC_IP:8000/health
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Digit
```bash
POST /predict
Content-Type: multipart/form-data
```

Send an image file (PNG, JPEG) containing a handwritten digit.

Response:
```json
{
  "prediction": 7,
  "probabilities": [0.0001, 0.0002, 0.0003, 0.0001, 0.0002, 0.0001, 0.0003, 0.9985, 0.0001, 0.0001]
}
```

Example with Python:

```python
import requests

# Test with an image file
with open('my_digit.png', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://PUBLIC_IP:8000/predict', files=files)
    print(response.json())
```

## Management Commands

### List Running Instances
```bash
make list-instances
```

### Terminate Instance
```bash
make terminate INSTANCE_ID=i-xxxxxxxxxxxxx
```

### Clean Local Deployment Artifacts
```bash
make clean-deploy
```

## Project Structure

```
deploy/
├── app.py                    # FastAPI application
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies for the API
├── requirements-deploy.txt  # Dependencies for deployment scripts
├── requirements-test.txt    # Dependencies for testing
├── deploy_ec2.py           # AWS EC2 deployment script (boto3)
├── test_client.py          # Client for testing the API
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Cost Considerations

This deployment uses **AWS Free Tier** resources:
- **t2.micro** instance (750 hours/month free for 12 months)
- Standard networking (15GB data transfer out/month free)

**Important:** 
- Remember to terminate your instance when not in use
- Monitor your AWS Free Tier usage in the AWS Console
- Set up billing alerts to avoid unexpected charges

## Troubleshooting

### Connection Timeout
- Check security group allows inbound traffic on port 8000
- Verify instance is running: `make list-instances`
- Check Docker container is running: `sudo docker ps`

### Model Not Loading
- Verify checkpoint file exists in the container
- Check Docker logs: `sudo docker logs mnist-api`
- Ensure model path is correct in environment variables

### SSH Connection Refused
- Wait 2-3 minutes after instance creation for initialization
- Verify you're using the correct key file
- Check security group allows port 22 from your IP

### Docker Build Fails
- Ensure you have enough disk space
- Check all required files are present (model checkpoint, Python files)
- Verify Docker is installed: `docker --version`

## Security Notes

1. **Never commit** `.env` or `*.pem` files to Git
2. Restrict security group to your IP when possible
3. Use IAM roles instead of access keys for production
4. Rotate AWS credentials regularly
5. Terminate instances when not in use

## License

For educational purposes - MLOps Assignment
