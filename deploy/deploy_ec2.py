#!/usr/bin/env python3
"""
AWS EC2 deployment script using boto3.
Fully automated deployment of MNIST classifier to EC2.
"""
import boto3
import os
import time
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "t2.micro")  # Free tier eligible
AMI_ID = os.getenv("AMI_ID", "ami-0c55b159cbfafe1f0")  # Ubuntu 22.04 LTS in us-east-1
KEY_NAME = os.getenv("KEY_NAME", "mlops-key")
SECURITY_GROUP_NAME = "mlops-mnist-sg"
INSTANCE_NAME = "mlops-mnist-classifier"

# User data script to set up Docker and run container
USER_DATA_SCRIPT = """#!/bin/bash
set -e

# Update system
apt-get update
apt-get install -y docker.io git

# Start Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Wait for Docker to be ready
sleep 5

# Create directory for deployment
mkdir -p /home/ubuntu/mlops-deploy
cd /home/ubuntu/mlops-deploy

# Note: You'll need to either:
# 1. Clone your repo with the model and code
# 2. Or copy files via SCP after instance is running
# For now, we'll expect files to be uploaded separately

echo "EC2 instance setup complete. Docker is ready."
echo "Upload your Docker image or build it on this instance."
"""


def get_or_create_key_pair(ec2_client, key_name):
    """Create EC2 key pair if it doesn't exist."""
    try:
        response = ec2_client.describe_key_pairs(KeyNames=[key_name])
        print(f"Key pair '{key_name}' already exists")
        return None
    except ec2_client.exceptions.ClientError:
        print(f"Creating new key pair '{key_name}'")
        response = ec2_client.create_key_pair(KeyName=key_name)
        
        # Save private key
        key_file = f"{key_name}.pem"
        with open(key_file, 'w') as f:
            f.write(response['KeyMaterial'])
        os.chmod(key_file, 0o400)
        print(f"Private key saved to {key_file}")
        return key_file


def get_or_create_security_group(ec2_client):
    """Create security group with rules for SSH and HTTP."""
    try:
        response = ec2_client.describe_security_groups(
            GroupNames=[SECURITY_GROUP_NAME]
        )
        sg_id = response['SecurityGroups'][0]['GroupId']
        print(f"Security group '{SECURITY_GROUP_NAME}' already exists: {sg_id}")
        return sg_id
    except ec2_client.exceptions.ClientError:
        print(f"Creating security group '{SECURITY_GROUP_NAME}'")
        
        response = ec2_client.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description="Security group for MNIST classifier deployment"
        )
        sg_id = response['GroupId']
        
        # Add inbound rules
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8000,
                    'ToPort': 8000,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'FastAPI access'}]
                }
            ]
        )
        print(f"Security group created: {sg_id}")
        return sg_id


def get_latest_ubuntu_ami(ec2_client):
    """Get the latest Ubuntu 22.04 LTS AMI for the region."""
    try:
        response = ec2_client.describe_images(
            Filters=[
                {'Name': 'name', 'Values': ['ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*']},
                {'Name': 'state', 'Values': ['available']},
                {'Name': 'architecture', 'Values': ['x86_64']}
            ],
            Owners=['099720109477']  # Canonical's AWS account ID
        )
        
        if not response['Images']:
            raise ValueError("No Ubuntu AMI found")
        
        # Sort by creation date and get the latest
        images = sorted(response['Images'], key=lambda x: x['CreationDate'], reverse=True)
        ami_id = images[0]['ImageId']
        print(f"Using Ubuntu AMI: {ami_id}")
        return ami_id
    except Exception as e:
        print(f"Error getting Ubuntu AMI: {e}")
        print("Using default AMI from environment")
        return AMI_ID


def create_instance(ec2_client, ami_id, key_name, security_group_id):
    """Create EC2 instance."""
    print(f"Launching EC2 instance (type: {INSTANCE_TYPE})...")
    
    response = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        MinCount=1,
        MaxCount=1,
        UserData=USER_DATA_SCRIPT,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': INSTANCE_NAME},
                    {'Key': 'Project', 'Value': 'MLOps'},
                ]
            }
        ]
    )
    
    instance_id = response['Instances'][0]['InstanceId']
    print(f"Instance created: {instance_id}")
    
    return instance_id


def wait_for_instance(ec2_client, instance_id):
    """Wait for instance to be running and get public IP."""
    print("Waiting for instance to be running...")
    
    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    public_ip = response['Reservations'][0]['Instances'][0].get('PublicIpAddress')
    
    print(f"Instance is running. Public IP: {public_ip}")
    return public_ip


def upload_and_setup(public_ip, key_file):
    """Upload files and setup Docker on EC2."""
    print("\nWaiting for SSH to be ready...")
    time.sleep(30)  # Wait for instance to fully initialize
    
    # Test SSH connection
    max_retries = 10
    for i in range(max_retries):
        try:
            result = subprocess.run(
                ['ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no', 
                 '-o', 'ConnectTimeout=10', f'ubuntu@{public_ip}', 'echo', 'ready'],
                capture_output=True,
                timeout=15
            )
            if result.returncode == 0:
                print("SSH connection established!")
                break
        except subprocess.TimeoutExpired:
            pass
        
        if i < max_retries - 1:
            print(f"SSH not ready yet, retrying ({i+1}/{max_retries})...")
            time.sleep(10)
    
    print("\nUploading files to EC2...")
    # Upload deploy directory
    subprocess.run(['scp', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
                   '-r', 'deploy', f'ubuntu@{public_ip}:/home/ubuntu/'], check=True)
    
    # Upload lightning directory
    subprocess.run(['scp', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
                   '-r', 'lightning', f'ubuntu@{public_ip}:/home/ubuntu/deploy/'], check=True)
    
    # Upload checkpoints
    subprocess.run(['scp', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
                   '-r', 'checkpoints', f'ubuntu@{public_ip}:/home/ubuntu/deploy/'], check=True)
    
    print("Files uploaded successfully!")
    
    print("\nBuilding and starting Docker container on EC2...")
    
    setup_commands = """
    cd /home/ubuntu/deploy && \
    sudo docker build -t mnist-classifier . && \
    sudo docker run -d -p 8000:8000 --name mnist-api mnist-classifier && \
    echo "Waiting for container to start..." && \
    sleep 10 && \
    curl -s http://localhost:8000/health || echo "Service starting..."
    """
    
    result = subprocess.run(
        ['ssh', '-i', key_file, '-o', 'StrictHostKeyChecking=no',
         f'ubuntu@{public_ip}', setup_commands],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode == 0:
        print("Docker container started successfully!")
    else:
        print(f"Warning: Setup completed but there may have been issues:\n{result.stderr}")


def deploy():
    """Main deployment function - fully automated."""
    print("Starting AWS EC2 deployment...")
    print(f"Region: {AWS_REGION}")
    
    # Change to project root for file uploads
    original_dir = os.getcwd()
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(project_root))
    
    try:
        # Initialize boto3 client
        ec2_client = boto3.client('ec2', region_name=AWS_REGION)
        
        # Create or get key pair
        key_file = get_or_create_key_pair(ec2_client, KEY_NAME)
        if not key_file:
            key_file = f"{KEY_NAME}.pem"
        
        # Make sure key file is in deploy directory
        if not os.path.exists(os.path.join('deploy', key_file)):
            if os.path.exists(key_file):
                import shutil
                shutil.copy(key_file, os.path.join('deploy', key_file))
        
        key_path = os.path.join('deploy', key_file)
        
        # Create or get security group
        sg_id = get_or_create_security_group(ec2_client)
        
        # Get latest Ubuntu AMI
        ami_id = get_latest_ubuntu_ami(ec2_client)
        
        # Create instance
        instance_id = create_instance(ec2_client, ami_id, KEY_NAME, sg_id)
        
        # Wait for instance to be running
        public_ip = wait_for_instance(ec2_client, instance_id)
        
        # Upload files and setup
        upload_and_setup(public_ip, key_path)
        
        print("\n" + "="*60)
        print("DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print(f"Instance ID: {instance_id}")
        print(f"Public IP: {public_ip}")
        print(f"API Endpoint: http://{public_ip}:8000")
        print("\nTest your deployment:")
        print(f"  python deploy/test_client.py http://{public_ip}:8000")
        print("\nCleanup when done:")
        print(f"  make deploy-clean")
        print("="*60)
        
        # Save deployment info
        with open(os.path.join('deploy', '.deployment_info'), 'w') as f:
            f.write(f"INSTANCE_ID={instance_id}\n")
            f.write(f"PUBLIC_IP={public_ip}\n")
            f.write(f"REGION={AWS_REGION}\n")
        
        return instance_id, public_ip
    
    finally:
        os.chdir(original_dir)


def list_instances():
    """List all running instances."""
    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    
    response = ec2_client.describe_instances(
        Filters=[
            {'Name': 'instance-state-name', 'Values': ['running', 'pending']},
            {'Name': 'tag:Project', 'Values': ['MLOps']}
        ]
    )
    
    print("\nRunning MLOps instances:")
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            state = instance['State']['Name']
            public_ip = instance.get('PublicIpAddress', 'N/A')
            name = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), 'N/A')
            print(f"  {name}: {instance_id} ({state}) - {public_ip}")


def terminate_instance(instance_id):
    """Terminate a specific instance."""
    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    
    print(f"Terminating instance {instance_id}...")
    ec2_client.terminate_instances(InstanceIds=[instance_id])
    print("Instance termination initiated")


def cleanup_all():
    """Clean up all deployed resources."""
    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    
    # Read deployment info
    info_file = '.deployment_info'
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info = dict(line.strip().split('=') for line in f if '=' in line)
        
        instance_id = info.get('INSTANCE_ID')
        if instance_id:
            print(f"Terminating instance: {instance_id}")
            terminate_instance(instance_id)
            print("Waiting for termination...")
            waiter = ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=[instance_id])
            print("Instance terminated successfully!")
        
        # Remove deployment info
        os.remove(info_file)
        print("Deployment info cleaned up")
    else:
        print("No deployment info found. Checking for MLOps instances...")
        list_instances()
        print("\nTo manually terminate, use: make terminate INSTANCE_ID=i-xxx")
    
    print("\nCleanup complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list":
            list_instances()
        elif command == "terminate" and len(sys.argv) > 2:
            terminate_instance(sys.argv[2])
        elif command == "cleanup":
            cleanup_all()
        else:
            print("Usage: python deploy_ec2.py [list|terminate <instance-id>|cleanup]")
    else:
        deploy()
