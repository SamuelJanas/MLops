#!/usr/bin/env python3
"""Fix security group to ensure port 8000 is open."""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SECURITY_GROUP_NAME = "mlops-mnist-sg"

ec2_client = boto3.client('ec2', region_name=AWS_REGION)

# Get the security group
response = ec2_client.describe_security_groups(GroupNames=[SECURITY_GROUP_NAME])
sg = response['SecurityGroups'][0]
sg_id = sg['GroupId']

print(f"Security Group: {sg_id}")
print(f"Current inbound rules:")
for rule in sg['IpPermissions']:
    print(f"  Port {rule.get('FromPort', 'N/A')}: {rule}")

# Check if port 8000 is open
has_8000 = any(rule.get('FromPort') == 8000 for rule in sg['IpPermissions'])

if not has_8000:
    print("\nPort 8000 is NOT open. Adding rule...")
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 8000,
                'ToPort': 8000,
                'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'FastAPI access'}]
            }
        ]
    )
    print("Port 8000 opened successfully!")
else:
    print("\nPort 8000 is already open.")

print("\nTry testing again: python test_client.py http://3.255.83.61:8000")
