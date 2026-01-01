#!/usr/bin/env python3
"""
Pre-deployment validation script.
Checks that all prerequisites are met before deployment.
"""
import os
import sys
import subprocess


def check_command(command, install_hint):
    """Check if a command is available."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"[OK] {command} is installed")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"[ERROR] {command} not found. {install_hint}")
        return False


def check_file(filepath, hint):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"[OK] {filepath} exists")
        return True
    else:
        print(f"[ERROR] {filepath} not found. {hint}")
        return False


def check_env_file():
    """Check if .env file is configured."""
    env_path = "deploy/.env"
    if not os.path.exists(env_path):
        print(f"[ERROR] deploy/.env not found")
        print("  Copy deploy/.env.example to deploy/.env and add your credentials")
        return False
    
    # Check if it has actual values
    with open(env_path, 'r') as f:
        content = f.read()
        if 'your_access_key_here' in content:
            print("[ERROR] deploy/.env has placeholder values")
            print("  Update with your actual AWS credentials")
            return False
    
    print("[OK] deploy/.env is configured")
    return True


def main():
    """Run all validation checks."""
    print("MLOps Deployment Pre-flight Checks")
    print("=" * 60)
    
    checks = []
    
    # Check Docker
    checks.append(check_command("docker", "Install from https://docker.com"))
    
    # Check Python
    checks.append(check_command("python", "Install Python 3.10+"))
    
    # Check checkpoint file
    checks.append(check_file(
        "checkpoints/mnist_classifier/epochepoch=08-val_accval_acc=0.9886.ckpt",
        "Train the model first or check the path"
    ))
    
    # Check model file
    checks.append(check_file(
        "lightning/model.py",
        "Ensure lightning/model.py exists"
    ))
    
    # Check deployment files
    checks.append(check_file(
        "deploy/app.py",
        "Deployment files missing"
    ))
    
    checks.append(check_file(
        "deploy/Dockerfile",
        "Dockerfile missing"
    ))
    
    checks.append(check_file(
        "deploy/deploy_ec2.py",
        "Deployment script missing"
    ))
    
    # Check .env file
    checks.append(check_env_file())
    
    # Summary
    print("\n" + "=" * 60)
    if all(checks):
        print("All checks passed! You're ready to deploy.")
        print("\nNext steps:")
        print("  1. make build       # Build Docker image")
        print("  2. make test-local  # Test locally")
        print("  3. make deploy      # Deploy to AWS")
        return 0
    else:
        print("Some checks failed. Fix the errors above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
