"""
Test client for MNIST classifier API.
Uses only the requests library as required.
"""
import requests
import numpy as np
from PIL import Image
import io
import sys


def create_sample_digit(digit=7):
    """Create a simple sample digit image for testing."""
    # Create a simple 28x28 image with a digit-like pattern
    img = Image.new('L', (28, 28), color=0)
    pixels = img.load()
    
    # Simple patterns for digits 0-9
    patterns = {
        0: [(i, j) for i in range(8, 20) for j in [8, 19]] + 
           [(i, j) for i in [8, 19] for j in range(8, 20)],
        1: [(i, 14) for i in range(5, 23)],
        7: [(7, i) for i in range(8, 20)] + 
           [(i, 19) for i in range(7, 23)],
    }
    
    pattern = patterns.get(digit, patterns[7])
    for i, j in pattern:
        if 0 <= i < 28 and 0 <= j < 28:
            pixels[j, i] = 255
    
    return img


def test_endpoint(base_url, test_image_path=None):
    """
    Test the deployed MNIST classifier endpoint.
    
    Args:
        base_url: Base URL of the deployed service (e.g., "http://35.123.45.67:8000")
        test_image_path: Optional path to a test image file
    """
    print(f"Testing endpoint: {base_url}")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 3: Prediction
    print("\n3. Testing prediction endpoint...")
    try:
        # Use provided image or create a sample
        if test_image_path:
            with open(test_image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            print("   Creating sample digit image (digit: 7)...")
            img = create_sample_digit(7)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
        
        files = {'file': ('test_image.png', image_bytes, 'image/png')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=10)
        
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Prediction: {result['prediction']}")
            print(f"   Probabilities: {[f'{p:.4f}' for p in result['probabilities']]}")
            print(f"   Confidence: {max(result['probabilities']):.2%}")
        else:
            print(f"   ERROR Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    return True


def main():
    """Main function to run tests."""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <BASE_URL> [IMAGE_PATH]")
        print("Example: python test_client.py http://3.123.45.67:8000")
        print("Example: python test_client.py http://localhost:8000 my_digit.png")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    test_image = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = test_endpoint(base_url, test_image)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
