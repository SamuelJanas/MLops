"""
Simple usage example for the MNIST Classifier API.
Shows how to use the deployed model for inference.
"""
import requests
import io
from PIL import Image, ImageDraw


def create_digit_image(digit=5):
    """
    Create a simple handwritten-style digit image.
    In practice, you would use real handwritten digit images.
    """
    # Create white background
    img = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw a simple digit pattern
    # This is just for demonstration - use real digit images for actual testing
    if digit == 5:
        # Simple "5" pattern
        draw.rectangle([8, 8, 20, 12], fill=0)
        draw.rectangle([8, 8, 12, 15], fill=0)
        draw.rectangle([8, 13, 20, 17], fill=0)
        draw.rectangle([16, 15, 20, 22], fill=0)
        draw.rectangle([8, 20, 20, 24], fill=0)
    elif digit == 7:
        # Simple "7" pattern
        draw.rectangle([8, 8, 20, 12], fill=0)
        draw.rectangle([16, 10, 20, 24], fill=0)
    else:
        # Default vertical line for other digits
        draw.rectangle([12, 8, 16, 24], fill=0)
    
    return img


def predict_digit(api_url, image):
    """
    Send an image to the API and get prediction.
    
    Args:
        api_url: Base URL of the API (e.g., "http://3.123.45.67:8000")
        image: PIL Image object or path to image file
    
    Returns:
        dict with prediction and probabilities
    """
    # Convert PIL Image to bytes if needed
    if isinstance(image, Image.Image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        files = {'file': ('digit.png', image_bytes, 'image/png')}
    else:
        # Assume it's a file path
        with open(image, 'rb') as f:
            files = {'file': f}
    
    # Make prediction request
    response = requests.post(f"{api_url}/predict", files=files)
    response.raise_for_status()
    
    return response.json()


def main():
    """Example usage of the MNIST Classifier API."""
    # Replace with your deployed API URL
    API_URL = "http://localhost:8000"  # Change to your EC2 public IP
    
    print("MNIST Classifier API - Usage Example")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Checking API health...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.json()}")
    
    # Test 2: Predict a digit
    print("\n2. Predicting a handwritten digit...")
    test_digit = 7
    img = create_digit_image(test_digit)
    
    # Optionally save the test image
    img.save("test_digit.png")
    print(f"   Created test image for digit: {test_digit}")
    
    # Get prediction
    result = predict_digit(API_URL, img)
    print(f"   Predicted digit: {result['prediction']}")
    print(f"   Confidence: {max(result['probabilities']):.2%}")
    print(f"   All probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
    
    # Test 3: Using a file
    print("\n3. Predicting from file...")
    result = predict_digit(API_URL, "test_digit.png")
    print(f"   Predicted digit: {result['prediction']}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nTo use with your own images:")
    print("  result = predict_digit('http://YOUR_IP:8000', 'path/to/digit.png')")


if __name__ == "__main__":
    main()
