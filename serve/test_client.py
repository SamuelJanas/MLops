import argparse
import base64
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Test TorchServe MNIST API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/predictions/mnist_classifier",
        help="TorchServe prediction URL",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="sample_digit.png",
        help="Path to a 28x28 digit image",
    )
    args = parser.parse_args()

    try:
        with open(args.image, "rb") as f:
            img_bytes = f.read()
    except FileNotFoundError:
        print(f"Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Example 1: raw bytes
    resp_raw = requests.post(
        args.url,
        data=img_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=10,
    )
    print("Raw bytes response status:", resp_raw.status_code)
    print("Raw bytes response body:", resp_raw.text)

    # Example 2: JSON with base64-encoded image
    payload = {"image": base64.b64encode(img_bytes).decode("utf-8")}
    resp_json = requests.post(args.url, json=payload, timeout=10)
    print("JSON response status:", resp_json.status_code)
    print("JSON response body:", resp_json.text)


if __name__ == "__main__":
    main()
