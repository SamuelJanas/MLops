import argparse
from pathlib import Path

from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser(description="Create a sample digit image from MNIST dataset")
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=range(10),
        help="Digit to create (0-9). If not specified, uses the first image in the dataset.",
    )
    args = parser.parse_args()

    out_path = Path(__file__).parent / "sample_digit.png"

    transform = transforms.ToTensor()
    mnist_test = datasets.MNIST(root="../lightning/data", train=False, download=True, transform=transform)

    # Find the first image with the requested digit
    if args.digit is not None:
        for idx, (img_tensor, label) in enumerate(mnist_test):
            if label == args.digit:
                break
        else:
            print(f"No image found for digit {args.digit}")
            return
    else:
        img_tensor, label = mnist_test[0]

    img = transforms.ToPILImage()(img_tensor)
    img.save(out_path)
    print(f"Saved {out_path} with label {label}")


if __name__ == "__main__":
    main()
