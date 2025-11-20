from pathlib import Path

from torchvision import datasets, transforms


def main():
    out_path = Path(__file__).parent / "sample_digit.png"

    transform = transforms.ToTensor()
    mnist_test = datasets.MNIST(root="../lightning/data", train=False, download=True, transform=transform)
    img_tensor, label = mnist_test[0]
    img = transforms.ToPILImage()(img_tensor)

    img.save(out_path)
    print(f"Saved {out_path} with label {label}")


if __name__ == "__main__":
    main()
