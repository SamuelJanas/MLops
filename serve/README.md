# Serve MNIST Classifier with TorchServe

## Setup

**Important**: All commands should be run from the project root directory:

```bash
cd /Users/samuel/Documents/uni/MLops
```

## Quick Start

1. **Package the model**
   ```bash
   make package
   ```
   This creates a `.mar` (Model ARchive) file containing:
   - The Lightning checkpoint (`model.ckpt`)
   - The custom handler (`handler.py`)
   - The model definition (`model.py`)

2. **Start TorchServe**
   ```bash
   make serve-api
   ```
   This starts TorchServe with token authentication disabled for local development.

3. **Test the model** (in a new terminal)
   ```bash
   cd /Users/samuel/Documents/uni/MLops
   make serve-test
   ```
   This sends `sample_digit.png` to the model and returns predictions.

4. **Stop TorchServe**
   ```bash
   make serve-stop
   ```

5. **Clean up** (optional)
   ```bash
   make serve-clean
   ```
   This removes `.mar` files and logs.

## Generate a Test Image

Create a custom test digit image:
```bash
python create_sample_digit.py
```

## Manual Testing

You can also test manually with curl:

```bash
# Test with the sample image
curl -X POST "http://localhost:8080/predictions/mnist_classifier" \
  -T sample_digit.png \
  -H "Content-Type: application/octet-stream"

# Test with a custom image
curl -X POST "http://localhost:8080/predictions/mnist_classifier" \
  -T path/to/your/image.png \
  -H "Content-Type: application/octet-stream"
```

Expected response:
```json
[{"prediction": 7, "probabilities": [0.001, 0.002, ..., 0.95, ...]}]
```

## Makefile Targets

- `make package` - Package the Lightning model into a .mar file
- `make serve-api` - Start TorchServe with the model
- `make serve-test` - Send a test prediction request
- `make serve-stop` - Stop TorchServe
- `make serve-clean` - Remove .mar files and logs
- `make all` - Package and start (equivalent to `make package serve-api`)
