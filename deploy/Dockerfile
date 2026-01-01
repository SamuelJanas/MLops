FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/lightning /app/checkpoints/mnist_classifier

# Copy application code
COPY deploy/app.py .

# Copy model files (building from project root with: docker build -f deploy/Dockerfile .)
COPY lightning/model.py /app/lightning/model.py
COPY lightning/checkpoints/mnist_classifier/model.ckpt /app/checkpoints/mnist_classifier/epochepoch=08-val_accval_acc=0.9886.ckpt

# Set environment variable for model path
ENV MODEL_PATH=/app/checkpoints/mnist_classifier/epochepoch=08-val_accval_acc=0.9886.ckpt

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
