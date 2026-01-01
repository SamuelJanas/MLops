PYTHON := python
TORCHSERVE := torchserve
MODEL_ARCHIVER := torch-model-archiver

PROJECT_ROOT := .
LIGHTNING_DIR := $(PROJECT_ROOT)/lightning
SERVE_DIR := $(PROJECT_ROOT)/serve
DEPLOY_DIR := $(PROJECT_ROOT)/deploy
MODEL_STORE := $(SERVE_DIR)/model_store

MODEL_NAME := mnist_classifier
HANDLER_FILE := $(SERVE_DIR)/handler.py
CHECKPOINT_PATH := $(LIGHTNING_DIR)/checkpoints/mnist_classifier/model.ckpt
MAR_FILE := $(MODEL_STORE)/$(MODEL_NAME).mar

CONFIG_FILE := $(SERVE_DIR)/config.properties

.PHONY: all package serve-api serve-stop serve-test serve-clean build deploy test-local clean-deploy list-instances terminate validate deploy-clean

all: package serve-api

package: $(MAR_FILE)

$(MODEL_STORE):
	mkdir -p $(MODEL_STORE)

$(MAR_FILE): $(MODEL_STORE) $(HANDLER_FILE) $(CHECKPOINT_PATH)
	# Package the Lightning checkpoint and custom handler into a .mar
	$(MODEL_ARCHIVER) \
		--model-name $(MODEL_NAME) \
		--version 1.0 \
		--serialized-file $(CHECKPOINT_PATH) \
		--handler $(HANDLER_FILE) \
		--export-path $(MODEL_STORE) \
		--extra-files "$(LIGHTNING_DIR)/model.py" \
		--force

serve-api: package
	# Start TorchServe with our model (disable token auth for local dev)
	TS_DISABLE_TOKEN_AUTHORIZATION=true $(TORCHSERVE) \
		--start \
		--model-store $(MODEL_STORE) \
		--models $(MODEL_NAME)=$(MODEL_NAME).mar \
		--ts-config $(CONFIG_FILE) \
		--disable-token-auth \
		--ncs

serve-stop:
	$(TORCHSERVE) --stop

serve-test:
	curl -X POST "http://localhost:8080/predictions/$(MODEL_NAME)" \
		-T $(SERVE_DIR)/sample_digit.png \
		-H "Content-Type: application/octet-stream"

serve-clean:
	rm -rf $(MODEL_STORE)/*.mar
	rm -rf $(SERVE_DIR)/logs/

# Cloud Deployment Commands

validate:
	@echo "Running pre-deployment validation..."
	cd $(DEPLOY_DIR) && $(PYTHON) validate.py

build:
	@echo "Building Docker image for deployment..."
	docker build -f $(DEPLOY_DIR)/Dockerfile -t mnist-classifier:latest .
	@echo "Docker image built successfully!"

deploy:
	@echo "Deploying to AWS EC2..."
	@echo "This will automatically set up everything on EC2"
	cd $(DEPLOY_DIR) && $(PYTHON) deploy_ec2.py
	@echo ""
	@echo "Deployment complete! Test with: python deploy/test_client.py http://PUBLIC_IP:8000"

deploy-clean:
	@echo "Cleaning up AWS resources..."
	cd $(DEPLOY_DIR) && $(PYTHON) deploy_ec2.py cleanup
	@echo "All AWS resources cleaned up!"

test-local:
	@echo "Testing local Docker deployment..."
	@echo "Starting container on port 8000..."
	docker run -d --name mnist-test -p 8000:8000 mnist-classifier:latest
	@echo "Waiting for container to start..."
	@ping -n 6 127.0.0.1 > nul
	cd $(DEPLOY_DIR) && $(PYTHON) test_client.py http://localhost:8000
	@echo "Stopping test container..."
	docker stop mnist-test
	docker rm mnist-test

clean-deploy:
	@echo "Cleaning deployment artifacts..."
	docker rmi mnist-classifier:latest || true
	cd $(DEPLOY_DIR) && rm -f .deployment_info *.pem

list-instances:
	@echo "Listing EC2 instances..."
	cd $(DEPLOY_DIR) && $(PYTHON) deploy_ec2.py list

terminate:
	@echo "Terminate instance by ID: make terminate INSTANCE_ID=i-xxxxx"
ifdef INSTANCE_ID
	cd $(DEPLOY_DIR) && $(PYTHON) deploy_ec2.py terminate $(INSTANCE_ID)
else
	@echo "ERROR: Please provide INSTANCE_ID"
	@echo "Usage: make terminate INSTANCE_ID=i-xxxxxxxxxxxxx"
endif
