PYTHON := python
TORCHSERVE := torchserve
MODEL_ARCHIVER := torch-model-archiver

PROJECT_ROOT := .
LIGHTNING_DIR := $(PROJECT_ROOT)/lightning
SERVE_DIR := $(PROJECT_ROOT)/serve
MODEL_STORE := $(SERVE_DIR)/model_store

MODEL_NAME := mnist_classifier
HANDLER_FILE := $(SERVE_DIR)/handler.py
CHECKPOINT_PATH := $(LIGHTNING_DIR)/checkpoints/mnist_classifier/model.ckpt
MAR_FILE := $(MODEL_STORE)/$(MODEL_NAME).mar

CONFIG_FILE := $(SERVE_DIR)/config.properties

.PHONY: all package serve-api serve-stop serve-test serve-clean

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
