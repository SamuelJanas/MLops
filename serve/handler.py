import io
import json
import base64
from typing import List, Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from ts.torch_handler.base_handler import BaseHandler
except ImportError:
    # Fallback for newer TorchServe versions
    from abc import ABC
    class BaseHandler(ABC):
        def __init__(self):
            self.manifest = None
            self.model = None
            self.device = None
            self.initialized = False


# Import from the model.py file bundled in the MAR archive
from model import MNISTClassifier


class MNISTLightningHandler(BaseHandler):
    """
    TorchServe handler for a PyTorch Lightning MNISTClassifier.
    Expects either:
      - binary image in the request body (PNG/JPEG 28x28 grayscale or RGB),
      - or JSON: {"image": "<base64-encoded-image-bytes>"}.
    Returns:
      {"prediction": int, "probabilities": [float]*10}
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Load model checkpoint specified by model_dir."""
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # TorchServe passes checkpoint via .mar archive;
        # in this example, we assume `model.ckpt` is inside the model archive.
        checkpoint_path = f"{model_dir}/model.ckpt"

        # Load Lightning model
        self.model = MNISTClassifier.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Preprocessing: same as training (normalize with MNIST stats)
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.initialized = True

    def _decode_image(self, data: bytes) -> Image.Image:
        """Decode raw bytes (image file) into a PIL.Image."""
        return Image.open(io.BytesIO(data)).convert("L")

    def preprocess(self, data: List[Any]):
        """
        Data can come in as:
          - raw image bytes (first element in data),
          - or JSON with base64-encoded image.
        """
        images = []
        for row in data:
            body = row.get("body")
            if body is None:
                body = row.get("data")

            if isinstance(body, (bytes, bytearray)):
                img = self._decode_image(body)
            else:
                if isinstance(body, str):
                    body = json.loads(body)

                if "image" in body:
                    # base64-encoded image
                    img_bytes = base64.b64decode(body["image"])
                    img = self._decode_image(img_bytes)
                else:
                    raise ValueError("Request must contain raw image bytes or JSON with 'image' field.")

            tensor = self.transform(img)
            images.append(tensor)

        batch = torch.stack(images, dim=0).to(self.device)  # [B, 1, 28, 28]
        return batch

    def inference(self, inputs: torch.Tensor):
        """Run forward pass and return predictions."""
        with torch.no_grad():
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=1)  # [B, 10]
            conf, pred = probs.max(dim=1)
        return probs.cpu(), pred.cpu()

    def postprocess(self, inference_output):
        probs, preds = inference_output
        results = []
        for i in range(len(preds)):
            results.append(
                {
                    "prediction": int(preds[i].item()),
                    "probabilities": probs[i].tolist(),
                }
            )
        return [json.dumps(results)]
