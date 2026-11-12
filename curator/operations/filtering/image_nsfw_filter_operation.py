import os
from typing import Any, Dict, Optional
from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class ImageNsfwFilterOperation(FilteringOperation):
    """Image NSFW content filter

    Detect and filter NSFW (not safe for work) image content using a pretrained model.
    Based on the Hugging Face nsfw_image_detection model, with a configurable confidence threshold.

    Note: this operation requires the transformers and torch libraries.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter NSFW image content with an AI model",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image",
                "description": "Image field name"
            },
            "model_path": {
                "type": "string",
                "required": False,
                "default": "Falconsai/nsfw_image_detection",
                "description": "Hugging Face model path"
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "NSFW detection threshold; images above it are filtered"
            },
            "device_preference": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Device selection: auto (prefers NPU > CUDA > CPU), npu, npu:0, cuda, cuda:0, cpu, etc.; a device index may be specified"
            }
        },
        "example": {
            "image_key": "image",
            "threshold": 0.5
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self._model = None
        self._processor = None
        self._device = None

    def _get_available_device(self) -> str:
        """Get an available compute device, with priority NPU > CUDA > CPU

        Supports formats with a device index, such as npu:0 and cuda:1
        """
        preference = self.params.get('device_preference', 'auto')
        if preference.startswith('npu'):
            try:
                import torch_npu
                if torch_npu.npu.is_available():
                    return preference
            except ImportError:
                pass
            raise RuntimeError(f"NPU device requested but not available: {preference}")
        elif preference.startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                return preference
            raise RuntimeError(f"CUDA device requested but not available: {preference}")
        elif preference == 'cpu':
            return 'cpu'
        # auto
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return 'npu'
        except ImportError:
            pass
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForImageClassification, AutoProcessor
                model_path = self.params.get('model_path', 'Falconsai/nsfw_image_detection')
                self._device = self._get_available_device()
                self._model = AutoModelForImageClassification.from_pretrained(model_path)
                self._model = self._model.to(self._device)
                self._model.eval()
                self._processor = AutoProcessor.from_pretrained(model_path)
            except ImportError:
                raise ImportError("transformers library is required for ImageNsfwFilterOperation. "
                                "Install it with: pip install transformers")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        image_key = self.params.get('image_key', 'image')
        threshold = self.params.get('threshold', 0.5)

        def nsfw_filter(row):
            image_data = row.get(image_key)
            if image_data is None:
                return False

            try:
                from PIL import Image
                import torch
                import torch.nn.functional as F

                self._load_model()

                # Determine whether the input is a file path string or already loaded image data (ndarray)
                if isinstance(image_data, str):
                    # Path string: read from a file
                    if not os.path.exists(image_data):
                        return False
                    image = Image.open(image_data)
                else:
                    # Already an ndarray: convert to a PIL Image
                    import numpy as np
                    image = Image.fromarray(image_data)

                with torch.no_grad():
                    inputs = self._processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self._model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    outputs = self._model(**inputs)
                    logits = outputs.logits[0]
                    probs = F.softmax(logits, dim=0)

                # probs[1] is NSFW probability
                return probs[1] < threshold

            except Exception:
                return False

        return dataset.filter(nsfw_filter)
