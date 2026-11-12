from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.device import resolve_torch_device
from ...utils.local_model_paths import local_model_roots_from_params, resolve_local_model_path
from typing import Any, Dict, Optional
import os


class ImageCaptioningOperation(AugmentationOperation):
    """Image caption generation operation

    Generate text descriptions for images using pretrained image captioning or vision-language models.
    Automatically selects a compute device: NPU > CUDA > CPU.

    Note: this operation requires the transformers and torch libraries.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Generate image captions with the BLIP model",
        "params": {
            "image_key": {
                "type": "string",
                "required": False,
                "default": "image_path",
                "description": "Input image field name"
            },
            "fallback_image_keys": {
                "type": "array",
                "required": False,
                "default": ["image", "image_path"],
                "description": "Fallback image fields tried in order when image_key is missing"
            },
            "output_key": {
                "type": "string",
                "required": False,
                "default": "caption",
                "description": "Output caption field name"
            },
            "model_path": {
                "type": "string",
                "required": False,
                "default": "Salesforce/blip-image-captioning-base",
                "description": "Hugging Face model path or local path"
            },
            "local_model_roots": {
                "type": "array",
                "required": False,
                "default": None,
                "description": "Local model search roots; defaults to MODEL_ROOTS and checks /data/llm_models"
            },
            "local_files_only": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Load models only from local files"
            },
            "model_type": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Model type: auto, blip, qwen2_vl, qwen2_5_vl, qwen3_vl"
            },
            "device_preference": {
                "type": "string",
                "required": False,
                "default": "auto",
                "description": "Device priority: auto (prefers NPU > CUDA > CPU), npu, npu:0, cuda, cuda:0, cpu, etc.; a device index may be specified"
            },
            "prompt_text": {
                "type": "string",
                "required": False,
                "default": "a photography of",
                "description": "Prefix prompt for image captions"
            },
            "max_new_tokens": {
                "type": "integer",
                "required": False,
                "default": 64,
                "description": "Maximum new tokens for caption generation"
            },
            "error_key": {
                "type": "string",
                "required": False,
                "default": "caption_error",
                "description": "Error field written when model loading or generation fails; omitted when empty"
            },
            "resolved_model_path_key": {
                "type": "string",
                "required": False,
                "default": "caption_model_path",
                "description": "Field for the resolved model path; omitted when empty"
            }
        },
        "example": {
            "image_key": "image_path",
            "output_key": "caption",
            "device_preference": "auto"
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self._model = None
        self._processor = None
        self._device = None

    def _get_available_device(self) -> str:
        """Get an available compute device, with priority NPU > CUDA > CPU

        Supports formats with a device index, such as npu:0 and cuda:0
        """
        return resolve_torch_device(self.params.get('device_preference', 'auto'))

    def _target_torch_device(self):
        import torch

        return torch.device(self._device or 'cpu')

    def _infer_model_type(self, model_path: str) -> str:
        model_type = str(self.params.get('model_type', 'auto')).lower()
        if model_type != 'auto':
            return model_type

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            detected = str(getattr(config, 'model_type', '')).lower()
            if detected in {'qwen2_vl', 'qwen2_5_vl', 'qwen3_vl'}:
                return detected
            if detected == 'blip-2':
                return 'blip2'
        except Exception:
            pass
        return 'blip'

    def _load_model(self):
        """Lazily load the model and processor"""
        if self._model is None:
            try:
                from transformers import AutoProcessor
                import torch

                model_path = self._resolve_model_path(
                    self.params.get('model_path', 'Salesforce/blip-image-captioning-base')
                )
                model_type = self._infer_model_type(model_path)
                self._device = self._get_available_device()
                local_files_only = bool(self.params.get('local_files_only', True))

                self._processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                if model_type == 'qwen2_vl':
                    from transformers import Qwen2VLForConditionalGeneration

                    self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype='auto',
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                elif model_type == 'qwen2_5_vl':
                    from transformers import Qwen2_5_VLForConditionalGeneration

                    self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype='auto',
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                elif model_type == 'qwen3_vl':
                    from transformers import Qwen3VLForConditionalGeneration

                    self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype='auto',
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                elif model_type == 'blip2':
                    from transformers import Blip2ForConditionalGeneration

                    self._model = Blip2ForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype='auto',
                        local_files_only=local_files_only,
                    )
                else:
                    from transformers import BlipForConditionalGeneration

                    self._model = BlipForConditionalGeneration.from_pretrained(
                        model_path,
                        local_files_only=local_files_only,
                    )

                self._model = self._model.to(self._target_torch_device())
                self._model.eval()
                self._model_type = model_type
                self._resolved_model_path = model_path

            except ImportError as e:
                raise ImportError(
                    "transformers library is required for ImageCaptioningOperation. "
                    "Install it with: pip install transformers"
                ) from e

    def _resolve_model_path(self, model_path: str) -> str:
        roots = local_model_roots_from_params(self.params.get("local_model_roots"))
        return resolve_local_model_path(model_path, roots)


    def _caption_with_qwen_vl(self, raw_image, prompt_text: str) -> str:
        import torch

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': raw_image},
                    {'type': 'text', 'text': prompt_text},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[raw_image],
            return_tensors='pt',
        )
        inputs = inputs.to(self._model.device)
        max_new_tokens = int(self.params.get('max_new_tokens', 64))
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return self._processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _caption_with_standard_model(self, raw_image, prompt_text: str) -> str:
        import torch

        inputs = self._processor(
            raw_image,
            prompt_text,
            return_tensors='pt',
        ).to(self._model.device)
        max_new_tokens = int(self.params.get('max_new_tokens', 64))
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.decode(output[0], skip_special_tokens=True).strip()

    def _caption_image(self, raw_image, prompt_text: str) -> str:
        if self._model_type in {'qwen2_vl', 'qwen2_5_vl', 'qwen3_vl'}:
            return self._caption_with_qwen_vl(raw_image, prompt_text)
        return self._caption_with_standard_model(raw_image, prompt_text)

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Execute the image caption generation operation"""
        image_key = self.params.get('image_key', 'image_path')
        output_key = self.params.get('output_key', 'caption')
        prompt_text = self.params.get('prompt_text', 'a photography of')
        error_key = self.params.get('error_key', 'caption_error')
        resolved_model_path_key = self.params.get('resolved_model_path_key', 'caption_model_path')

        def augment_func(row):
            row = dict(row)
            image_data = self._resolve_image_data(row, image_key)
            if image_data is None:
                return row

            raw_image = None
            try:
                from PIL import Image

                if isinstance(image_data, str):
                    if not os.path.exists(image_data):
                        if error_key:
                            row[error_key] = f"image file not found: {image_data}"
                        return row
                    raw_image = Image.open(image_data).convert('RGB')
                else:
                    raw_image = Image.fromarray(image_data).convert('RGB')

                self._load_model()
                row[output_key] = self._caption_image(raw_image, prompt_text)
                if resolved_model_path_key:
                    row[resolved_model_path_key] = getattr(self, "_resolved_model_path", None)

            except Exception as exc:
                if error_key:
                    row[error_key] = str(exc)
            finally:
                if raw_image is not None:
                    raw_image.close()

            return row

        return dataset.map(augment_func)

    def _resolve_image_data(self, row: dict[str, Any], image_key: str) -> Any:
        if row.get(image_key) is not None:
            return row.get(image_key)
        fallback_keys = self.params.get("fallback_image_keys") or []
        for candidate in fallback_keys:
            if row.get(candidate) is not None:
                return row.get(candidate)
        return None
