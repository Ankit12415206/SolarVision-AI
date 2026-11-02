"""
Module: pipeline
Description: Unified inference pipeline for SolarVision AI multi-modal solar panel inspection
Author: SolarVision AI Team
Created: 2026-11-02
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.inference.types import BoundingBox, InferenceResult, InspectionType
from src.inference.visualizer import (
    annotate_classification,
    annotate_detection,
    annotate_segmentation,
)
from src.models.classification.mobilenet import SolarMobileNetV2
from src.models.detection.postprocess import postprocess_detections
from src.models.detection.yolo_lightweight import SolarYOLOv8n
from src.models.segmentation.unet import SolarLightweightUNet

# Default checkpoint locations
DEFAULT_CLASSIFICATION_CKPT = Path("checkpoints/classification/mobilenet_v2_best.pt")
DEFAULT_DETECTION_CKPT = Path("checkpoints/detection/yolov8n_best.pt")
DEFAULT_SEGMENTATION_CKPT = Path("checkpoints/segmentation/unet_best.pt")

# Standard class mappings
DEFAULT_CLASSIFICATION_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
]
DEFAULT_DETECTION_CLASSES = {0: "Dust", 1: "bird-droppings"}


class SolarVisionInferencePipeline:
    """Unified inference engine integrating classification, detection, and segmentation.

    Features:
    - Lazy loading and warm caching of trained PyTorch models.
    - CPU-safe execution with configurable device fallback.
    - Strict input validation supporting file paths, PIL Images, and NumPy arrays.
    - Standardized InferenceResult outputs across all modalities.
    - Integrated visual annotation generation.
    - Batch inference support.
    """

    def __init__(
        self,
        classification_ckpt: Optional[Union[str, Path]] = DEFAULT_CLASSIFICATION_CKPT,
        detection_ckpt: Optional[Union[str, Path]] = DEFAULT_DETECTION_CKPT,
        segmentation_ckpt: Optional[Union[str, Path]] = DEFAULT_SEGMENTATION_CKPT,
        device: Optional[Union[str, torch.device]] = None,
        auto_load: bool = False,
    ) -> None:
        """Initialize the unified inference pipeline.

        Args:
            classification_ckpt: Path to trained classification checkpoint.
            detection_ckpt: Path to trained detection checkpoint.
            segmentation_ckpt: Path to trained segmentation checkpoint.
            device: Computation device. If None, auto-selects cuda if available, else cpu.
            auto_load: If True, immediately load all three models into memory. Defaults to False.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.ckpt_paths = {
            InspectionType.CLASSIFICATION: (
                Path(classification_ckpt) if classification_ckpt else None
            ),
            InspectionType.DETECTION: (
                Path(detection_ckpt) if detection_ckpt else None
            ),
            InspectionType.SEGMENTATION: (
                Path(segmentation_ckpt) if segmentation_ckpt else None
            ),
        }

        # Model cache
        self._models: Dict[InspectionType, nn.Module] = {}
        self._configs: Dict[InspectionType, Dict[str, Any]] = {}

        # Preprocessing transforms for classification
        self._cls_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if auto_load:
            for itype in InspectionType:
                self._load_model(itype)

    def _load_model(self, inspection_type: InspectionType) -> nn.Module:
        """Load model weights from checkpoint into cache if not already loaded.

        Args:
            inspection_type: The inspection modality to load.

        Returns:
            nn.Module: Loaded PyTorch model in eval mode.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            ValueError: If the checkpoint structure is invalid.
        """
        if inspection_type in self._models:
            return self._models[inspection_type]

        ckpt_path = self.ckpt_paths.get(inspection_type)
        if ckpt_path is None or not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {inspection_type.value} not found at '{ckpt_path}'."
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        if inspection_type == InspectionType.CLASSIFICATION:
            num_classes = config.get("num_classes", 6)
            model = SolarMobileNetV2(
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=False,
            )
            model.load_state_dict(state_dict)
            self._configs[inspection_type] = {
                "classes": config.get("classes", DEFAULT_CLASSIFICATION_CLASSES),
                "model_name": "SolarMobileNetV2",
                "model_version": "v1.0-finetuned",
            }

        elif inspection_type == InspectionType.DETECTION:
            num_classes = config.get("num_classes", 2)
            model = SolarYOLOv8n(num_classes=num_classes)
            model.load_state_dict(state_dict)
            classes_cfg = config.get("classes", DEFAULT_DETECTION_CLASSES)
            # Ensure integer keys for detection class mapping
            classes_map = {int(k): str(v) for k, v in classes_cfg.items()}
            self._configs[inspection_type] = {
                "classes": classes_map,
                "model_name": "SolarYOLOv8n",
                "model_version": "v1.1-balanced",
            }

        elif inspection_type == InspectionType.SEGMENTATION:
            in_channels = config.get("in_channels", 1)
            out_channels = config.get("out_channels", 1)
            base_channels = config.get("base_channels", 16)
            model = SolarLightweightUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
            )
            model.load_state_dict(state_dict)
            self._configs[inspection_type] = {
                "default_threshold": 0.70,
                "model_name": "SolarLightweightUNet",
                "model_version": "v1.0",
            }

        model.to(self.device)
        model.eval()
        self._models[inspection_type] = model
        return model

    def _validate_and_load_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Validate input format and return PIL Image alongside metadata.

        Args:
            image_input: File path, PIL Image, or NumPy array.

        Returns:
            Tuple[Image.Image, Dict[str, Any]]: Validated PIL Image and metadata dictionary.

        Raises:
            FileNotFoundError: If string/Path input file does not exist.
            ValueError: If input format or image dimensions are invalid.
        """
        source = "memory"
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            try:
                img = Image.open(path)
                img.load()  # Ensure file is read into memory
                source = str(path)
            except Exception as exc:
                raise ValueError(f"Failed to read image at {path}: {exc}") from exc
        elif isinstance(image_input, Image.Image):
            img = image_input.copy()
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input NumPy array is empty.")
            if image_input.ndim == 2:
                img = Image.fromarray(image_input.astype(np.uint8), mode="L")
            elif image_input.ndim == 3:
                if image_input.shape[2] == 1:
                    img = Image.fromarray(
                        image_input[:, :, 0].astype(np.uint8), mode="L"
                    )
                elif image_input.shape[2] == 3:
                    img = Image.fromarray(image_input.astype(np.uint8), mode="RGB")
                elif image_input.shape[2] == 4:
                    img = Image.fromarray(image_input.astype(np.uint8), mode="RGBA")
                else:
                    raise ValueError(
                        f"Unsupported number of channels in array: {image_input.shape[2]}"
                    )
            else:
                raise ValueError(f"Unsupported array dimensions: {image_input.shape}")
        else:
            raise ValueError(
                f"Unsupported image input type: {type(image_input)}. "
                "Expected file path, PIL.Image, or numpy.ndarray."
            )

        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Image has invalid dimensions: {w}x{h}")

        metadata = {
            "source": source,
            "width": w,
            "height": h,
            "original_mode": img.mode,
        }
        return img, metadata

    @torch.inference_mode()
    def predict_classification(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 6,
        create_visualization: bool = True,
    ) -> InferenceResult:
        """Perform panel condition classification.

        Args:
            image_input: Input image.
            top_k: Number of top class probabilities to retain. Defaults to 6.
            create_visualization: Whether to render annotated image. Defaults to True.

        Returns:
            InferenceResult: Classification prediction result.
        """
        img, metadata = self._validate_and_load_image(image_input)
        model = self._load_model(InspectionType.CLASSIFICATION)
        cfg = self._configs[InspectionType.CLASSIFICATION]
        classes: List[str] = cfg["classes"]

        start_time = time.perf_counter()

        img_rgb = img.convert("RGB")
        tensor = self._cls_transform(img_rgb).unsqueeze(0).to(self.device)

        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_class = classes[pred_idx]

        class_probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}

        proc_time = (time.perf_counter() - start_time) * 1000.0

        annotated_image = None
        if create_visualization:
            annotated_image = annotate_classification(
                image=img_rgb,
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probabilities,
            )

        return InferenceResult(
            inspection_type=InspectionType.CLASSIFICATION,
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            processing_time_ms=proc_time,
            image_metadata=metadata,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            annotated_image=annotated_image,
        )

    @torch.inference_mode()
    def predict_detection(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        create_visualization: bool = True,
    ) -> InferenceResult:
        """Perform aerial soiling defect detection.

        Args:
            image_input: Input image.
            conf_threshold: Minimum confidence threshold. Defaults to 0.10.
            iou_threshold: Non-Maximum Suppression IoU threshold. Defaults to 0.45.
            max_detections: Maximum detections to return. Defaults to 100.
            create_visualization: Whether to render annotated image. Defaults to True.

        Returns:
            InferenceResult: Detection prediction result containing bounding boxes.
        """
        img, metadata = self._validate_and_load_image(image_input)
        model = self._load_model(InspectionType.DETECTION)
        cfg = self._configs[InspectionType.DETECTION]
        classes_map: Dict[int, str] = cfg["classes"]

        start_time = time.perf_counter()

        img_rgb = img.convert("RGB")
        resized = img_rgb.resize((640, 640), resample=Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        preds = model(tensor)
        post_results = postprocess_detections(
            preds=preds,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        det_tensor = post_results[0].cpu().numpy()
        boxes: List[BoundingBox] = []

        if len(det_tensor) > 0:
            for row in det_tensor:
                cid = int(row[5])
                cname = classes_map.get(cid, f"class_{cid}")
                box = BoundingBox(
                    x1=float(row[0]),
                    y1=float(row[1]),
                    x2=float(row[2]),
                    y2=float(row[3]),
                    score=float(row[4]),
                    class_id=cid,
                    class_name=cname,
                )
                boxes.append(box)

        proc_time = (time.perf_counter() - start_time) * 1000.0

        annotated_image = None
        if create_visualization:
            annotated_image = annotate_detection(image=img_rgb, boxes=boxes)

        return InferenceResult(
            inspection_type=InspectionType.DETECTION,
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            processing_time_ms=proc_time,
            image_metadata=metadata,
            boxes=boxes,
            annotated_image=annotated_image,
        )

    @torch.inference_mode()
    def predict_segmentation(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        threshold: float = 0.70,
        create_visualization: bool = True,
    ) -> InferenceResult:
        """Perform electroluminescence microcrack defect segmentation.

        Args:
            image_input: Input grayscale or RGB EL image.
            threshold: Probability threshold for binary mask. Defaults to 0.70.
            create_visualization: Whether to render multi-panel overlay. Defaults to True.

        Returns:
            InferenceResult: Segmentation prediction result containing defect mask and ratio.
        """
        img, metadata = self._validate_and_load_image(image_input)
        model = self._load_model(InspectionType.SEGMENTATION)
        cfg = self._configs[InspectionType.SEGMENTATION]

        start_time = time.perf_counter()

        img_gray = img.convert("L")
        orig_w, orig_h = img_gray.size

        resized = img_gray.resize((256, 256), resample=Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        # Binary threshold at 256x256
        mask_256 = (probs >= threshold).astype(np.uint8)

        # Scale mask back to original image dimensions if needed
        if (orig_w, orig_h) != (256, 256):
            mask_pil = Image.fromarray(mask_256 * 255).resize(
                (orig_w, orig_h), resample=Image.NEAREST
            )
            mask_final = (np.array(mask_pil) > 127).astype(np.uint8)
        else:
            mask_final = mask_256

        defect_pixels = int(np.sum(mask_final))
        total_pixels = int(mask_final.size)
        defect_ratio = float(defect_pixels / total_pixels) if total_pixels > 0 else 0.0

        proc_time = (time.perf_counter() - start_time) * 1000.0

        annotated_image = None
        if create_visualization:
            annotated_image = annotate_segmentation(
                image=img_gray,
                mask=mask_final,
            )

        return InferenceResult(
            inspection_type=InspectionType.SEGMENTATION,
            model_name=cfg["model_name"],
            model_version=cfg["model_version"],
            processing_time_ms=proc_time,
            image_metadata=metadata,
            mask=mask_final,
            defect_ratio=defect_ratio,
            threshold=threshold,
            annotated_image=annotated_image,
        )

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        inspection_type: Union[InspectionType, str],
        **kwargs: Any,
    ) -> InferenceResult:
        """Route input to the appropriate inspection pipeline.

        Args:
            image_input: Input image file path, PIL Image, or NumPy array.
            inspection_type: Target inspection modality ('classification', 'detection', 'segmentation').
            **kwargs: Extra parameters passed to the modality-specific predictor.

        Returns:
            InferenceResult: Standardized result object.
        """
        itype = (
            inspection_type
            if isinstance(inspection_type, InspectionType)
            else InspectionType.from_str(str(inspection_type))
        )

        if itype == InspectionType.CLASSIFICATION:
            return self.predict_classification(image_input, **kwargs)
        elif itype == InspectionType.DETECTION:
            return self.predict_detection(image_input, **kwargs)
        elif itype == InspectionType.SEGMENTATION:
            return self.predict_segmentation(image_input, **kwargs)
        else:
            raise ValueError(f"Unsupported inspection type: {itype}")

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
        inspection_type: Union[InspectionType, str],
        **kwargs: Any,
    ) -> List[InferenceResult]:
        """Execute inference over a batch of images sequentially.

        Args:
            image_inputs: List of images.
            inspection_type: Inspection modality.
            **kwargs: Extra parameters passed to the predictor.

        Returns:
            List[InferenceResult]: List of standardized result objects.
        """
        return [self.predict(img, inspection_type, **kwargs) for img in image_inputs]

    def get_status(self) -> Dict[str, Any]:
        """Return runtime status of inference engine and cached models.

        Returns:
            Dict[str, Any]: Device, cache status, and checkpoint paths.
        """
        return {
            "device": str(self.device),
            "loaded_models": [itype.value for itype in self._models.keys()],
            "checkpoints": {
                itype.value: str(path) for itype, path in self.ckpt_paths.items()
            },
        }

    def unload_models(self) -> None:
        """Release cached models to free system memory."""
        self._models.clear()
        self._configs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
