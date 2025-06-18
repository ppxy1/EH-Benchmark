import torch
import torchvision
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform
import traceback
from .models.unetplus import UnetPlusPlus
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import uuid
import cv2
import os
from typing import Dict, Tuple, List, Optional, Any, Type
from scipy.ndimage import label, binary_closing
import json

class OCTSegmentationInput(BaseModel):
    """Input schema for the OCT Segmentation Tool."""
    image_path: str = Field(..., description="Path to the OCT image file (PNG/JPG) to be segmented")

class OrganMetrics(BaseModel):
    """Detailed metrics for a segmented region in OCT images."""
    area_pixels: int = Field(..., description="Area in pixels")
    area_cm2: float = Field(..., description="Approximate area in cm²")
    centroid: Tuple[float, float] = Field(..., description="(y, x) coordinates of centroid")
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box coordinates (min_y, min_x, max_y, max_x)"
    )
    width: int = Field(..., description="Width of the region in pixels")
    height: int = Field(..., description="Height of the region in pixels")
    aspect_ratio: float = Field(..., description="Height/width ratio")
    mean_intensity: float = Field(..., description="Mean pixel intensity in the region")
    std_intensity: float = Field(..., description="Standard deviation of pixel intensity")
    confidence_score: float = Field(..., description="Model confidence score for this region")
    contour_key_points: List[Tuple[int, int]] = Field(
        ..., description="Key points from the approximated contour of the region"
    )

class OCTSegmentationTool(BaseTool):
    """Tool for segmenting choroid, retina, and macular hole in OCT images."""

    name: str = "OCTSegmentationTool"
    description: str = (
        "Segments OCT images to identify choroid, retina, and macular hole (mh). "
        "Returns segmentation visualization and key points for each segmented part."
    )
    args_schema: Type[BaseModel] = OCTSegmentationInput

    choroid_model: Any = None
    retina_model: Any = None
    mh_model: Any = None
    device: Optional[str] = "cuda"
    image_transform: Any = None
    pixel_spacing_mm: float = 0.2
    temp_dir: Path = Path("temp")

    def __init__(
        self,
        device: Optional[str] = "cuda",
        temp_dir: Optional[Path] = Path("temp"),
        choroid_model_weights: Optional[str] = "/raid/baiyang/xiaoyu/fundusagent/tools/pth/choroid_Net_epoch_best.pth",
        retina_model_weights: Optional[str] = "/raid/baiyang/xiaoyu/fundusagent/tools/pth/retina_Net_epoch_best.pth",
        mh_model_weights: Optional[str] = "/raid/baiyang/xiaoyu/fundusagent/tools/pth/mh_Net_epoch_best.pth"
    ):
        """Initialize the segmentation tool with U-Net++ models and temporary directory."""
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")

        # Initialize U-Net++ models for choroid, retina, and mh
        self.choroid_model = UnetPlusPlus()
        self.retina_model = UnetPlusPlus()
        self.mh_model = UnetPlusPlus()

        # Load pre-trained weights
        if choroid_model_weights and Path(choroid_model_weights).exists():
            self.choroid_model.load_state_dict(torch.load(choroid_model_weights, map_location=self.device))
        if retina_model_weights and Path(retina_model_weights).exists():
            self.retina_model.load_state_dict(torch.load(retina_model_weights, map_location=self.device))
        if mh_model_weights and Path(mh_model_weights).exists():
            self.mh_model.load_state_dict(torch.load(mh_model_weights, map_location=self.device))

        self.choroid_model = self.choroid_model.to(self.device)
        self.retina_model = self.retina_model.to(self.device)
        self.mh_model = self.mh_model.to(self.device)
        self.choroid_model.eval()
        self.retina_model.eval()
        self.mh_model.eval()

        # Transform pipeline for RGB input
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def post_process_mask(self, mask: np.ndarray, scale_to_255: bool = False) -> np.ndarray:
        """Post-process the predicted mask."""
        binary_mask = (mask > 0).astype(np.uint8)
        binary_mask = binary_closing(binary_mask, structure=np.ones((3, 3)))
        labeled_mask, num_components = label(binary_mask)

        if num_components > 1:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            binary_mask = (labeled_mask == largest_component).astype(np.uint8)
        else:
            binary_mask = binary_mask.astype(np.uint8)

        if scale_to_255:
            binary_mask = binary_mask * 255
        return binary_mask

    def _align_mask_to_original(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Align a mask from the transformed space back to the original image."""
        orig_h, orig_w = original_shape
        resized_mask = skimage.transform.resize(
            mask, (orig_h, orig_w), order=0, preserve_range=True, anti_aliasing=False
        )
        return resized_mask

    def _compute_organ_metrics(
        self, mask: np.ndarray, original_img: np.ndarray, confidence: float
    ) -> Optional[OrganMetrics]:
        """Compute comprehensive metrics for a single region mask, including contour key points."""
        if len(original_img.shape) == 3:
            original_img = original_img[:, :, 0]

        if mask.shape != original_img.shape:
            mask = self._align_mask_to_original(mask, original_img.shape)

        props = skimage.measure.regionprops(mask.astype(int))
        if not props:
            return None

        props = props[0]
        area_cm2 = mask.sum() * (self.pixel_spacing_mm / 10) ** 2

        img_height, img_width = mask.shape
        cy, cx = props.centroid

        organ_pixels = original_img[mask > 0]
        mean_intensity = organ_pixels.mean() if len(organ_pixels) > 0 else 0
        std_intensity = organ_pixels.std() if len(organ_pixels) > 0 else 0

        min_y, min_x, max_y, max_x = props.bbox

        # Compute contour key points
        mask_uint8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_key_points = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            contour_key_points = [tuple(point[0]) for point in approx]

        return OrganMetrics(
            area_pixels=int(mask.sum()),
            area_cm2=float(area_cm2),
            centroid=(float(cy), float(cx)),
            bbox=(min_y, min_x, max_y, max_x),
            width=int(max_x - min_x),
            height=int(max_y - min_y),
            aspect_ratio=float((max_y - min_y) / max(1, max_x - min_x)),
            mean_intensity=float(mean_intensity),
            std_intensity=float(std_intensity),
            confidence_score=float(confidence),
            contour_key_points=contour_key_points
        )

    def _save_visualization(
        self, original_img: np.ndarray, choroid_mask: np.ndarray, retina_mask: np.ndarray, mh_mask: np.ndarray
    ) -> str:
        """Save visualization with choroid (red), retina (green), and mh (blue) masks overlaid."""
        if len(original_img.shape) == 3:
            if original_img.shape[2] == 4:
                original_img = original_img[:, :, :3]
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(original_img)

        # Overlay choroid mask (red)
        if choroid_mask.sum() > 0:
            choroid_mask = self._align_mask_to_original(choroid_mask, original_img.shape[:2])
            colored_mask = np.zeros((*original_img.shape[:2], 4))
            colored_mask[choroid_mask > 0] = (1, 0, 0, 0.3)  # Red
            plt.imshow(colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0])

        # Overlay retina mask (green)
        if retina_mask.sum() > 0:
            retina_mask = self._align_mask_to_original(retina_mask, original_img.shape[:2])
            colored_mask = np.zeros((*original_img.shape[:2], 4))
            colored_mask[retina_mask > 0] = (0, 1, 0, 0.3)  # Green
            plt.imshow(colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0])

        # Overlay mh mask (blue)
        if mh_mask.sum() > 0:
            mh_mask = self._align_mask_to_original(mh_mask, original_img.shape[:2])
            colored_mask = np.zeros((*original_img.shape[:2], 4))
            colored_mask[mh_mask > 0] = (0, 0, 1, 0.3)  # Blue
            plt.imshow(colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0])

        # Add text labels
        plt.text(10, 20, "Choroid: Red", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(10, 40, "Retina: Green", color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.text(10, 60, "MH: Blue", color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.title("OCT Segmentation Overlay")
        plt.axis("off")

        save_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        return str(save_path)

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run segmentation analysis for choroid, retina, and mh."""
        try:
            img = skimage.io.imread(image_path)
            # print("Loaded image shape:", img.shape)
            original_img = img.copy()

            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if img.max() > 1:
                img = img.astype(np.uint8)
            img_pil = Image.fromarray(img).convert('RGB')

            img_transformed = self.image_transform(img_pil)
            # print("Transformed image shape:", img_transformed.shape)
            img = img_transformed.unsqueeze(0).to(self.device)

            with torch.no_grad():
                choroid_pred = self.choroid_model(img)
                retina_pred = self.retina_model(img)
                mh_pred = self.mh_model(img)
            choroid_probs = torch.softmax(choroid_pred, dim=1)[:, 1, :, :]
            retina_probs = torch.softmax(retina_pred, dim=1)[:, 1, :, :]
            mh_probs = torch.softmax(mh_pred, dim=1)[:, 1, :, :]
            choroid_mask = (choroid_probs > 0.5).float().squeeze().cpu().numpy()
            retina_mask = (retina_probs > 0.5).float().squeeze().cpu().numpy()
            mh_mask = (mh_probs > 0.5).float().squeeze().cpu().numpy()

            choroid_mask_processed = self.post_process_mask(choroid_mask)
            retina_mask_processed = self.post_process_mask(retina_mask)
            mh_mask_processed = self.post_process_mask(mh_mask)

            viz_path = self._save_visualization(original_img, choroid_mask_processed, retina_mask_processed, mh_mask_processed)

            results = {}
            for organ_name, mask, probs in [
                ("Choroid", choroid_mask_processed, choroid_probs),
                ("Retina", retina_mask_processed, retina_probs),
                ("MH", mh_mask_processed, mh_probs)
            ]:
                if mask.sum() > 0:
                    metrics = self._compute_organ_metrics(mask, original_img, float(probs.mean().cpu()))
                    if metrics:
                        results[organ_name] = metrics

            output = {
                "segmentation_image_path": viz_path,
                "metrics": {organ: metrics.model_dump() for organ, metrics in results.items()}
            }

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "original_size": original_img.shape,
                "model_size": (256, 256),
                "pixel_spacing_mm": self.pixel_spacing_mm,
                "segmented_parts": list(results.keys()),
                "analysis_status": "completed",
            }

            result = {
                "output": output,
                "metadata": metadata
            }

            return json.dumps(result)

        except Exception as e:
            print("Error occurred:", str(e))
            print("Traceback:", traceback.format_exc())
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }

            result = {
                "output": error_output,
                "metadata": error_metadata
            }
            
            return json.dumps(result)

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path)