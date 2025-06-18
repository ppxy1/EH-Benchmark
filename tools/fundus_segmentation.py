import torch
import torchvision
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform
import traceback
import segmentation_models_pytorch as smp
from .models.unetplus import UnetPlusPlus
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import json
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

class FundusSegmentationInput(BaseModel):
    """Input schema for the Fundus Segmentation Tool."""
    image_path: str = Field(..., description="Path to the fundus image file (PNG/JPG) to be segmented")

class OrganMetrics(BaseModel):
    """Detailed metrics for a segmented optic cup or disk."""
    area_pixels: int = Field(..., description="Area in pixels")
    area_cm2: float = Field(..., description="Approximate area in cm²")
    centroid: Tuple[float, float] = Field(..., description="(y, x) coordinates of centroid")
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box coordinates (min_y, min_x, max_y, max_x)"
    )
    width: int = Field(..., description="Width of the organ in pixels")
    height: int = Field(..., description="Height of the organ in pixels")
    aspect_ratio: float = Field(..., description="Height/width ratio")
    mean_intensity: float = Field(..., description="Mean pixel intensity in the organ region")
    std_intensity: float = Field(..., description="Standard deviation of pixel intensity")
    confidence_score: float = Field(..., description="Model confidence score for this organ")
    bounding_box_points: List[Tuple[int, int]] = Field(
        ..., description="Four points of the minimum enclosing rectangle (top-left, top-right, bottom-right, bottom-left)"
    )

class FundusSegmentationTool(BaseTool):
    """Tool for segmenting optic cup and optic disk in fundus images and computing cup-to-disk ratio."""

    name: str = "FundusSegmentationTool"
    description: str = (
        "Segments fundus images to identify optic cup and optic disk. "
        "Returns segmentation visualization, bounding box coordinates, and cup-to-disk ratio for diagnosis."
    )
    args_schema: Type[BaseModel] = FundusSegmentationInput

    cup_model: Any = None
    disk_model: Any = None
    device: Optional[str] = "cuda"
    image_transform: Any = None
    pixel_spacing_mm: float = 0.2
    temp_dir: Path = Path("temp")
    organ_map: Dict[str, int] = None

    def __init__(
        self,
        device: Optional[str] = "cuda",
        temp_dir: Optional[Path] = Path("temp"),
        cup_model_weights: Optional[str] = "tools/pth/cup_Net_epoch_best.pth",
        disk_model_weights: Optional[str] = "tools/pth/disc_Net_epoch_best.pth"
    ):
        """Initialize the segmentation tool with U-Net++ models and temporary directory."""
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize U-Net++ models for optic cup and optic disk
        self.cup_model = UnetPlusPlus()
        self.disk_model = UnetPlusPlus()

        # Load pre-trained weights
        if cup_model_weights and Path(cup_model_weights).exists():
            self.cup_model.load_state_dict(torch.load(cup_model_weights, map_location=self.device))
        if disk_model_weights and Path(disk_model_weights).exists():
            self.disk_model.load_state_dict(torch.load(disk_model_weights, map_location=self.device))

        self.cup_model = self.cup_model.to(self.device)
        self.disk_model = self.disk_model.to(self.device)
        self.cup_model.eval()
        self.disk_model.eval()

        # Updated transform pipeline as requested
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # Converts PIL Image to tensor and normalizes to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

        self.organ_map = {
            "Optic Cup": 0,
            "Optic Disk": 1
        }

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
        """Compute comprehensive metrics for a single organ mask."""
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
        bbox_points = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ]

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
            bounding_box_points=bbox_points
        )

    def _save_visualization(
            self, original_img: np.ndarray, cup_mask: np.ndarray, disk_mask: np.ndarray
    ) -> str:
        """
        Save visualization of original image (in color) with optic cup (red) and optic disk (blue) masks overlaid.
        """
        # Prepare the original image for visualization
        if len(original_img.shape) == 3:
            if original_img.shape[2] == 4:  # Handle RGBA images
                original_img = original_img[:, :, :3]  # Remove alpha channel
            # Ensure uint8 for proper display
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
        else:  # Grayscale image (2D)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)

        # Set up the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(original_img)  # Display in natural color (no cmap="gray")

        # Overlay optic cup mask (red)
        if cup_mask.sum() > 0:
            # Align the mask to the original image size
            cup_mask = self._align_mask_to_original(cup_mask, original_img.shape[:2])
            # Create a colored overlay with transparency
            colored_mask = np.zeros((*original_img.shape[:2], 4))
            colored_mask[cup_mask > 0] = (1, 0, 0, 0.3)  # Red with 30% transparency
            plt.imshow(
                colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0]
            )
            # Add legend entry for optic cup
            # plt.plot([], [], color=(1, 0, 0), label="Optic Cup", linewidth=3)

        # Overlay optic disk mask (blue)
        if disk_mask.sum() > 0:
            # Align the mask to the original image size
            disk_mask = self._align_mask_to_original(disk_mask, original_img.shape[:2])
            # Create a colored overlay with transparency
            colored_mask = np.zeros((*original_img.shape[:2], 4))
            colored_mask[disk_mask > 0] = (0, 0, 1, 0.3)  # Blue with 30% transparency
            plt.imshow(
                colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0]
            )
            # Add legend entry for optic disk
            # plt.plot([], [], color=(0, 0, 1), label="Optic Disk", linewidth=3)

        # Add title and legend
        plt.title("Segmentation Overlay")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.axis("off")

        # Save the plot
        save_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        return str(save_path)

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run segmentation analysis for optic cup and optic disk."""
        try:
            # Load image as numpy array
            img = skimage.io.imread(image_path)
            # print("Loaded image shape:", img.shape)
            original_img = img.copy()

            # Convert numpy array to PIL Image and handle RGBA
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]  # Remove alpha channel if present
            if img.max() > 1:
                img = img.astype(np.uint8)  # Ensure uint8 for PIL conversion
            img_pil = Image.fromarray(img).convert('RGB')  # Convert to PIL Image and ensure RGB

            # Apply the transform
            img_transformed = self.image_transform(img_pil)
            # print("Transformed image shape:", img_transformed.shape)
            img = img_transformed.unsqueeze(0).to(self.device)

            with torch.no_grad():
                cup_pred = self.cup_model(img)
                disk_pred = self.disk_model(img)
            cup_probs = torch.softmax(cup_pred, dim=1)[:, 1, :, :]
            disk_probs = torch.softmax(disk_pred, dim=1)[:, 1, :, :]
            cup_mask = (cup_probs > 0.5).float().squeeze().cpu().numpy()
            disk_mask = (disk_probs > 0.5).float().squeeze().cpu().numpy()
            # print("Cup mask shape:", cup_mask.shape)
            # print("Disk mask shape:", disk_mask.shape)

            cup_mask_processed = self.post_process_mask(cup_mask)
            disk_mask_processed = self.post_process_mask(disk_mask)

            viz_path = self._save_visualization(original_img, cup_mask_processed, disk_mask_processed)

            results = {}
            for organ_name, mask, probs in [
                ("Optic Cup", cup_mask_processed, cup_probs),
                ("Optic Disk", disk_mask_processed, disk_probs)
            ]:
                if mask.sum() > 0:
                    metrics = self._compute_organ_metrics(
                        mask, original_img, float(probs.mean().cpu())
                    )
                    if metrics:
                        results[organ_name] = metrics

            cup_area = results.get("Optic Cup", OrganMetrics(
                area_pixels=0, area_cm2=0, centroid=(0,0), bbox=(0,0,0,0),
                width=0, height=0, aspect_ratio=0, mean_intensity=0,
                std_intensity=0, confidence_score=0, bounding_box_points=[]
            )).area_pixels
            disk_area = results.get("Optic Disk", OrganMetrics(
                area_pixels=0, area_cm2=0, centroid=(0,0), bbox=(0,0,0,0),
                width=0, height=0, aspect_ratio=0, mean_intensity=0,
                std_intensity=0, confidence_score=0, bounding_box_points=[]
            )).area_pixels
            cup_to_disk_ratio = cup_area / max(disk_area, 1)

            output = {
                "segmentation_image_path": viz_path,
                "metrics": {organ: metrics.model_dump() for organ, metrics in results.items()},
                "cup_to_disk_ratio": float(cup_to_disk_ratio)
            }

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "original_size": original_img.shape,
                "model_size": (256, 256),
                "pixel_spacing_mm": self.pixel_spacing_mm,
                "processed_organs": list(results.keys()),
                "cup_to_disk_ratio": float(cup_to_disk_ratio),
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