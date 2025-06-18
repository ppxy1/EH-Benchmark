from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from pydantic import BaseModel, Field
from collections import Counter

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import json

from .models.yolo import YOLO
from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from pydantic import BaseModel, Field
from collections import Counter

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

class FundusYOLODetectionInput(BaseModel):
    """Input schema for the Fundus YOLO Detection Tool. Only supports JPG or PNG images."""

    image_path: str = Field(
        description="Path to the Fundus image file, only supports JPG or PNG images",
    )

class FundusYOLODetectionTool(BaseTool):
    """Tool for detecting specific classes in Fundus images using the YOLO model.

    This tool processes Fundus images and detects hardcoded classes ['ex', 'he', 'se', 'ma'].
    It returns bounding box coordinates, a visualization of the detected classes,
    and the count of detections per class.
    """

    name: str = "FundusYOLODetectionTool"
    description: str = (
        "Detects and visualizes hardcoded classes ['Hard Exudate', 'Hemohedge', 'Soft Exudate', 'Microaneurysms'] "
        "in Fundus images. Takes a Fundus image path and returns bounding box coordinates in format "
        "[x_topleft, y_topleft, x_bottomright, y_bottomright] where each value is between 0-1, "
        "a visualization of the detected classes, confidence metadata, and the count of detections per class. "
        "Example input: {'image_path': '/path/to/fundus.png'}"
    )
    args_schema: Type[BaseModel] = FundusYOLODetectionInput

    yolo: YOLO = None
    temp_dir: Path = None
    classes: List[str] = None

    def __init__(
        self,
        model_path: str = 'tools/pth/yolo_best_epoch_weights.pth',
        temp_dir: Optional[str] = None,
        cuda: bool = True,
    ):
        """Initialize the Fundus YOLO Detection Tool."""
        super().__init__()
        # Hardcode classes and anchors
        self.classes = ['ex', 'he', 'se', 'ma']
        anchors = [
            [12, 16], [19, 36], [40, 28],
            [36, 75], [76, 55], [72, 146],
            [142, 110], [192, 243], [459, 401]
        ]
        self.yolo = YOLO(
            model_path=model_path,
            classes_path=None,
            anchors_path=None,
            classes=self.classes,
            anchors=anchors,
            cuda=cuda,
        )
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _visualize_bboxes(
        self, image: Image.Image, boxes: List[Tuple[float, float, float, float]], labels: List[str], confs: List[float] = None
    ) -> str:
        """Create and save visualization of multiple bounding boxes on the image."""
        plt.figure(figsize=(12, 12))
        plt.imshow(image)

        # print("Visualizing boxes:", len(boxes))
        for i, (bbox, label, conf) in enumerate(zip(boxes, labels, confs or [None] * len(boxes))):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            # print(f"Box {i}: {label}, Conf: {conf}, Coords: ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})")

            # 验证坐标有效性
            if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                # print(f"Warning: Invalid box coordinates for {label}: ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})")
                continue

            # 绘制边界框
            rect = plt.Rectangle(
                (x1 * image.width, y1 * image.height),
                width * image.width,
                height * image.height,
                fill=False,
                color="red",
                linewidth=2,
            )
            plt.gca().add_patch(rect)
            # print(f"Added rectangle: {rect}")

            # 调整标签位置，防止溢出
            text_x = x1 * image.width
            text_y = y1 * image.height - 10  # 上移 10 像素
            if text_y < 0:
                text_y = y1 * image.height + 10  # 如果上移溢出，改为下移
            if text_x < 0:
                text_x = 0
            if text_x > image.width - 50:  # 粗略估计文本宽度
                text_x = image.width - 50

            # 绘制标签，无背景框或边框
            text = f"{label} {conf:.3f}" if conf is not None else label
            plt.text(
                text_x, text_y, text,
                color="red", fontsize=12, fontfamily='sans-serif'
            )

        plt.axis("off")
        viz_path = self.temp_dir / f"detection_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(viz_path, bbox_inches="tight", dpi=150)
        plt.close()
        # print("Saved visualization to:", viz_path)
        return str(viz_path)

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Detect hardcoded classes ['ex', 'he', 'se', 'ma'] in a Fundus image and count detections.

        Args:
            image_path: Path to the Fundus image file
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary and metadata dictionary
        """
        # Class mapping dictionary
        class_mapping = {
            'ex': 'Hard Exudate',
            'he': 'Hemorrhage',
            'se': 'Soft Exudate',
            'ma': 'Microaneurysms'
        }

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Use hardcoded classes
            classes = self.classes

            # Perform detection
            results = self.yolo.detect_image(image, crop=False, count=False)

            # Debug: Print results
            # print("Type of results:", type(results))
            # print("Results content:", results)

            # Check if results is None or empty
            if results is None or results[0] is None:
                output = {
                    "predictions": [],
                    "class_counts": {class_mapping[cls]: 0 for cls in classes},
                    "visualization_path": None,
                }
                metadata = {
                    "image_path": image_path,
                    "original_size": image.size,
                    "device": "cuda" if self.yolo.cuda and torch.cuda.is_available() else "cpu",
                    "analysis_status": "completed_no_finding",
                }

                result = {
                    "output": output,
                    "metadata": metadata
                } 

                return  json.dumps(result)

            # Extract bounding boxes and labels
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

            # # Debug: Print extracted data
            # print("Top labels:", top_label)
            # print("Top confidences:", top_conf)
            # print("Top boxes:", top_boxes)

            # Filter by hardcoded classes
            class_indices = [self.yolo.class_names.index(cls) for cls in classes if cls in self.yolo.class_names]
            filtered_indices = [i for i, label in enumerate(top_label) if label in class_indices]

            filtered_boxes = top_boxes[filtered_indices]
            filtered_labels = [self.yolo.class_names[top_label[i]] for i in filtered_indices]
            filtered_conf = top_conf[filtered_indices]

            # Count the number of detections per class
            class_counts = Counter(filtered_labels)

            # Normalize boxes to [0, 1]
            normalized_boxes = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = box
                normalized_boxes.append([
                    x1 / image.width,
                    y1 / image.height,
                    x2 / image.width,
                    y2 / image.height,
                ])

            # Map labels to full names for visualization
            mapped_labels = [class_mapping[label] for label in filtered_labels]

            # Create visualization
            viz_path = self._visualize_bboxes(image, normalized_boxes, mapped_labels)

            output = {
                "predictions": [
                    {
                        "class": class_mapping[label],  # Map to full name
                        "bounding_box": box,
                        "confidence": float(conf),
                    }
                    for label, box, conf in zip(filtered_labels, normalized_boxes, filtered_conf)
                ],
                "class_counts": {class_mapping[cls]: class_counts.get(cls, 0) for cls in classes},
                "visualization_path": viz_path,
            }

            metadata = {
                "image_path": image_path,
                "original_size": image.size,
                "device": "cuda",
                "analysis_status": "completed",
            }

            result = {
                    "output": output,
                    "metadata": metadata
            } 

            return  json.dumps(result)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "class_counts": {class_mapping[cls]: 0 for cls in self.classes},
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": str(e)
            })

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Asynchronous version of _run."""
        return self._run(image_path, run_manager)
