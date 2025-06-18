from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field
import torch
from PIL import Image
from torchvision import transforms
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from .models.modeling import VisionTransformer, CONFIGS
import json
import os
import torch.nn.functional as F
import traceback

class DR_Input(BaseModel):
    """Input for Fundus image analysis tool. Only supports JPG or PNG images."""
    image_path: str = Field(
        ..., description="Path to the Fundus image file, only supports JPG or PNG images"
    )

class DR_ClassifierTool(BaseTool):
    """Tool that classifies Fundus images for diabetic retinopathy (DR) into five stages."""
    args_schema: Type[BaseModel] = DR_Input
    model: VisionTransformer = None
    device: Optional[str] = "cuda"
    transform_field: transforms.Compose = None

    name: str = "DR_ClassifierTool"
    description: str = (
        "A tool that analyzes Fundus images and classifies them into five stages of diabetic retinopathy (DR). "
        "Input should be the path to a Fundus image file (JPG or PNG). "
        "Output is a JSON string with 'probabilities' (DR stages and their predicted probabilities, 0 to 1) and "
        "'metadata' (image path, status, note, and top 3 predictions). "
        "The five stages are: Stage 0: No apparent retinopathy, Stage 1: Mild non-proliferative DR, "
        "Stage 2: Moderate non-proliferative DR, Stage 3: Severe non-proliferative DR, and Stage 4: Proliferative DR. "
        "Higher values indicate a higher likelihood of the stage being present."
    )

    DR_STAGES: list = [
        "Stage 0: No apparent retinopathy",
        "Stage 1: Mild non-proliferative DR",
        "Stage 2: Moderate non-proliferative DR",
        "Stage 3: Severe non-proliferative DR",
        "Stage 4: Proliferative DR"
    ]

    def __init__(self, model_type: str = "R50-ViT-B_16", device: Optional[str] = "cuda", img_size: int = 256):
        super().__init__()
        config = CONFIGS[model_type]
        num_classes = 5
        self.model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        checkpoint_path = "tools/pth/DR-Grade.bin"
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model weights file not found at {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Model weights loaded successfully from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            traceback.print_exc()

        self.transform_field = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process the input Fundus image for model inference."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at {image_path}")
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform_field(image)
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            traceback.print_exc()
            raise

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the DR classification on the input Fundus image and return a JSON string."""
        try:
            image_tensor = self._process_image(image_path)

            with torch.no_grad():
                logits = self.model(image_tensor)[0]
                probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

                # Map probabilities to DR stages
                output = {stage: float(prob) for stage, prob in zip(self.DR_STAGES, probabilities)}

                # Get top 3 predictions for metadata
                sorted_output = sorted(output.items(), key=lambda x: x[1], reverse=True)
                top3 = sorted_output[:3]

                # Prepare metadata
                metadata = {
                    "image_path": image_path,
                    "analysis_status": "completed",
                    "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood.",
                    "top_3_predictions": [
                        {"stage": stage, "probability": prob} for stage, prob in top3
                    ]
                }

                # Combine probabilities and metadata into a single result
                result = {
                    "output": output,
                    "metadata": metadata
                }

                return json.dumps(result)
            
        except Exception as e:
            print(f"DR_ClassifierTool: Error during classification: {str(e)}")
            traceback.print_exc()
            return json.dumps({
                "probabilities": {},
                "metadata": {
                    "image_path": image_path,
                    "analysis_status": "failed",
                    "note": f"Error during classification: {str(e)}",
                    "top_3_predictions": []
                }
            })

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronous run method for DR classification."""
        return self._run(image_path)
    
