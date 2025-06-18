from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field
import sys
import os
import torch
import torchvision
import torch.nn.functional as F
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import json
import logging
import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from .models.modeling import VisionTransformer, CONFIGS
import numpy as np
import traceback
import os
class Fundus_OCT_Input(BaseModel):
    """Input for Fundus and OCT image analysis tools. Only supports JPG or PNG images."""
    image_path: str = Field(
        ..., description="Path to the Fundus or OCT image file, only supports JPG or PNG images"
    )

class Fundus_OCT_ClassifierTool(BaseTool):
    """Tool that classifies Fundus and OCT images for 18 pathologies using a Vision Transformer model.

    This tool uses a pre-trained Vision Transformer (ViT) model to analyze Fundus and OCT images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Retinitis Pigmentosa in Fundus, Retinal Detachment in Fundus, Pterygium in Fundus, Normal in Fundus,
    Myopia in Fundus, Macular Scar in Fundus, Glaucoma in Fundus, Disc Edema in Fundus,
    Diabetic Retinopathy in Fundus, Central Serous Chorioretinopathy in Fundus,
    Age-related Macular Degeneration in OCT, Choroidal Neovascularization in OCT,
    Central Serous Retinopathy in OCT, Diabetic Macular Edema in OCT, Diabetic Retinopathy in OCT,
    Yellow deposits under the retina in OCT, Macular Hole in OCT, Normal in OCT.

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.
    """

    args_schema: Type[BaseModel] = Fundus_OCT_Input
    model: VisionTransformer = None
    device: Optional[str] = "cuda"
    transform_field: transforms.Compose = None

    name: str = "Fundus_OCT_ClassifierTool"
    description: str = (
        "A tool that analyzes OCT and Fundus images and classifies them for 18 different pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Specific possible disease types include: Retinitis Pigmentosa in Fundus, Retinal Detachment in Fundus, Pterygium in Fundus, Normal in Fundus, Myopia in Fundus, Macular Scar in Fundus, Glaucoma in Fundus, Disc Edema in Fundus, Diabetic Retinopathy in Fundus, Central Serous Chorioretinopathy in Fundus,"
        "Age-related Macular Degeneration in OCT, Choroidal Neovascularization in OCT, Central Serous Retinopathy in OCT, Diabetic Macular Edema in OCT," \
        "Diabetic Retinopathy in OCT, Yellow deposits under the retina in OCT, Macular Hole in OCT, Normal in OCT."
        "Higher values indicate a higher likelihood of the condition being present."
    )

    TRAINING_CLASSES: list = [
        'AMD_F', 'CNV_F', 'CSC_O', 'CSR_F', 'DE_O', 'DME_F', 'DRUSEN_F', 'DR_F', 
        'Diabetic Retinopathy_O', 'Glaucoma_O', 'MH_F', 'MS_O', 'Myopia_O', 'NORMAL_F', 'Normal_O', 'Pterygium_O', 'RD_O', 'RP_O'
    ]
    
    CLASS_MAPPING: dict = {
            "AMD_F": "Age-related Macular Degeneration in OCT",
            "CNV_F": "Choroidal Neovascularization in OCT",
            "CSC_O": "Central Serous Chorioretinopathy in Fundus",
            "CSR_F": "Central Serous Retinopathy in OCT",
            "DE_O": "Disc Edema in Fundus",
            "Diabetic Retinopathy_O": "Diabetic Retinopathy in Fundus",
            "DME_F": "Diabetic Macular Edema in OCT",
            "DR_F": "Diabetic Retinopathy in OCT",
            "DRUSEN_F": "Yellow deposits under the retina in OCT",
            "Glaucoma_O": "Glaucoma in Fundus",
            "MH_F": "Macular Hole in OCT",
            "MS_O": "Macular Scar in Fundus",
            "Myopia_O": "Myopia in Fundus",
            "NORMAL_F": "Normal in OCT",
            "Normal_O": "Normal in Fundus",
            "Pterygium_O": "Pterygium in Fundus",
            "RD_O": "Retinal Detachment in Fundus",
            "RP_O": "Retinitis Pigmentosa in Fundus"
    }

    model: VisionTransformer = None
    device: Optional[str] = "cuda"

    def __init__(self, model_type: str = "R50-ViT-B_16", device: Optional[str] = "cuda", img_size: int = 256):
        super().__init__()
        config = CONFIGS[model_type]
        num_classes = 18
        self.model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        checkpoint_path = "tools/pth/OCT_Fundus_Diagnose.bin"
        device = torch.device('cuda:0')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.transform_field = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process the input Fundus or OCT image for model inference."""
        image = Image.open(image_path).convert('RGB')  # Load as RGB
        image_tensor = self.transform_field(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the pathology classification on the input Fundus or OCT image and return a JSON string."""
        try:
            image_tensor = self._process_image(image_path)

            with torch.no_grad():
                logits = self.model(image_tensor)[0]
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                train_class_probs = probabilities.squeeze().cpu().numpy().tolist()  # Convert to list of Python floats

                output = {}
                for i, train_class in enumerate(self.TRAINING_CLASSES):
                    pathology_name = self.CLASS_MAPPING[train_class]
                    output[pathology_name] = train_class_probs[i]

                sorted_output = sorted(output.items(), key=lambda x: x[1], reverse=True)
                top3 = sorted_output[:3]

                metadata = {
                    "image_path": image_path,
                    "analysis_status": "completed",
                    "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood.",
                    "top_3_predictions": [
                        {"stage": stage, "probability": float(prob)} for stage, prob in top3
                    ]
                }

                result = {
                    "output": output,
                    "metadata": metadata
                }

                return json.dumps(result)

        except Exception as e:
            print(f"Fundus_OCT_ClassifierTool error: {str(e)}")
            traceback.print_exc()
            return json.dumps({
                "output": {},
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
    ) -> str:
        return self._run(image_path)