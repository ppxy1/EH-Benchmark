import warnings
from typing import *
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import base64
import json
import os
import re
import time
from agent import *
from tools.dr import DR_ClassifierTool
from tools.diagnose import Fundus_OCT_ClassifierTool
from tools.fundus_segmentation import FundusSegmentationTool
from tools.oct_segmentation import OCTSegmentationTool
from tools.detection import FundusYOLODetectionTool
from tools.utils import ImageVisualizerTool
from utils import *
from langchain_core.tools import BaseTool
from agent.agent import Agent, DecisionMakerTool, EvaluatorTool
from tools.rag import RAGTool
from pathlib import Path
from tqdm import tqdm
import csv

class Config:
    """Configuration class for agent settings."""
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    ALLOWED_IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    DEFAULT_TOOLS = [
        "ImageVisualizerTool", "Fundus_OCT_ClassifierTool", "DR_ClassifierTool",
        "FundusYOLODetectionTool", "FundusSegmentationTool", "OCTSegmentationTool",
        "RAGTool", "DecisionMakerTool", "EvaluatorTool"
    ]
    RAG_URLS = [
        "https://www.rnib.org.uk/your-eyes/eye-conditions-az/"
        "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases",
        "https://www.health.tas.gov.au/health-topics/eyes-and-vision-ophthalmology/common-eye-disorders",
        "https://www.allaboutvision.com/eye-care/eye-anatomy/eye-structure/fundus/",
        "https://www.allaboutvision.com/conditions/injuries/",
        "https://eyewiki.org/Category:Articles",
        "https://eyewiki.org/Primary_Open-Angle_Glaucoma",
        "https://www.aao.org/education/munnerlyn-laser-surgery-center/angleclosure-glaucoma-19"
    ]
    MODEL = "gpt-4.1"
    TEMPERATURE = 0.7
    TOP_P = 0.95
    TEMP_DIR = "temp"
    DEVICE = "cuda"
    PROMPT_FILE = "docs/system_prompts.txt"

class ToolFactory:
    """Factory for creating tool instances."""
    @staticmethod
    def create_tools(tools_to_use: List[str], config: Config) -> Dict[str, BaseTool]:
        all_tools = {
            "Fundus_OCT_ClassifierTool": lambda: Fundus_OCT_ClassifierTool(device=config.DEVICE),
            "FundusSegmentationTool": lambda: FundusSegmentationTool(device=config.DEVICE),
            "OCTSegmentationTool": lambda: OCTSegmentationTool(device=config.DEVICE),
            "FundusYOLODetectionTool": lambda: FundusYOLODetectionTool(),
            "DR_ClassifierTool": lambda: DR_ClassifierTool(device=config.DEVICE),
            "ImageVisualizerTool": lambda: ImageVisualizerTool(),
            "EvaluatorTool": lambda: EvaluatorTool(),
            "RAGTool": lambda: RAGTool(openai_api_key=config.OPENAI_API_KEY, url_list=config.RAG_URLS),
        }
        tools_dict = {}
        for tool_name in tools_to_use:
            if tool_name in all_tools:
                tools_dict[tool_name] = all_tools[tool_name]()
            elif tool_name != "DecisionMakerTool":
                print(f"Warning: Tool {tool_name} not recognized.")
        
        if "DecisionMakerTool" in tools_to_use:
            tools_dict["DecisionMakerTool"] = DecisionMakerTool(tools=list(tools_dict.values()))
        
        return tools_dict

class MedicalAgent:
    """Main class for initializing and running the medical agent."""
    def __init__(self, config: Config, tools_to_use: Optional[List[str]] = None):
        self.config = config
        self.tools_to_use = tools_to_use or config.DEFAULT_TOOLS
        self.agent, self.tools_dict = self._initialize_agent()

    def _load_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            prompts = load_prompts_from_file(self.config.PROMPT_FILE)
            prompt = prompts.get("MEDICAL_ASSISTANT", "")
            if not prompt:
                raise ValueError("MEDICAL_ASSISTANT prompt not found")
            return prompt
        except Exception as e:
            raise Exception(f"Error loading prompts: {e}")

    def _initialize_agent(self) -> Tuple[Agent, Dict[str, BaseTool]]:
        """Initialize the agent and tools."""
        try:
            prompt = self._load_prompt()
            tools_dict = ToolFactory.create_tools(self.tools_to_use, self.config)
            checkpointer = MemorySaver()
            model = ChatOpenAI(
                model=self.config.MODEL,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P
            )
            agent = Agent(
                model=model,
                tools=list(tools_dict.values()),
                log_tools=True,
                log_dir=self.config.TEMP_DIR,
                system_prompt=prompt,
                checkpointer=checkpointer,
            )
            print("Agent initialized successfully")
            return agent, tools_dict
        except Exception as e:
            raise Exception(f"Error initializing agent: {e}")

    def process_image(self, image_path: str, title: str, thread_id: Optional[str] = None) -> List[str]:
        """Process an image with the agent."""
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
        if Path(image_path).suffix.lower() not in self.config.ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(f"Only {', '.join(self.config.ALLOWED_IMAGE_SUFFIXES)} formats supported")

        try:
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            query_text = f"Image file path: {image_path}\nQuestion: {title}"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }]
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {e}")

        thread_id = thread_id or str(time.time())
        results = []
        try:
            for event in self.agent.workflow.stream({"messages": messages}, {"configurable": {"thread_id": thread_id}}):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        if content:
                            results.append(content)
                    elif "execute" in event:
                        for message in event["execute"]["messages"]:
                            try:
                                tool_result = json.loads(message.content)
                                if tool_result:
                                    results.append(str(tool_result))
                            except json.JSONDecodeError as e:
                                results.append(f"Error parsing tool result: {e}")
            return results
        except Exception as e:
            print(f"Error processing request: {e}")
            return [f"Error processing request: {str(e)}"]

    def process_json(self, json_file: str, output_csv: str = "result.csv") -> float:
        """Process JSON data and write results to CSV."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file {json_file}: {e}")
            return 0.0

        total_items = 0
        correct_total = 0
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'question', 'correct_answer', 'model_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in tqdm(data, desc="Processing JSON entries"):
                image_id = entry.get('id')
                question = entry.get('question')
                correct_answer = entry.get('answer', '').upper()
                if not all([image_id, question, correct_answer]):
                    print(f"Skipping entry due to missing data: {entry}")
                    continue

                image_path = os.path.join(image_id)
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                try:
                    results = self.process_image(image_path, question)
                    model_response = results[-1] if results else "ERROR"
                    match = re.search(r'\b[A-E]\b', model_response.upper())
                    model_prediction = match.group(0) if match else "ERROR"
                    print(f"ID: {image_id}, Correct: {correct_answer}, Predicted: {model_prediction}")

                    writer.writerow({
                        'id': image_id,
                        'question': question,
                        'correct_answer': correct_answer,
                        'model_prediction': model_prediction
                    })

                    total_items += 1
                    if model_prediction == correct_answer:
                        correct_total += 1

                except Exception as e:
                    print(f"Error processing {image_id}: {e}")
                    writer.writerow({
                        'id': image_id,
                        'question': question,
                        'correct_answer': correct_answer,
                        'model_prediction': "ERROR"
                    })
                    total_items += 1

        accuracy = correct_total / total_items if total_items else 0
        print(f"Overall accuracy: {accuracy:.2%}")
        return accuracy

def main():
    """Main function to run the medical agent."""
    load_dotenv()
    config = Config()
    try:
        agent = MedicalAgent(config)
        json_file = "JSON_FILE"
        agent.process_json(json_file)
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()