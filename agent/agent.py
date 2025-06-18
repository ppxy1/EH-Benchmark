import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import Field
import traceback

_ = load_dotenv()

class ToolCallLog(TypedDict):
    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    query: Optional[str]
    rag_context: Optional[str]
    tool_outputs: Annotated[List[Dict], operator.add]
    image_path: Optional[str]
    selection_tools: Optional[List[str]]
    decision_tools: Optional[List[str]]

class DecisionMakerTool(BaseTool):
    name: str = "DecisionMaker"
    description: str = "Decides which tools to use and in what order based on the user's query."
    model: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4.1", temperature=0.15))
    tools: List[BaseTool] = Field(default_factory=list)

    def __init__(self, tools: List[BaseTool]):
        super().__init__()
        self.tools = tools

    def _run(self, input: str, rag_context: Optional[str] = None) -> str:
        query = input
        tool_names = [t.name for t in self.tools if t.name not in ["DecisionMaker", "RAGTool", "SelectionTool", "Evaluator"]]
        prompt = (
            f"You are a decision-making assistant for medical imaging queries. "
            f"Guidelines:\n"
            f"- For queries about diabetic retinopathy or its stages (e.g., pre-proliferative, non-proliferative, proliferative), select 'DR_ClassifierTool' as it is designed to classify diabetic retinopathy stages from fundus images.\n"
            f"- For queries involving OCT image analysis, select 'OCTSegmentationTool'.\n"
            f"- If you need to diagnose different types of ophthalmic problems in Fundus or OCT: e.g. Age-related Macular Degeneration, Choroidal Neovascularization in OCT, etc., select 'Fundus_OCT_ClassifierTool'.\n"
            f"- For queries needing segmentation of position only the optic cup and optic disc, calculate the cup-to-disc ratio in Fundus image, select 'FundusSegmentationTool'.\n"
            f"- For complex problems, you may choose to invoke more than one tool."
            f"- But, If every tool doesn't meet the call requirements, you can select none of them"
            f"Given the query: '{query}', choose tools from: {tool_names}. "
            "Respond with JSON: {\"tools\": [..], \"reasoning\": \"...\"}"
        )
        messages = [
            SystemMessage(content="You are a decision-making assistant."),
            HumanMessage(content=prompt),
        ]
        try:
            resp = self.model.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else "{}"
            if content.startswith("```json") and content.endswith("```"):
                content = content.split("```json")[1].split("```")[0].strip()
            json.loads(content)
            return content
        except Exception as e:
            return json.dumps({"tools": [], "reasoning": f"error: {e}"})

    async def _arun(self, input: str, rag_context: Optional[str] = None) -> str:
        return self._run(input, rag_context)

class SelectionTool(BaseTool):
    name: str = "SelectionTool"
    description: str = "Determine the required tools based on the query."
    model: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4.1", temperature=0.3))

    def _run(self, query: str) -> str:
        prompt = (
            f"You are a tool selection assistant for medical imaging queries. Your task is to select the most appropriate tools "
            f"from the following list based on the query: '{query}': "
            f"Guidelines:\n"
            f"- For queries about diabetic retinopathy or its stages (e.g., pre-proliferative, non-proliferative, proliferative), select 'DR_ClassifierTool' as it is designed to classify diabetic retinopathy stages from fundus images.\n"
            f"- For queries involving OCT image analysis, select 'OCTSegmentationTool'.\n"
            f"- If you need to diagnose different types of ophthalmic problems in Fundus or OCT: e.g. Age-related Macular Degeneration, Choroidal Neovascularization in OCT, etc., select 'Fundus_OCT_ClassifierTool'.\n"
            f"- For queries needing segmentation of position only the optic cup and optic disc, calculate the cup-to-disc ratio in Fundus image, select 'FundusSegmentationTool'.\n"
            f"- Please select the appropriate tool, inappropriate tools should not be selected."
            f"- But, If every tool doesn't meet the requirements, you can select none of them"
            "Respond with JSON: {\"tools\": [\"tool1\", \"tool2\", ...]}.\n"
            "Output ONLY the JSON object, e.g., {\"tools\": [..]}."
        )
        messages = [
            SystemMessage(content="You are a tool selection assistant for medical imaging."),
            HumanMessage(content=prompt),
        ]
        try:
            resp = self.model.invoke(messages)
            content = str(resp.content).strip()
            if content.startswith("```json") and content.endswith("```"):
                content = content.split("```json")[1].split("```")[0].strip()
            json.loads(content)
            parsed_content = json.loads(content)
            if "tools" not in parsed_content:
                content = json.dumps({"tools": []})
            return content
        except Exception as e:
            return json.dumps({"tools": []})

class EvaluatorTool(BaseTool):
    name: str = "Evaluator"
    description: str = "Evaluates the correctness and completeness of the final answer and verifies the consistency of tool selection and order."
    model: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4.1", temperature=0.3))
    selection_tool: BaseTool = Field(default=None)

    def __init__(self, selection_tool: BaseTool):
        super().__init__()
        self.selection_tool = selection_tool

    def _run(self, input: str) -> str:
        try:
            payload = json.loads(input)
            query = payload.get("query", "")
            answer = payload.get("answer", None)
            outputs = payload.get("outputs", None)
            decision_tools = payload.get("decision_tools", [])
            selection_tools = payload.get("selection_tools", [])
            rag_context = payload.get("rag_context", None)
        except Exception as e:
            return json.dumps({
                "is_correct": False,
                "is_complete": False,
                "is_followed": False,
                "feedback": f"EvaluatorTool: Invalid input - {str(e)}"
            })
        is_followed = selection_tools == decision_tools
        followed_feedback = (
            "Tool selection and order are identical between SelectionTool and DecisionMakerTool."
            if is_followed
            else f"Tool selection or order differs: SelectionTool returned {selection_tools}, DecisionMakerTool returned {decision_tools}."
        )
        if answer is None or outputs is None:
            feedback = f"{followed_feedback}\nNote: Full evaluation (is_correct, is_complete) requires answer and outputs, which are not available yet."
            result = {
                "is_correct": False,
                "is_complete": False,
                "is_followed": is_followed,
                "feedback": feedback,
            }
            return json.dumps(result)
        prompt = (
            f"Query: {query or 'No query provided'}\n\n"
            f"Answer: {answer or 'No answer provided'}\n\n"
            f"Tool outputs: {json.dumps(outputs) if outputs else 'No tool outputs provided'}\n\n"
            f"RAG context: {json.dumps(rag_context) if rag_context else 'None'}\n\n"
            "Evaluate the correctness and completeness of the answer based on the following criteria:\n"
            "- Correctness: The answer is correct if it contains any relevant information related to the tool outputs' findings or their associated diagnoses, even if it's not the exact match."
            "- Completeness: The answer is complete if it addresses the query, such as selecting an option for multiple-choice queries or providing a response for open-ended queries. Partial responses that address the query's intent are considered complete.\n"
            "Return a valid JSON object: {\"is_correct\": boolean, \"is_complete\": boolean, \"is_followed\": boolean, \"feedback\": \"string\"}. "
            "The feedback should explain why the answer is correct/incorrect and complete/incomplete."
        )
        messages = [
            SystemMessage(content="You are a senior ophthalmology evaluation expert."),
            HumanMessage(content=prompt),
        ]
        try:
            resp = self.model.invoke(messages)
            content = str(resp.content) if resp.content is not None else "{}"
            if content.startswith("```json") and content.endswith("```"):
                content = content.split("```json")[1].split("```")[0].strip()
            if not content.strip():
                raise ValueError("Model returned an empty response")
            evaluation = json.loads(content)
            is_correct = evaluation.get("is_correct", False)
            is_complete = evaluation.get("is_complete", False)
            feedback = evaluation.get("feedback", "")
            final_feedback = f"{feedback}\n{followed_feedback}"
            result = {
                "is_correct": is_correct,
                "is_complete": is_complete,
                "is_followed": is_followed,
                "feedback": final_feedback,
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({
                "is_correct": False,
                "is_complete": False,
                "is_followed": is_followed,
                "feedback": f"Evaluation error: {str(e)}\n{followed_feedback}"
            })

class Agent:
    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        *,
        system_prompt: str = "",
        checkpointer: Any = None,
        log_tools: bool = True,
        log_dir: str | Path = "logs",
    ) -> None:
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        if log_tools:
            self.log_path = Path(log_dir)
            self.log_path.mkdir(parents=True, exist_ok=True)
        self.tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self.model = model.bind_tools(
            [t for t in tools if t.name not in {"DecisionMaker", "Evaluator", "RAGTool"}]
        )
        workflow = StateGraph(AgentState)
        workflow.add_node("rag", self.rag)
        workflow.add_node("decide", self.decide)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_node("process", self.process_request)
        workflow.add_node("evaluate", self.evaluate)
        workflow.add_edge("rag", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_edge("execute", "process")
        workflow.add_edge("process", "evaluate")
        workflow.set_entry_point("rag")
        self.workflow = workflow.compile(checkpointer=checkpointer)

    def rag(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        query = None
        image_path = None
        if messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
            content = messages[-1]["content"]
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text" and "text" in item:
                        text = item["text"]
                        query = text
                        if "Image file path:" in text:
                            image_path = text.split("Image file path:")[1].split("\n")[0].strip()
                    elif item.get("type") == "image_url":
                        pass
        elif messages and isinstance(messages[-1], HumanMessage):
            query = messages[-1].content
            if "Image file path:" in query:
                image_path = query.split("Image file path:")[1].split("\n")[0].strip()
        if not query:
            query = ""
        rag_tool = self.tools.get("RAGTool")
        if rag_tool is None:
            return {
                "messages": [SystemMessage(content="Error: RAGTool not available.")],
                "query": query,
                "rag_context": None,
                "tool_outputs": [],
                "image_path": image_path,
            }
        try:
            rag_ctx_raw = rag_tool.invoke(query)
            rag_ctx = json.loads(rag_ctx_raw) if isinstance(rag_ctx_raw, str) else rag_ctx_raw
            msg = SystemMessage(
                content="RAG context retrieved:\n"
                + (json.dumps(rag_ctx) if isinstance(rag_ctx, dict) else str(rag_ctx))
            )
            return {
                "messages": [msg],
                "query": query,
                "rag_context": rag_ctx,
                "tool_outputs": [],
                "image_path": image_path,
            }
        except Exception as e:
            return {
                "messages": [SystemMessage(content=f"Error in RAGTool: {e}")],
                "query": query,
                "rag_context": None,
                "tool_outputs": [],
                "image_path": image_path,
            }

    def decide(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        rag_context = state.get("rag_context", None)
        image_path = state.get("image_path", None)
        if not query:
            return {
                "messages": [SystemMessage(content="Error: empty query")],
                "query": query,
                "rag_context": rag_context,
                "tool_outputs": [],
                "image_path": image_path,
                "decision_tools": [],
                "selection_tools": []
            }
        dm_tool = self.tools.get("DecisionMaker")
        sel_tool = self.tools.get("SelectionTool")
        ev_tool = self.tools.get("Evaluator")
        if dm_tool is None or sel_tool is None or ev_tool is None:
            return {
                "messages": [SystemMessage(content="Error: DecisionMaker, SelectionTool, or Evaluator missing")],
                "query": query,
                "rag_context": rag_context,
                "tool_outputs": [],
                "image_path": image_path,
                "decision_tools": [],
                "selection_tools": []
            }
        try:
            decision_raw = dm_tool.invoke(query)
            decision = json.loads(decision_raw)
            decision_tools = decision.get("tools", [])
            selection_raw = sel_tool.invoke(query)
            selection = json.loads(selection_raw)
            selection_tools = selection.get("tools", [])
            reasoning = ""
            if not set(selection_tools).intersection(set(decision_tools)):
                reasoning = "SelectionTool and DecisionMaker tools do not match. Retrying DecisionMaker."
                payload = json.dumps({
                    "query": query,
                    "decision_tools": decision_tools,
                    "selection_tools": selection_tools,
                    "rag_context": rag_context
                })
                ev_raw = ev_tool.invoke(payload)
                ev_result = json.loads(ev_raw)
                reasoning += f"\nEvaluatorTool feedback: {ev_result.get('feedback', '')}"
                return {
                    "messages": [SystemMessage(content=reasoning)],
                    "query": query,
                    "rag_context": rag_context,
                    "tool_outputs": [],
                    "image_path": image_path,
                    "decision_tools": [],
                    "selection_tools": selection_tools
                }
            elif len(selection_tools) > len(decision_tools):
                additional_tools = list(set(selection_tools) - set(decision_tools))
                decision_tools.extend(additional_tools)
                reasoning = f"Added tools from SelectionTool: {additional_tools}"
                payload = json.dumps({
                    "query": query,
                    "decision_tools": decision_tools,
                    "selection_tools": selection_tools,
                    "rag_context": rag_context
                })
                ev_raw = ev_tool.invoke(payload)
                ev_result = json.loads(ev_raw)
                reasoning += f"\nEvaluatorTool feedback: {ev_result.get('feedback', '')}"
            else:
                reasoning = decision.get("reasoning", "Tool selection matches or SelectionTool has fewer tools.")
            tool_calls = [
                {
                    "id": f"call_{i}",
                    "name": tool_name,
                    "args": {"image_path": image_path, "query query": query} if image_path and tool_name in ["DR_ClassifierTool", "Fundus_OCT_ClassifierTool", "FundusSegmentationTool", "OCTSegmentationTool", "FundusYOLODetectionTool"] else {"query": query},
                    "type": "tool_call",
                }
                for i, tool_name in enumerate(decision_tools)
            ]
            return {
                "messages": [AIMessage(content=reasoning, tool_calls=tool_calls)],
                "query": query,
                "rag_context": rag_context,
                "tool_outputs": [],
                "image_path": image_path,
                "decision_tools": decision_tools,
                "selection_tools": selection_tools
            }
        except Exception as e:
            return {
                "messages": [SystemMessage(content=f"DecisionMaker, SelectionTool, or Evaluator error: {e}")],
                "query": query,
                "rag_context": rag_context,
                "tool_outputs": [],
                "image_path": image_path,
                "decision_tools": [],
                "selection_tools": []
            }

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        query = state.get("query", "")
        rag_context = state.get("rag_context", None)
        if rag_context:
            context_msg = SystemMessage(
                content=f"Use the following retrieved context to inform your response:\n{json.dumps(rag_context) if isinstance(rag_context, dict) else str(rag_context)}"
            )
            messages = [context_msg] + messages
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        try:
            response = self.model.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error in processing request: {e}")]}

    @staticmethod
    def has_tool_calls(state: AgentState) -> bool:
        resp = state["messages"][-1]
        return bool(getattr(resp, "tool_calls", None))

    def execute_tools(self, state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls'):
            return {
                "messages": [SystemMessage(content="Error: The last message is not an AIMessage with tool calls.")],
                "tool_outputs": [],
                "image_path": state.get("image_path")
            }
        tool_calls = last_message.tool_calls
        tool_outputs = state.get("tool_outputs", [])
        result_messages: list[ToolMessage] = []
        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {})
            tool = self.tools.get(name)
            if tool is None:
                result_json = json.dumps({"error": f"Tool '{name}' not found"})
            else:
                try:
                    input_val = args.get("image_path") if name in ["DR_ClassifierTool", "Fundus_OCT_ClassifierTool", "FundusSegmentationTool", "OCTSegmentationTool", "FundusYOLODetectionTool"] else args.get("query", "")
                    if not input_val:
                        raise ValueError(f"Tool {name} has no valid input")
                    result_json = tool.invoke(input_val)
                except Exception as e:
                    result_json = json.dumps({"error": str(e)})
            try:
                parsed = json.loads(result_json) if isinstance(result_json, str) else result_json
            except json.JSONDecodeError as e:
                parsed = {"error": f"Invalid JSON: {e}"}
            tool_outputs.append(parsed)
            result_messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=name,
                    args=args,
                    content=str(result_json),
                )
            )
        self._save_tool_calls(result_messages)
        return {"messages": result_messages, "tool_outputs": tool_outputs, "image_path": state.get("image_path")}

    def evaluate(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        query = state.get("query", "")
        tool_outputs = state.get("tool_outputs", [])
        decision_tools = state.get("decision_tools", [])
        selection_tools = state.get("selection_tools", [])
        rag_context = state.get("rag_context", None)
        if not query:
            return {"messages": [AIMessage(content="Error: Empty query for evaluation.")]}
        if not tool_outputs:
            return {"messages": [AIMessage(content="Error: No tool outputs for evaluation.")]}
        final_answer = next(
            (m.content for m in reversed(messages)
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
            "No final answer generated."
        )
        payload = json.dumps({
            "query": query,
            "answer": final_answer,
            "outputs": tool_outputs,
            "decision_tools": decision_tools,
            "selection_tools": selection_tools,
            "rag_context": rag_context
        })
        ev_tool = self.tools.get("Evaluator")
        if ev_tool is None:
            return {"messages": [AIMessage(content="Error: Evaluator tool missing.")]}
        ev_raw = ev_tool.invoke(payload)
        return {"messages": [AIMessage(content=ev_raw)]}
    
    def _save_tool_calls(self, tool_messages: List[ToolMessage]) -> None:
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.log_path / f"tool_calls_{timestamp}.json"

        serialized: list[ToolCallLog] = []
        for msg in tool_messages:
            serialized.append(
                ToolCallLog(
                    timestamp=datetime.now().isoformat(),
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    args=msg.args,
                    content=msg.content,
                )
            )

        with open(file_path, "w") as f:
            json.dump(serialized, f, indent=4, ensure_ascii=False)