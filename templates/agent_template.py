from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    image_path: str
    messages: Annotated[Sequence[str], operator.add]
    final_answer: str

# --- Mock VLM Tool ---
# In a real scenario, this would call GPT-4V or InternVL
def vlm_tool(question: str, image_path: str) -> str:
    # Improve this mock logic for better testing
    if "color" in question.lower():
        return "The object is red."
    elif "count" in question.lower():
        return "There are 3 objects."
    else:
        return "I see a generic object."

# --- Nodes ---
def call_model(state: AgentState):
    """Simple node that calls the VLM tool directly."""
    q = state["question"]
    img = state["image_path"]
    
    # Simple prompt logic (Evolution target)
    response = vlm_tool(q, img)
    
    return {"messages": [f"VLM Output: {response}"], "final_answer": response}

# --- Graph Construction Factory ---
def create_graph():
    """Factory function to build and return the graph app."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("model", call_model)
    
    # Set entry point
    workflow.set_entry_point("model")
    
    # Set edges (Simple linear flow)
    workflow.add_edge("model", END)
    
    # Compile
    app = workflow.compile()
    return app
