
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    question: str
    image_path: str
    messages: Annotated[Sequence[str], operator.add]
    final_answer: str

def vlm_tool(question: str, image_path: str) -> str:
    if "color" in question.lower(): return "The object is red."
    elif "count" in question.lower(): return "There are 3 objects."
    else: return "I see a generic object."

def call_model(state: AgentState):
    q = state["question"]
    img = state["image_path"]
    response = vlm_tool(q, img)
    return {"messages": [f"VLM: {response}"], "final_answer": response}

def verifier(state: AgentState):
    # Added verifier node
    ans = state["final_answer"]
    return {"messages": ["Verified: " + ans]}

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("model", call_model)
    workflow.add_node("verifier", verifier) # Added node
    workflow.set_entry_point("model")
    workflow.add_edge("model", "verifier")
    workflow.add_edge("verifier", END)
    app = workflow.compile()
    return app
