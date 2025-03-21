from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from enum import Enum
import json

class AgentState(TypedDict):
    """State for the agent."""
    # Input
    input: str
    # History of the conversation so far
    chat_history: List[Dict[str, str]]
    # Current plan for addressing the task
    plan: Optional[List[str]]
    # Current tool being used
    current_tool: Optional[str]
    # Input for current tool
    tool_input: Optional[Dict[str, Any]]
    # Output from current tool
    tool_output: Optional[Dict[str, Any]]
    # List of errors encountered
    errors: List[Dict[str, Any]]
    # Attempt to fix the current error
    error_fix_attempts: int
    # Final output to show to the user
    final_output: Optional[str]
    # Additional context the agent might need
    context: Dict[str, Any]
    # Flag to indicate debugging mode
    debug: bool

class NodeNames(str, Enum):
    """Node names for the graph."""
    START = "start"
    PLAN = "plan"
    CHOOSE_TOOL = "choose_tool"
    EXECUTE_TOOL = "execute_tool"
    PROCESS_TOOL_OUTPUT = "process_tool_output"
    HANDLE_ERROR = "handle_error"
    GENERATE_FINAL_OUTPUT = "generate_final_output"
    END = "end"

class EdgeNames(str, Enum):
    """Edge names for the graph."""
    PLAN = "plan"
    CHOOSE_TOOL = "choose_tool"
    EXECUTE_TOOL = "execute_tool"
    PROCESS_OUTPUT = "process_output"
    ERROR = "error"
    RETRY = "retry"
    GENERATE_OUTPUT = "generate_output"
    FINISH = "finish"
    
def create_initial_state() -> AgentState:
    """Create an initial state for the agent."""
    return {
        "input": "",
        "chat_history": [],
        "plan": None,
        "current_tool": None,
        "tool_input": None,
        "tool_output": None,
        "errors": [],
        "error_fix_attempts": 0,
        "final_output": None,
        "context": {},
        "debug": False
    }

def add_message_to_history(state: AgentState, role: str, content: str) -> AgentState:
    """Add a message to the chat history."""
    state["chat_history"].append({"role": role, "content": content})
    return state

def pretty_print_state(state: AgentState) -> None:
    """Pretty print the state for debugging."""
    # Copy state to avoid modifying the original
    state_copy = state.copy()
    
    # Remove chat history to make it cleaner
    if "chat_history" in state_copy:
        state_copy["chat_history"] = f"[{len(state_copy['chat_history'])} messages]"
    
    # Format tool input/output for better readability
    for key in ["tool_input", "tool_output", "context"]:
        if key in state_copy and state_copy[key]:
            try:
                state_copy[key] = json.dumps(state_copy[key], indent=2)
            except:
                pass
    
    # Print the state
    print("==== AGENT STATE ====")
    for key, value in state_copy.items():
        print(f"{key}: {value}")
    print("=====================")