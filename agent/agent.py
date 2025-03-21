from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
import traceback

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

from .state import AgentState, NodeNames, EdgeNames, add_message_to_history

# Import tools
from tools.code_execution import execute_code, code_interpreter
from tools.file_operations import read_file, write_file, get_dataframe
from tools.azure_tools import create_search_index, upload_to_search_index, search_index

# Import OpenAI client
from utils.openai_client import get_langchain_openai_client

# Define available tools
TOOLS = {
    "execute_code": execute_code,
    "code_interpreter": code_interpreter,
    "read_file": read_file,
    "write_file": write_file,
    "get_dataframe": get_dataframe,
    "create_search_index": create_search_index,
    "upload_to_search_index": upload_to_search_index,
    "search_index": search_index
}

# System prompts
SYSTEM_PROMPT = """You are an advanced AI assistant with the ability to use various tools to help users. 
Your capabilities include:

1. Code execution & interpretation
2. File reading and writing
3. Azure Cognitive Search operations for creating indexes and uploading data

Your goal is to help the user accomplish their tasks by following these steps:
1. Understand the user's request
2. Create a plan to address it
3. Choose and execute appropriate tools
4. Handle any errors adaptively
5. Provide a clear, helpful response

Be thorough but concise in your explanations.
"""

PLANNING_PROMPT = """Based on the user's request, create a step-by-step plan to accomplish the task.
Be thorough but concise. Focus on how to use the available tools effectively.

Available tools:
- execute_code: Run Python code and get results
- code_interpreter: Run Python code to answer specific questions
- read_file: Read from a file
- write_file: Write to a file
- get_dataframe: Load data from a file into a pandas DataFrame
- create_search_index: Create an Azure Cognitive Search index
- upload_to_search_index: Upload data to an Azure Search index
- search_index: Search an Azure Cognitive Search index

User's request: {input}

Your plan should be a list of steps, with each step being a specific action to take.
"""

TOOL_SELECTION_PROMPT = """Based on the user's request and your plan, select the most appropriate tool to use next.

Available tools:
- execute_code: Run Python code and get results
- code_interpreter: Run Python code to answer specific questions
- read_file: Read from a file
- write_file: Write to a file
- get_dataframe: Load data from a file into a pandas DataFrame
- create_search_index: Create an Azure Cognitive Search index
- upload_to_search_index: Upload data to an Azure Search index
- search_index: Search an Azure Cognitive Search index

Current plan: {plan}
Current progress: {chat_history}

Select a tool from the available list and specify the required input parameters.
Respond in JSON format:
```json
{{
  "tool": "tool_name",
  "tool_input": {{
    "param1": "value1",
    "param2": "value2"
  }},
  "reasoning": "Brief explanation of why this tool was chosen"
}}
```
"""

TOOL_PROCESSING_PROMPT = """Process the output from the tool and determine the next steps.
If the tool execution was successful, update the plan and decide what to do next.
If there was an error, we'll need to handle it.

Tool used: {current_tool}
Tool input: {tool_input}
Tool output: {tool_output}

Current plan: {plan}
Current progress: {chat_history}

Based on the tool output, decide what to do next:
1. Continue with the plan (if the tool executed successfully)
2. Report an error (if there was an error that needs handling)
3. Generate the final output (if we've completed the plan)

Respond in JSON format:
```json
{{
  "decision": "continue_plan | report_error | generate_output",
  "reasoning": "Brief explanation of your decision",
  "updated_plan": ["Step 1", "Step 2", ...] (if applicable)
}}
```
"""

ERROR_HANDLING_PROMPT = """An error occurred during tool execution. Let's try to fix it and adapt our approach.

Tool used: {current_tool}
Tool input: {tool_input}
Error: {tool_output}

Current plan: {plan}
Current progress: {chat_history}
Previous errors: {errors}
Fix attempts: {error_fix_attempts}

Analyze the error and provide a solution to fix it. Be adaptive in your approach.
If the error persists after multiple attempts, consider an alternative approach.

Respond in JSON format:
```json
{{
  "error_analysis": "Analysis of what went wrong",
  "solution": "Proposed solution",
  "updated_tool": "tool_name (same or different tool)",
  "updated_tool_input": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```
"""

FINAL_OUTPUT_PROMPT = """Based on all the interactions and tool executions, generate a comprehensive final response for the user.

User's original request: {input}
Plan executed: {plan}
Progress and results: {chat_history}

Provide a clear, concise summary of what was accomplished and any relevant results or outputs.
If there were any limitations or issues, mention them briefly along with any suggested next steps.
"""

def plan(state: AgentState) -> AgentState:
    """Create a step-by-step plan based on the user's request."""
    llm = get_langchain_openai_client()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=PLANNING_PROMPT.format(input=state["input"]))
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"chat_history": state["chat_history"]})
    
    # Extract the plan from the response
    plan_text = response.content
    
    # Convert the plan text to a list
    plan_lines = plan_text.strip().split("\n")
    plan_list = [line.strip() for line in plan_lines if line.strip() and not line.strip().startswith("#")]
    
    # Update the state
    state["plan"] = plan_list
    state = add_message_to_history(state, "assistant", f"I'll help you with this. Here's my plan:\n\n{plan_text}")
    
    return state

def choose_tool(state: AgentState) -> AgentState:
    """Choose the appropriate tool based on the plan."""
    llm = get_langchain_openai_client()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=TOOL_SELECTION_PROMPT.format(
            plan=state["plan"],
            chat_history=state["chat_history"]
        ))
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"chat_history": state["chat_history"]})
    
    # Extract the JSON from the response
    response_text = response.content
    
    try:
        # Extract JSON object if it's wrapped in ```json ... ```
        if "```json" in response_text and "```" in response_text.split("```json", 1)[1]:
            json_str = response_text.split("```json", 1)[1].split("```", 1)[0]
            tool_selection = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            tool_selection = json.loads(response_text)
        
        # Update the state
        state["current_tool"] = tool_selection["tool"]
        state["tool_input"] = tool_selection["tool_input"]
        
        # Add to chat history
        reasoning = tool_selection.get("reasoning", "")
        state = add_message_to_history(
            state, 
            "assistant", 
            f"I'll use the {state['current_tool']} tool with these parameters: {json.dumps(state['tool_input'], indent=2)}\n\nReasoning: {reasoning}"
        )
        
    except (json.JSONDecodeError, KeyError) as e:
        # If there's an error parsing the JSON, add it to the errors list
        state["errors"].append({
            "type": "json_parse_error",
            "message": str(e),
            "response": response_text
        })
        
        # Still need to set a default tool to avoid breaking the flow
        state["current_tool"] = "execute_code"
        state["tool_input"] = {"code": "print('Error parsing tool selection JSON')"}
    
    return state

def execute_tool(state: AgentState) -> AgentState:
    """Execute the selected tool with the provided input."""
    tool_name = state["current_tool"]
    tool_input = state["tool_input"]
    
    # Check if the tool exists
    if tool_name not in TOOLS:
        state["tool_output"] = {
            "success": False,
            "error": f"Tool '{tool_name}' is not available. Available tools are: {', '.join(TOOLS.keys())}"
        }
        return state
    
    # Get the tool
    tool = TOOLS[tool_name]
    
    try:
        # Execute the tool
        tool_output = tool(**tool_input)
        state["tool_output"] = tool_output
        
    except Exception as e:
        # If there's an error executing the tool, capture it
        error_msg = f"Error executing {tool_name}: {str(e)}\n{traceback.format_exc()}"
        state["tool_output"] = {
            "success": False,
            "error": error_msg
        }
        
        # Add to errors list
        state["errors"].append({
            "type": "tool_execution_error",
            "tool": tool_name,
            "input": tool_input,
            "message": str(e),
            "traceback": traceback.format_exc()
        })
    
    return state

def process_tool_output(state: AgentState) -> Union[Tuple[str, AgentState], AgentState]:
    """Process the output from the tool and decide next steps."""
    llm = get_langchain_openai_client()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=TOOL_PROCESSING_PROMPT.format(
            current_tool=state["current_tool"],
            tool_input=state["tool_input"],
            tool_output=state["tool_output"],
            plan=state["plan"],
            chat_history=state["chat_history"]
        ))
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"chat_history": state["chat_history"]})
    
    # Extract the JSON from the response
    response_text = response.content
    
    try:
        # Extract JSON object if it's wrapped in ```json ... ```
        if "```json" in response_text and "```" in response_text.split("```json", 1)[1]:
            json_str = response_text.split("```json", 1)[1].split("```", 1)[0]
            decision = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            decision = json.loads(response_text)
        
        # Update the state based on the decision
        if "updated_plan" in decision:
            state["plan"] = decision["updated_plan"]
        
        # Add the reasoning to chat history
        tool_result = "success" if state["tool_output"].get("success", False) else "failure"
        message = f"Tool {state['current_tool']} execution result: {tool_result}\n\n"
        
        if state["tool_output"].get("success", False):
            # Format successful output nicely
            output_str = str(state["tool_output"])
            if len(output_str) > 500:
                output_str = output_str[:250] + "\n...\n" + output_str[-250:]
            message += f"Output: {output_str}\n\n"
        else:
            # Format error nicely
            error = state["tool_output"].get("error", "Unknown error")
            message += f"Error: {error}\n\n"
        
        message += f"Reasoning: {decision.get('reasoning', '')}"
        state = add_message_to_history(state, "assistant", message)
        
        # Return the appropriate next state based on the decision
        decision_type = decision.get("decision", "").lower()
        
        if decision_type == "report_error":
            # We need to handle an error
            return EdgeNames.ERROR, state
        elif decision_type == "generate_output":
            # We're done and need to generate the final output
            return EdgeNames.GENERATE_OUTPUT, state
        else:
            # Continue with the plan (default)
            return EdgeNames.CHOOSE_TOOL, state
        
    except (json.JSONDecodeError, KeyError) as e:
        # If there's an error parsing the JSON, add it to the errors list
        state["errors"].append({
            "type": "json_parse_error",
            "message": str(e),
            "response": response_text
        })
        
        # If we can't parse the response, assume we need to continue with the plan
        return EdgeNames.CHOOSE_TOOL, state

def handle_error(state: AgentState) -> Union[Tuple[str, AgentState], AgentState]:
    """Handle errors that occur during tool execution."""
    # Increment the error fix attempts counter
    state["error_fix_attempts"] += 1
    
    # If we've tried to fix the error too many times, generate the final output
    if state["error_fix_attempts"] > 3:
        error_message = "I've made several attempts to resolve the issues but ran into persistent errors. Let me provide you with the current results and some suggestions for how to proceed."
        state = add_message_to_history(state, "assistant", error_message)
        return EdgeNames.GENERATE_OUTPUT, state
    
    llm = get_langchain_openai_client()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=ERROR_HANDLING_PROMPT.format(
            current_tool=state["current_tool"],
            tool_input=state["tool_input"],
            tool_output=state["tool_output"],
            plan=state["plan"],
            chat_history=state["chat_history"],
            errors=state["errors"],
            error_fix_attempts=state["error_fix_attempts"]
        ))
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"chat_history": state["chat_history"]})
    
    # Extract the JSON from the response
    response_text = response.content
    
    try:
        # Extract JSON object if it's wrapped in ```json ... ```
        if "```json" in response_text and "```" in response_text.split("```json", 1)[1]:
            json_str = response_text.split("```json", 1)[1].split("```", 1)[0]
            error_solution = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            error_solution = json.loads(response_text)
        
        # Update the state based on the solution
        state["current_tool"] = error_solution["updated_tool"]
        state["tool_input"] = error_solution["updated_tool_input"]
        
        # Add the analysis and solution to chat history
        message = f"I encountered an error. Here's how I'll fix it:\n\n"
        message += f"Error analysis: {error_solution.get('error_analysis', '')}\n\n"
        message += f"Solution: {error_solution.get('solution', '')}\n\n"
        message += f"I'll retry with the {state['current_tool']} tool using updated parameters."
        
        state = add_message_to_history(state, "assistant", message)
        
        # Retry executing the tool
        return EdgeNames.EXECUTE_TOOL, state
        
    except (json.JSONDecodeError, KeyError) as e:
        # If there's an error parsing the JSON, add it to the errors list
        state["errors"].append({
            "type": "json_parse_error",
            "message": str(e),
            "response": response_text
        })
        
        # If we can't parse the response, generate the final output
        return EdgeNames.GENERATE_OUTPUT, state

def generate_final_output(state: AgentState) -> AgentState:
    """Generate the final output to present to the user."""
    llm = get_langchain_openai_client()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=FINAL_OUTPUT_PROMPT.format(
            input=state["input"],
            plan=state["plan"],
            chat_history=state["chat_history"]
        ))
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({"chat_history": state["chat_history"]})
    
    # Extract the final output from the response
    final_output = response.content
    
    # Update the state
    state["final_output"] = final_output
    state = add_message_to_history(state, "assistant", final_output)
    
    return state

def start(state: AgentState) -> AgentState:
    """Starting node that adds the user's request to chat history."""
    state = add_message_to_history(state, "user", state["input"])
    return state

def get_agent_executor() -> Dict[str, Callable]:
    """Get all the functions needed for the agent executor."""
    return {
        NodeNames.START: start,
        NodeNames.PLAN: plan,
        NodeNames.CHOOSE_TOOL: choose_tool,
        NodeNames.EXECUTE_TOOL: execute_tool,
        NodeNames.PROCESS_TOOL_OUTPUT: process_tool_output,
        NodeNames.HANDLE_ERROR: handle_error,
        NodeNames.GENERATE_FINAL_OUTPUT: generate_final_output,
    }