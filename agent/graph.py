from typing import Dict, List, Any, Optional, Tuple, Union, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState, NodeNames, EdgeNames, create_initial_state
from .agent import get_agent_executor

def create_agent_graph() -> StateGraph:
    """Create the LangGraph workflow for the agent."""
    # Create a new graph
    graph = StateGraph(AgentState)
    
    # Get all the node functions
    agent_executor = get_agent_executor()
    
    # Add nodes to the graph
    graph.add_node(NodeNames.START, agent_executor[NodeNames.START])
    graph.add_node(NodeNames.PLAN, agent_executor[NodeNames.PLAN])
    graph.add_node(NodeNames.CHOOSE_TOOL, agent_executor[NodeNames.CHOOSE_TOOL])
    graph.add_node(NodeNames.EXECUTE_TOOL, agent_executor[NodeNames.EXECUTE_TOOL])
    graph.add_node(NodeNames.PROCESS_TOOL_OUTPUT, agent_executor[NodeNames.PROCESS_TOOL_OUTPUT])
    graph.add_node(NodeNames.HANDLE_ERROR, agent_executor[NodeNames.HANDLE_ERROR])
    graph.add_node(NodeNames.GENERATE_FINAL_OUTPUT, agent_executor[NodeNames.GENERATE_FINAL_OUTPUT])
    
    # Define edges between nodes
    graph.add_edge(NodeNames.START, NodeNames.PLAN)
    graph.add_edge(NodeNames.PLAN, NodeNames.CHOOSE_TOOL)
    graph.add_edge(NodeNames.CHOOSE_TOOL, NodeNames.EXECUTE_TOOL)
    graph.add_edge(NodeNames.EXECUTE_TOOL, NodeNames.PROCESS_TOOL_OUTPUT)
    
    # Add conditional edges
    graph.add_conditional_edges(
        NodeNames.PROCESS_TOOL_OUTPUT,
        lambda state: state[0] if isinstance(state, tuple) else EdgeNames.CHOOSE_TOOL,
        {
            EdgeNames.CHOOSE_TOOL: NodeNames.CHOOSE_TOOL,
            EdgeNames.ERROR: NodeNames.HANDLE_ERROR,
            EdgeNames.GENERATE_OUTPUT: NodeNames.GENERATE_FINAL_OUTPUT,
        }
    )
    
    graph.add_conditional_edges(
        NodeNames.HANDLE_ERROR,
        lambda state: state[0] if isinstance(state, tuple) else EdgeNames.EXECUTE_TOOL,
        {
            EdgeNames.EXECUTE_TOOL: NodeNames.EXECUTE_TOOL,
            EdgeNames.GENERATE_OUTPUT: NodeNames.GENERATE_FINAL_OUTPUT,
        }
    )
    
    # Set the final node
    graph.add_edge(NodeNames.GENERATE_FINAL_OUTPUT, END)
    
    # Set the entry point
    graph.set_entry_point(NodeNames.START)
    
    return graph

def build_runnable_agent():
    """Build a runnable agent from the graph."""
    # Create the graph
    graph = create_agent_graph()
    
    # Compile the graph
    return graph.compile()