import os
import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import agent components
from agent.graph import build_runnable_agent
from agent.state import create_initial_state, pretty_print_state

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="LangGraph Agent Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
st.sidebar.title("🤖 LangGraph Agent")
st.sidebar.markdown("An agentic framework with multiple capabilities")

# Azure OpenAI configuration
with st.sidebar.expander("Azure OpenAI Configuration", expanded=False):
    azure_endpoint = st.text_input(
        "Azure OpenAI Endpoint", 
        value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        type="password"
    )
    
    azure_api_key = st.text_input(
        "Azure OpenAI API Key", 
        value=os.getenv("AZURE_OPENAI_API_KEY", ""),
        type="password"
    )
    
    azure_api_version = st.text_input(
        "Azure OpenAI API Version", 
        value=os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
    )
    
    azure_model = st.text_input(
        "Azure OpenAI Model Deployment", 
        value=os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
    )

# Azure Search configuration
with st.sidebar.expander("Azure Search Configuration", expanded=False):
    azure_search_service = st.text_input(
        "Azure Search Service", 
        value=os.getenv("AZURE_SEARCH_SERVICE", "")
    )
    
    azure_search_key = st.text_input(
        "Azure Search Key", 
        value=os.getenv("AZURE_SEARCH_KEY", ""),
        type="password"
    )
    
    azure_search_index = st.text_input(
        "Default Azure Search Index", 
        value=os.getenv("AZURE_SEARCH_INDEX", "")
    )

# Update environment variables
if st.sidebar.button("Save Configuration"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
    os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version
    os.environ["AZURE_OPENAI_MODEL"] = azure_model
    os.environ["AZURE_SEARCH_SERVICE"] = azure_search_service
    os.environ["AZURE_SEARCH_KEY"] = azure_search_key
    os.environ["AZURE_SEARCH_INDEX"] = azure_search_index
    st.sidebar.success("Configuration saved!")

# Add debug toggle
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# File upload widget
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "xlsx", "txt", "json"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"File saved to {file_path}")
    
    # Display file info
    file_info = {
        "Name": uploaded_file.name,
        "Size": f"{uploaded_file.size / 1024:.2f} KB",
        "Path": file_path
    }
    
    # Try to display preview for certain file types
    if uploaded_file.name.endswith((".csv", ".xlsx")):
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.sidebar.markdown("#### File Preview")
            st.sidebar.dataframe(df.head(5), use_container_width=True)
            
            file_info["Columns"] = ", ".join(df.columns.tolist())
            file_info["Rows"] = df.shape[0]
        except Exception as e:
            st.sidebar.error(f"Error previewing file: {str(e)}")
    
    st.sidebar.markdown("#### File Info")
    for key, value in file_info.items():
        st.sidebar.text(f"{key}: {value}")

# Main chat interface
st.title("🤖 LangGraph Agentic Framework")
st.markdown("""
This is an agentic framework built with LangGraph and Streamlit. It can:
- Execute Python code
- Read and write files
- Create Azure Cognitive Search indexes
- Upload data to Azure Search
- Adaptively fix errors when they occur
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask the agent to do something...")

# When the user submits a message
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display agent response with a spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Create debug containers if debug mode is enabled
        if debug_mode:
            debug_expander = st.expander("Debug Information", expanded=False)
        
        # Create initial agent state
        state = create_initial_state()
        state["input"] = prompt
        state["debug"] = debug_mode
        
        # Build and run the agent
        agent = build_runnable_agent()
        
        # Streamlit doesn't support real-time updates well, so we'll collect the intermediate states
        intermediate_states = []
        
        def collect_intermediate_states(state):
            intermediate_states.append(state.copy())
            return state
            
        # Create a progress indicator
        progress = st.progress(0)
        
        # Run the agent with event streaming
        for i, event in enumerate(agent.stream(state, {"recursion_limit": 20})):
            # Update progress based on event count
            progress.progress(min(i / 10, 1.0))
            
            # Store the event for debugging
            if isinstance(event, dict) and "state" in event:
                collect_intermediate_states(event["state"])
        
            # Get the latest message from chat history
            if isinstance(event, dict) and "state" in event and "chat_history" in event["state"]:
                chat_history = event["state"]["chat_history"]
                if chat_history and len(chat_history) > 0:
                    latest_message = chat_history[-1]
                    if latest_message["role"] == "assistant":
                        message_placeholder.markdown(latest_message["content"])
        
        # Set progress to 100% when done
        progress.progress(1.0)
        
        # Display the final result
        final_state = intermediate_states[-1] if intermediate_states else None
        
        if final_state and "chat_history" in final_state and final_state["chat_history"]:
            final_messages = [msg for msg in final_state["chat_history"] if msg["role"] == "assistant"]
            if final_messages:
                final_message = final_messages[-1]
                message_placeholder.markdown(final_message["content"])
                # Add the assistant message to chat history
                if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
                    st.session_state.messages.append(final_message)
        
        # Display debug information if enabled
        if debug_mode and final_state:
            with debug_expander:
                st.subheader("Agent Execution Details")
                
                # Show the plan
                if final_state.get("plan"):
                    st.markdown("#### Plan")
                    for i, step in enumerate(final_state["plan"]):
                        st.markdown(f"{i+1}. {step}")
                
                # Show tools used
                if "chat_history" in final_state:
                    tool_executions = []
                    for msg in final_state["chat_history"]:
                        if msg["role"] == "assistant" and "I'll use the" in msg["content"]:
                            tool_name = msg["content"].split("I'll use the")[1].split()[0] if "I'll use the" in msg["content"] else "Unknown"
                            tool_executions.append(tool_name)
                    
                    if tool_executions:
                        st.markdown("#### Tools Used")
                        st.write(", ".join(tool_executions))
                
                # Show errors if any
                if final_state.get("errors"):
                    st.markdown("#### Errors")
                    for i, error in enumerate(final_state["errors"]):
                        st.markdown(f"**Error {i+1}:** {error.get('type')}")
                        st.code(error.get('message', ''))
                
                # Show full state as JSON
                st.markdown("#### Full State")
                
                # Clean up the state for display (remove large content)
                display_state = final_state.copy()
                if "chat_history" in display_state:
                    display_state["chat_history"] = f"[{len(display_state['chat_history'])} messages]"
                
                st.json(display_state)
                
                # Allow downloading the full trace
                trace_data = json.dumps(intermediate_states, default=str)
                st.download_button(
                    label="Download Full Execution Trace",
                    data=trace_data,
                    file_name="agent_execution_trace.json",
                    mime="application/json"
                )

# Add some examples at the bottom
with st.expander("Example Prompts", expanded=False):
    examples = [
        "Read the file I uploaded and show me the first 10 rows",
        "Create a Cognitive Search index called 'products' with fields 'id' (string), 'name' (string), 'price' (double), and 'category' (string)",
        "Write a Python function to process my CSV file, calculate the average of each numeric column, and save the results to a new file",
        "Upload the data in my CSV file to the Azure Search index 'products'",
        "Write me a script that reads in a CSV file, cleans the data by removing duplicates and handling missing values, and outputs a new CSV"
    ]
    
    for example in examples:
        if st.button(example):
            # Add to chat history and submit as if the user typed it
            st.session_state.messages.append({"role": "user", "content": example})
            st.experimental_rerun()

# Show a footer
st.markdown("---")
st.markdown("LangGraph Agent Framework | Made with Streamlit, LangGraph, and Azure OpenAI")