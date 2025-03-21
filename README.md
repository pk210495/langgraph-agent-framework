# LangGraph Agentic Framework with Streamlit UI

This project implements an agentic framework using LangGraph with a Streamlit frontend UI. The agent has multiple capabilities including code execution, file operations, and Azure Cognitive Search integration.

## Features

- **Code Execution**: Execute Python code and see the results
- **File Operations**: Read and write files (CSV, Excel, JSON, text)
- **Azure Cognitive Search**: Create indexes and upload data
- **Adaptive Error Handling**: The agent can fix errors and adapt its approach
- **Streamlit UI**: User-friendly interface for interacting with the agent

## Project Structure

```
langgraph_agent/
├── app.py                     # Streamlit application entry point
├── requirements.txt           # Dependencies
├── agent/
│   ├── __init__.py
│   ├── agent.py               # Main LangGraph agent definition
│   ├── graph.py               # LangGraph workflow definition
│   └── state.py               # State management
├── tools/
│   ├── __init__.py
│   ├── code_execution.py      # Code execution & interpreter tools
│   ├── file_operations.py     # File I/O tools
│   └── azure_tools.py         # Azure Cognitive Search tools
└── utils/
    ├── __init__.py
    └── openai_client.py       # Azure OpenAI client setup
```

## Prerequisites

- Python 3.9 or higher
- Azure OpenAI API access
- Azure Cognitive Search service (for search-related functionality)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/pk210495/langgraph-agent-framework.git
cd langgraph-agent-framework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Azure credentials:
```
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-07-01-preview
AZURE_OPENAI_MODEL=your-deployment-name

# Azure Cognitive Search
AZURE_SEARCH_SERVICE=your-search-service-name
AZURE_SEARCH_KEY=your-search-key
AZURE_SEARCH_INDEX=default-index-name
```

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

1. Configure your Azure credentials in the sidebar (if not set via environment variables)
2. Upload files if needed for processing
3. Enter prompts in the chat interface to instruct the agent
4. Toggle debug mode to see detailed agent execution information

## Example Prompts

- "Read the CSV file I uploaded and calculate summary statistics"
- "Create a search index named 'products' with fields 'id', 'name', 'price', and 'description'"
- "Write a Python function to clean the data in my CSV"
- "Upload the data from my Excel file to the Azure Search index"

## Agent Workflow

1. The agent starts by creating a plan based on the user's request
2. It selects and executes appropriate tools based on the plan
3. It processes the results and decides the next steps
4. If an error occurs, it attempts to fix it and retry
5. Finally, it generates a comprehensive response for the user

## Extending the Framework

To add new capabilities:

1. Create a new tool function in the appropriate file or create a new file in the `tools` directory
2. Add the tool to the `TOOLS` dictionary in `agent.py`
3. Update the tool selection prompt to include your new tool

## Troubleshooting

- If you encounter issues with Azure services, verify your credentials in the sidebar
- Enable debug mode to see detailed execution information
- Check the Streamlit logs for any errors

## License

MIT