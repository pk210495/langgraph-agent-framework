import sys
import io
from typing import Dict, Any, List, Tuple
import traceback
import contextlib
from langchain_core.tools import tool

class CodeExecutionError(Exception):
    """Exception raised for errors in code execution."""
    pass

@contextlib.contextmanager
def capture_stdout_stderr():
    """Context manager to capture stdout and stderr"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    sys.stdout, sys.stderr = stdout, stderr
    try:
        yield stdout, stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

@tool
def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code and return result
    
    Args:
        code: The Python code to execute
    
    Returns:
        Dictionary containing execution success flag, output, and error message if any
    """
    local_vars = {}
    
    with capture_stdout_stderr() as (stdout, stderr):
        try:
            exec(code, globals(), local_vars)
            output = stdout.getvalue()
            error = stderr.getvalue()
            success = not error or error.strip() == ""
        except Exception as e:
            output = stdout.getvalue()
            error = f"{str(e)}\n{traceback.format_exc()}"
            success = False
    
    return {
        "success": success,
        "output": output,
        "error": error if not success else "",
        "variables": {k: v for k, v in local_vars.items() 
                    if not k.startswith('_') and k != 'contextlib' and k != 'io' and k != 'sys'
                    and not callable(v) and not isinstance(v, type)}
    }

@tool
def code_interpreter(code: str, question: str) -> Dict[str, Any]:
    """
    Run Python code to answer a specific question
    
    Args:
        code: The Python code to execute
        question: The question to be answered using the code
    
    Returns:
        Dictionary containing execution result and the answer to the question
    """
    execution_result = execute_code(code)
    
    # Add interpreter context
    execution_result["question"] = question
    
    # If code execution was successful, add space for the answer that the agent will fill
    if execution_result["success"]:
        execution_result["answer"] = ""
    
    return execution_result