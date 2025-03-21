import os
import json
import csv
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import tool

@tool
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Read a file and return its contents
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        Dictionary containing success flag, content, and error message if any
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "content": "",
                "error": f"File not found: {file_path}"
            }
        
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() in ['.csv']:
            df = pd.read_csv(file_path)
            return {
                "success": True,
                "content": df.to_dict(orient='records'),
                "sample": df.head(5).to_dict(orient='records'),
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "file_type": "csv"
            }
        
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            return {
                "success": True,
                "content": df.to_dict(orient='records'),
                "sample": df.head(5).to_dict(orient='records'),
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "file_type": "excel"
            }
            
        elif file_extension.lower() in ['.json']:
            with open(file_path, 'r') as f:
                content = json.load(f)
            return {
                "success": True,
                "content": content,
                "file_type": "json"
            }
            
        else:  # Default to text file
            with open(file_path, 'r') as f:
                content = f.read()
            return {
                "success": True,
                "content": content,
                "file_type": "text"
            }
            
    except Exception as e:
        return {
            "success": False,
            "content": "",
            "error": str(e)
        }

@tool
def write_file(file_path: str, content: Union[str, List, Dict]) -> Dict[str, Any]:
    """
    Write content to a file
    
    Args:
        file_path: Path where the file should be written
        content: Content to write to the file (string, list, or dictionary)
    
    Returns:
        Dictionary containing success flag and error message if any
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() in ['.csv']:
            # Convert to DataFrame if not already
            if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                df = pd.DataFrame(content)
                df.to_csv(file_path, index=False)
            elif isinstance(content, pd.DataFrame):
                content.to_csv(file_path, index=False)
            else:
                return {
                    "success": False,
                    "error": "Content must be a list of dictionaries or a pandas DataFrame for CSV files."
                }
        
        elif file_extension.lower() in ['.xlsx', '.xls']:
            # Convert to DataFrame if not already
            if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                df = pd.DataFrame(content)
                df.to_excel(file_path, index=False)
            elif isinstance(content, pd.DataFrame):
                content.to_excel(file_path, index=False)
            else:
                return {
                    "success": False,
                    "error": "Content must be a list of dictionaries or a pandas DataFrame for Excel files."
                }
                
        elif file_extension.lower() in ['.json']:
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
                
        else:  # Default to text file
            with open(file_path, 'w') as f:
                f.write(str(content))
                
        return {
            "success": True,
            "message": f"File successfully written to {file_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def get_dataframe(file_path: str) -> Dict[str, Any]:
    """
    Read a file into a pandas DataFrame
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        Dictionary containing success flag, DataFrame information, and error message if any
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() in ['.csv']:
            df = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension.lower() in ['.json']:
            df = pd.read_json(file_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_extension}"
            }
        
        # Convert DataFrame to a variable that can be used in code execution
        # This is a placeholder - the actual implementation will depend on how 
        # this interacts with the code execution tool
        df_info = {
            "success": True,
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "head": df.head(5).to_dict(orient='records'),
            "dataframe_code": f"df = pd.read_{'csv' if file_extension.lower() == '.csv' else 'excel' if file_extension.lower() in ['.xlsx', '.xls'] else 'json'}('{file_path}')"
        }
        
        return df_info
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }