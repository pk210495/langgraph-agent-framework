import os
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, 
    SearchField, 
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField
)
from azure.core.credentials import AzureKeyCredential
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Azure Search configuration
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

def get_search_index_client():
    """Get Azure Search Index client"""
    if not all([AZURE_SEARCH_SERVICE, AZURE_SEARCH_KEY]):
        raise ValueError("Azure Search credentials not properly configured")
    
    return SearchIndexClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/",
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

def get_search_client(index_name):
    """Get Azure Search client for a specific index"""
    if not all([AZURE_SEARCH_SERVICE, AZURE_SEARCH_KEY, index_name]):
        raise ValueError("Azure Search credentials or index name not properly configured")
    
    return SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/",
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

def get_field_type(field_type: str) -> SearchFieldDataType:
    """Convert string field type to Azure Search field type"""
    type_mapping = {
        "string": SearchFieldDataType.STRING,
        "int": SearchFieldDataType.INT32,
        "integer": SearchFieldDataType.INT32,
        "long": SearchFieldDataType.INT64,
        "double": SearchFieldDataType.DOUBLE,
        "boolean": SearchFieldDataType.BOOLEAN,
        "date": SearchFieldDataType.DATE_TIME_OFFSET,
        "datetime": SearchFieldDataType.DATE_TIME_OFFSET,
        "point": SearchFieldDataType.GEOGRAPHY_POINT,
        "collection": SearchFieldDataType.COLLECTION(SearchFieldDataType.STRING),
        "complex": None  # Complex fields need special handling
    }
    
    return type_mapping.get(field_type.lower(), SearchFieldDataType.STRING)

@tool
def create_search_index(index_name: str, fields: List[str], field_types: List[str]) -> Dict[str, Any]:
    """
    Create an Azure Cognitive Search index with specified fields and types
    
    Args:
        index_name: Name of the index to create
        fields: List of field names
        field_types: List of field types (string, int, double, boolean, date, point, collection)
    
    Returns:
        Dictionary containing success flag and error message if any
    """
    try:
        if len(fields) != len(field_types):
            return {
                "success": False,
                "error": "The number of fields must match the number of field types."
            }
        
        client = get_search_index_client()
        
        # Define fields for the index
        search_fields = []
        
        for i, (field, field_type) in enumerate(zip(fields, field_types)):
            # Key field (first field is always the key)
            if i == 0:
                search_fields.append(
                    SimpleField(name=field, type=get_field_type(field_type), key=True)
                )
            # Searchable text fields
            elif field_type.lower() in ["string", "text"]:
                search_fields.append(
                    SearchableField(name=field, type=SearchFieldDataType.STRING)
                )
            # Other field types
            else:
                search_fields.append(
                    SimpleField(name=field, type=get_field_type(field_type))
                )
        
        # Create the index
        index = SearchIndex(name=index_name, fields=search_fields)
        result = client.create_or_update_index(index)
        
        return {
            "success": True,
            "message": f"Index '{index_name}' created successfully.",
            "index_name": result.name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def upload_to_search_index(index_name: str, data_source: Union[str, List[Dict[str, Any]]], field_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Upload data to an Azure Cognitive Search index
    
    Args:
        index_name: Name of the index to upload to
        data_source: Either a file path to a CSV/Excel/JSON file or a list of dictionaries
        field_mappings: Optional mapping between source fields and index fields
    
    Returns:
        Dictionary containing success flag, number of documents uploaded, and error message if any
    """
    try:
        # Load data
        if isinstance(data_source, str):
            # Assuming it's a file path
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_source)
            elif data_source.endswith('.json'):
                df = pd.read_json(data_source)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {data_source}"
                }
            
            # Convert DataFrame to list of dictionaries
            data = df.to_dict(orient='records')
        else:
            # Assuming it's already a list of dictionaries
            data = data_source
        
        # Apply field mappings if provided
        if field_mappings:
            mapped_data = []
            for item in data:
                mapped_item = {}
                for source_field, index_field in field_mappings.items():
                    if source_field in item:
                        mapped_item[index_field] = item[source_field]
                mapped_data.append(mapped_item)
            data = mapped_data
        
        # Get search client
        search_client = get_search_client(index_name)
        
        # Upload data in batches
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            result = search_client.upload_documents(documents=batch)
            
            # Check if any document failed to upload
            if any(not succeeded for succeeded in [doc.succeeded for doc in result]):
                failed_docs = [
                    (doc.key, doc.error_message)
                    for doc in result
                    if not doc.succeeded
                ]
                return {
                    "success": False,
                    "error": f"Some documents failed to upload: {failed_docs}",
                    "total_documents": len(data),
                    "uploaded_documents": i + sum(1 for doc in result if doc.succeeded)
                }
        
        return {
            "success": True,
            "message": f"Successfully uploaded {len(data)} documents to index '{index_name}'.",
            "total_documents": len(data)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def search_index(index_name: str, query: str, top: int = 10) -> Dict[str, Any]:
    """
    Search an Azure Cognitive Search index
    
    Args:
        index_name: Name of the index to search
        query: Search query text
        top: Maximum number of results to return (default: 10)
    
    Returns:
        Dictionary containing search results and metadata
    """
    try:
        search_client = get_search_client(index_name)
        
        results = search_client.search(
            search_text=query,
            top=top,
            include_total_count=True
        )
        
        # Convert results to list of dictionaries
        documents = [doc for doc in results]
        
        return {
            "success": True,
            "count": len(documents),
            "total_count": results.get_count(),
            "results": documents
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }