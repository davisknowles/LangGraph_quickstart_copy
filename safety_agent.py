from dotenv import load_dotenv
import os
import pandas as pd
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from io import StringIO
import json

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

class SafetyResponse(BaseModel):
    response: str
    query_type: Literal["statistical", "semantic"]
    data_source: Optional[str] = None

checkpointer = InMemorySaver()

model = init_chat_model(
    "anthropic:claude-3-haiku-20240307",  # Using working model
    temperature=0
)

def classify_intent(question: str) -> str:
    """Classify if a question is statistical or semantic in nature."""
    classification_prompt = f"""
    Analyze this question and determine if it requires:
    - 'statistical': numerical analysis, calculations, data aggregation, trends, comparisons
    - 'semantic': conceptual understanding, explanations, definitions, procedures
    
    Question: {question}
    
    Respond with only 'statistical' or 'semantic'.
    """
    
    response = model.invoke(classification_prompt)
    return response.content.strip().lower()

def load_data_from_blob() -> pd.DataFrame:
    """Load safety data from Azure Blob Storage."""
    try:
        # Get blob credentials from environment
        blob_sas_url = os.getenv('blob_SAS_url')
        
        if not blob_sas_url:
            raise ValueError("blob_SAS_url not found in environment variables")
        
        print(f"Attempting to connect to blob storage...")
        
        # Use blob-level SAS URL directly (already contains full file path)
        import requests
        
        # Use blob-level SAS URL directly (already contains full file path)
        file_url = blob_sas_url
        
        filename = os.getenv('blob_file_name')
        print(f"Attempting to load: {filename}")
        print(f"Full URL: {file_url[:80]}...")  # Show first 80 chars for debugging
        
        try:
            response = requests.get(file_url, timeout=60)
            
            if response.status_code == 200:
                print(f"Successfully loaded file: {filename}")
                
                # Convert to DataFrame
                csv_string = response.text
                df = pd.read_csv(StringIO(csv_string))
                
                print(f"Successfully loaded {len(df)} rows from blob: {filename}")
                print(f"Columns: {df.shape[1]}")
                print(f"Sample columns: {list(df.columns[:5])}")
                
                return df
            else:
                print(f"HTTP Error {response.status_code}: {response.text[:200]}...")
                raise ValueError(f"Failed to download file: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as req_error:
            print(f"Request error: {str(req_error)}")
            raise ValueError(f"Request failed: {str(req_error)}")
        
    except Exception as e:
        print(f"Error loading from blob storage: {str(e)}")
        # Fallback to local file if blob fails
        try:
            df = pd.read_csv('sample_safety_data.csv', sep=',')
            if df.shape[1] == 1:
                df = pd.read_csv('sample_safety_data.csv', sep='\t')
            print(f"Fallback: loaded {len(df)} rows from local file")
            return df
        except Exception as local_error:
            raise Exception(f"Both blob and local file loading failed. Blob error: {str(e)}, Local error: {str(local_error)}")

def get_search_client() -> SearchClient:
    """Initialize Azure AI Search client."""
    search_service_name = os.getenv('AZURE_SEARCH_SERVICE')
    search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
    search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'safety-index')
    
    if not search_service_name or not search_api_key:
        raise ValueError("Azure Search credentials not found in environment variables")
    
    endpoint = f"https://{search_service_name}.search.windows.net"
    
    return SearchClient(
        endpoint=endpoint,
        index_name=search_index_name,
        credential=AzureKeyCredential(search_api_key)
    )

def search_vector_store(query: str) -> str:
    """Search the Azure AI Search vector store for semantic information."""
    try:
        search_client = get_search_client()
        
        # Search using your actual index structure
        try:
            search_results = search_client.search(
                search_text=query,
                top=5,
                query_type="simple"
            )
            
            # Format results using your actual field names
            results = []
            result_count = 0
            
            for result in search_results:
                result_count += 1
                
                # Get the search score
                score = getattr(result, '@search.score', 0)
                
                # Extract content from your actual fields
                incident_id = result.get('Row_ID', f'Incident {result_count}')
                description = result.get('Description_of_Safety_Concern', result.get('chunk', 'No description available'))
                incident_type = result.get('Type_of_Near_Miss', result.get('incident_category', 'Unknown'))
                business_unit = result.get('BusinessUnitCode', 'Unknown')
                location = result.get('Location', result.get('Detailed_Location', 'Unknown'))
                date_created = result.get('Created', result.get('Date_Identified', 'Unknown'))
                reported_by = result.get('Reported_By', 'Unknown')
                
                # Format the result with relevant safety information
                result_text = f"**Safety Incident {incident_id}** (Relevance: {score:.2f})\n"
                result_text += f"Type: {incident_type}\n"
                result_text += f"Business Unit: {business_unit}\n"
                result_text += f"Location: {location}\n"
                result_text += f"Date: {date_created}\n"
                result_text += f"Reported by: {reported_by}\n"
                
                # Truncate description if too long
                if isinstance(description, str) and len(description) > 200:
                    description = description[:200] + "..."
                
                result_text += f"Description: {description}\n"
                
                results.append(result_text)
            
            if results:
                formatted_results = "\n\n".join(results)
                return f"Vector Store Search Results for: '{query}'\n\n{formatted_results}"
            else:
                return f"Vector store is accessible but no results found for: '{query}'. You may need to index some documents first."
            
        except Exception as search_error:
            print(f"Search query failed: {str(search_error)}")
            return f"Vector store search failed: {str(search_error)}. This may indicate the index is empty or needs to be configured."
            
    except Exception as e:
        print(f"Error connecting to vector store: {str(e)}")
        # Check if it's a configuration issue
        if "credentials" in str(e).lower() or "authentication" in str(e).lower():
            return "Vector store connection failed: Please check your Azure AI Search credentials in the .env file."
        elif "index" in str(e).lower():
            return f"Vector store error: Index may not exist. Use vector_store_utils.py to create and populate your search index."
        else:
            return f"Vector store unavailable: {str(e)}. Please verify your Azure AI Search configuration."

def hybrid_search_vector_store(query: str) -> str:
    """Perform hybrid search combining both text and vector search."""
    try:
        search_client = get_search_client()
        
        # Hybrid search with both text and vector components
        search_results = search_client.search(
            search_text=query,
            vector_queries=[{
                "vector": None,  # This would be filled by your embedding model
                "k_nearest_neighbors": 5,
                "fields": "contentVector"  # Adjust based on your index schema
            }],
            select=["id", "content", "title", "category", "metadata"],
            top=3,
            query_type="semantic"
        )
        
        results = []
        for result in search_results:
            score = getattr(result, '@search.score', 0)
            result_text = f"**{result.get('title', 'Document')}** (Score: {score:.2f})\n"
            result_text += f"Content: {result.get('content', 'No content available')[:200]}...\n"
            results.append(result_text)
        
        if results:
            return f"Hybrid Search Results:\n\n" + "\n\n".join(results)
        else:
            return f"No results found for: '{query}'"
            
    except Exception as e:
        print(f"Hybrid search error: {str(e)}")
        # Fallback to regular semantic search
        return search_vector_store(query)

def statistical_analysis(query: str) -> str:
    """Perform statistical analysis on CSV data using Pandas with LLM-generated code."""
    try:
        # Load data from Azure Blob Storage
        df = load_data_from_blob()
        
        # Clean up NULL values
        df = df.replace('NULL', pd.NA)
        
        # Convert date columns to datetime with error handling
        if 'Created' in df.columns:
            df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        if 'Date Identified' in df.columns:
            df['Date Identified'] = pd.to_datetime(df['Date Identified'], errors='coerce')
        
        # Get column information for the LLM
        column_info = f"""
DataFrame columns and sample data:
{df.dtypes.to_string()}

Sample row:
{df.iloc[0].to_string()}

DataFrame shape: {df.shape}
        """
        
        # Generate Pandas code using LLM
        code_generation_prompt = f"""
You are a Python data analyst. Generate Pandas code to answer this question about safety incident data.

Question: {query}

{column_info}

Requirements:
1. The DataFrame is already loaded as 'df'
2. Generate ONLY the Python code needed to answer the question
3. Store the final result in a variable called 'result'
4. Use appropriate pandas methods like value_counts(), groupby(), describe(), etc.
5. Format the result as a string for display
6. Handle any potential errors gracefully
7. Include relevant context/totals where helpful

Example format:
```python
# Your analysis code here
result = "Your formatted result string"
```

Generate the code:
"""
        
        # Get the generated code from LLM
        code_response = model.invoke(code_generation_prompt)
        generated_code = code_response.content.strip()
        
        # Extract code from markdown blocks if present
        if '```python' in generated_code:
            generated_code = generated_code.split('```python')[1].split('```')[0].strip()
        elif '```' in generated_code:
            generated_code = generated_code.split('```')[1].strip()
        
        # Execute the generated code
        local_vars = {'df': df, 'pd': pd}
        exec(generated_code, globals(), local_vars)
        
        # Get the result
        if 'result' in local_vars:
            result = str(local_vars['result'])
        else:
            result = "Code executed successfully but no 'result' variable was created."
        
        # Add some metadata to the result
        result += f"\n\n[Analysis performed on {len(df)} safety incidents]"
        result += f"\n[Generated code: {generated_code[:100]}...]"
        
        return result
        
    except Exception as e:
        # Fallback to basic analysis if code generation fails
        try:
            basic_result = f"Statistical Analysis (Fallback):\n"
            basic_result += f"Total incidents: {len(df)}\n"
            basic_result += f"By Business Unit:\n{df['BusinessUnitCode'].value_counts().head().to_string()}\n"
            basic_result += f"By Incident Type:\n{df['Type of Near Miss'].value_counts().head().to_string()}"
            return basic_result
        except:
            return f"Error in statistical analysis: {str(e)}"

def format_response(question: str, query_type: str, raw_result: str) -> str:
    """Format the final response conversationally."""
    formatting_prompt = f"""
    Format this {query_type} query result into a natural, conversational response.
    
    Original question: {question}
    Query type: {query_type}
    Raw result: {raw_result}
    
    Provide a helpful, conversational response that directly answers the user's question.
    """
    
    response = model.invoke(formatting_prompt)
    return response.content

# Enhanced safety agent tools
def safety_query_tool(question: str) -> str:
    """Main tool that handles both statistical and semantic safety queries."""
    
    # Step 1: Classify intent
    query_type = classify_intent(question)
    
    # Step 2: Route to appropriate handler
    if query_type == "statistical":
        raw_result = statistical_analysis(question)
        data_source = "CSV Statistical Data"
    else:
        raw_result = search_vector_store(question)
        data_source = "Vector Store (Azure AI Search)"
    
    # Step 3: Format response
    formatted_response = format_response(question, query_type, raw_result)
    
    return f"Query Type: {query_type}\nData Source: {data_source}\n\nResponse: {formatted_response}"

def vector_search_tool(question: str) -> str:
    """Dedicated tool for vector store searches."""
    try:
        results = search_vector_store(question)
        return f"Vector Search Results:\n{results}"
    except Exception as e:
        return f"Vector search failed: {str(e)}"

def hybrid_search_tool(question: str) -> str:
    """Tool for hybrid search combining text and vector search."""
    try:
        results = hybrid_search_vector_store(question)
        return f"Hybrid Search Results:\n{results}"
    except Exception as e:
        return f"Hybrid search failed: {str(e)}"

def combined_analysis_tool(question: str) -> str:
    """Tool that combines both statistical data and vector store information."""
    try:
        # Get both statistical and semantic insights
        query_type = classify_intent(question)
        
        statistical_result = ""
        vector_result = ""
        
        # Always try to get vector store information for context
        try:
            vector_result = search_vector_store(question)
        except Exception as e:
            vector_result = f"Vector store unavailable: {str(e)}"
        
        # Get statistical analysis if relevant
        if query_type == "statistical" or "count" in question.lower() or "number" in question.lower():
            try:
                statistical_result = statistical_analysis(question)
            except Exception as e:
                statistical_result = f"Statistical analysis failed: {str(e)}"
        
        # Combine results
        combined_response = f"**Combined Analysis for: {question}**\n\n"
        
        if statistical_result:
            combined_response += f"**Statistical Analysis:**\n{statistical_result}\n\n"
        
        if vector_result and "Vector store unavailable" not in vector_result:
            combined_response += f"**Knowledge Base Context:**\n{vector_result}\n\n"
        
        # Use LLM to synthesize the information if we have meaningful content
        if statistical_result or (vector_result and "Vector store unavailable" not in vector_result):
            synthesis_prompt = f"""
            Based on the following information, provide a comprehensive answer to: {question}
            
            Statistical Data: {statistical_result if statistical_result else 'No statistical data available'}
            
            Knowledge Base Context: {vector_result if vector_result and 'Vector store unavailable' not in vector_result else 'No knowledge base context available'}
            
            Provide a well-structured, informative response that combines both quantitative and qualitative insights where available.
            """
            
            try:
                synthesis_response = model.invoke(synthesis_prompt)
                combined_response += f"**Synthesized Answer:**\n{synthesis_response.content}"
            except Exception as e:
                combined_response += f"**Note:** Unable to synthesize response: {str(e)}"
        else:
            combined_response += "**Note:** No data available from either statistical analysis or knowledge base."
        
        return combined_response
        
    except Exception as e:
        return f"Combined analysis failed: {str(e)}"

# Create the agent with enhanced tools
tools = [
    safety_query_tool,           # Main routing tool
    vector_search_tool,          # Direct vector search
    hybrid_search_tool,          # Hybrid search capability
    combined_analysis_tool       # Combines statistical + vector store
]

agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer
)

# Test the agent
# You need to provide a config with thread_id when using checkpointer
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "count the number of safety incidents by business unit"}]},
    config=config
)

print("Full response:", response)
print("*************")
print("Structured response:", response.get("structured_response"))