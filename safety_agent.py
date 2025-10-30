from dotenv import load_dotenv
import os
import pandas as pd
from typing import Literal, Optional
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
from io import StringIO

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

def clean_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up column headers by removing BOM, extra spaces, and normalizing names."""
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean column names
    new_columns = []
    for col in df_clean.columns:
        # Remove BOM (Byte Order Mark) if present
        clean_col = col.replace('ï»¿', '')
        
        # Remove leading/trailing whitespace
        clean_col = clean_col.strip()
        
        # Optional: You can add more cleaning rules here
        # For example, normalize spaces or remove special characters
        
        new_columns.append(clean_col)
    
    df_clean.columns = new_columns
    
    print(f"Column headers cleaned. First 5: {list(df_clean.columns[:5])}")
    return df_clean

def load_data_from_blob() -> pd.DataFrame:
    """Load safety data from Azure Blob Storage."""
    try:
        # Get blob credentials from environment
        blob_sas_url = os.getenv('blob_SAS_url')
        
        if not blob_sas_url:
            raise ValueError("blob_SAS_url not found in environment variables")
        
        print(f"Attempting to connect to blob storage...")
        
        # Specific file URL - encode the filename properly
        import requests
        from urllib.parse import quote
        
        # Extract base URL and SAS token
        base_url = blob_sas_url.split('?')[0]
        sas_token = blob_sas_url.split('?')[1] if '?' in blob_sas_url else ''
        
        # The exact file name with proper URL encoding
        filename = os.getenv('blob_file_name')
        encoded_filename = quote(filename)
        
        # Construct the full URL
        file_url = f"{base_url}/{encoded_filename}?{sas_token}"
        
        print(f"Attempting to load: {filename}")
        print(f"Full URL: {file_url[:80]}...")  # Show first 80 chars for debugging
        
        try:
            response = requests.get(file_url, timeout=60)
            
            if response.status_code == 200:
                print(f"Successfully loaded file: {filename}")
                
                # Convert to DataFrame
                csv_string = response.text
                df = pd.read_csv(StringIO(csv_string))
                
                # Clean up column headers
                df = clean_column_headers(df)
                
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
            
            # Clean up column headers for local file too
            df = clean_column_headers(df)
            
            print(f"Fallback: loaded {len(df)} rows from local file")
            return df
        except Exception as local_error:
            raise Exception(f"Both blob and local file loading failed. Blob error: {str(e)}, Local error: {str(local_error)}")

def search_vector_store(query: str) -> str:
    """Search the Azure AI Search vector store for semantic information."""
    # TODO: Implement Azure AI Search integration
    # For now, return a placeholder response
    return f"Semantic search results for: {query}\n[Vector store integration needed - placeholder response]"

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

# Main safety agent tools
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
        data_source = "Vector Store"
    
    # Step 3: Format response
    formatted_response = format_response(question, query_type, raw_result)
    
    return f"Query Type: {query_type}\nData Source: {data_source}\n\nResponse: {formatted_response}"

# Create the agent
agent = create_react_agent(
    model=model,
    tools=[safety_query_tool],
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