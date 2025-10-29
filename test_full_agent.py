from dotenv import load_dotenv
import os
import pandas as pd
from typing import Literal, Optional

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-3-haiku-20240307",
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

def search_vector_store(query: str) -> str:
    """Search the Azure AI Search vector store for semantic information."""
    return f"Semantic search results for: {query}\n[Vector store integration needed - placeholder response]"

def statistical_analysis(query: str) -> str:
    """Perform statistical analysis on CSV data using Pandas with LLM-generated code."""
    try:
        # Load actual CSV data
        df = pd.read_csv('sample_safety_data.csv', sep='\t')
        
        # Convert date columns to datetime with error handling
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
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

def safety_query_tool(question: str) -> str:
    """Main tool that handles both statistical and semantic safety queries."""
    
    print(f"Processing question: {question}")
    
    # Step 1: Classify intent
    query_type = classify_intent(question)
    print(f"Classified as: {query_type}")
    
    # Step 2: Route to appropriate handler
    if query_type == "statistical":
        raw_result = statistical_analysis(question)
        data_source = "CSV Statistical Data"
    else:
        raw_result = search_vector_store(question)
        data_source = "Vector Store"
    
    # Step 3: Return direct result (no additional LLM formatting to see raw output)
    return f"Query Type: {query_type}\nData Source: {data_source}\n\nResults:\n{raw_result}"

if __name__ == "__main__":
    print("=== ENHANCED SAFETY AGENT - LLM-GENERATED PANDAS TESTING ===\n")
    
    # Test with various statistical queries
    queries = [
        "Count incidents by business unit",
        "Show me the worst performing locations",
        "What are the seasonal trends in incidents?",
        "Compare electrical vs fall incidents",
        "How many incidents were resolved this year?",
        "What are the safety procedures for chemical handling?"  # semantic query
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        response = safety_query_tool(query)
        print(response)
        print("\n")