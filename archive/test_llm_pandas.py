from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "anthropic:claude-3-haiku-20240307",
    temperature=0
)

def test_statistical_analysis(query: str) -> str:
    """Test the statistical analysis with LLM-generated Pandas code."""
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
        print(f"Query: {query}")
        print("="*60)
        
        code_response = model.invoke(code_generation_prompt)
        generated_code = code_response.content.strip()
        
        print("Generated Code:")
        print(generated_code)
        print("="*60)
        
        # Extract code from markdown blocks if present
        if '```python' in generated_code:
            generated_code = generated_code.split('```python')[1].split('```')[0].strip()
        elif '```' in generated_code:
            generated_code = generated_code.split('```')[1].strip()
        
        print("Extracted Code:")
        print(generated_code)
        print("="*60)
        
        # Execute the generated code
        local_vars = {'df': df, 'pd': pd}
        exec(generated_code, globals(), local_vars)
        
        # Get the result
        if 'result' in local_vars:
            result = str(local_vars['result'])
            print("Execution Result:")
            print(result)
        else:
            print("No 'result' variable found in executed code")
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error in statistical analysis: {str(e)}"

if __name__ == "__main__":
    print("Testing LLM-Generated Pandas Code Analysis\n")
    
    # Test different types of queries
    queries = [
        "count the number of safety incidents by business unit",
        "show me the top 5 incident types",
        "what is the monthly trend of incidents?",
        "how many open vs complete incidents are there?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        test_statistical_analysis(query)
        print("\n")