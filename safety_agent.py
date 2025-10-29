from dotenv import load_dotenv
import os
import pandas as pd
from typing import Literal, Optional
from pydantic import BaseModel

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

def search_vector_store(query: str) -> str:
    """Search the Azure AI Search vector store for semantic information."""
    # TODO: Implement Azure AI Search integration
    # For now, return a placeholder response
    return f"Semantic search results for: {query}\n[Vector store integration needed - placeholder response]"

def statistical_analysis(query: str) -> str:
    """Perform statistical analysis on CSV data using Pandas."""
    try:
        # Load actual CSV data
        df = pd.read_csv('sample_safety_data.csv', sep='\t')
        
        # Convert date columns to datetime with error handling
        df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        df['Date Identified'] = pd.to_datetime(df['Date Identified'], errors='coerce')
        
        # Basic statistical analysis based on query keywords with aggregations
        if 'business unit' in query.lower() or 'businessunit' in query.lower():
            result = f"Incidents by Business Unit Code:\n"
            bu_counts = df['BusinessUnitCode'].value_counts()
            result += bu_counts.to_string()
            result += f"\n\nTotal incidents: {len(df)} across {len(bu_counts)} business units"
            
        elif 'job' in query.lower() and 'location' in query.lower():
            result = f"Incidents by Job Name/Location:\n"
            job_counts = df['Job Name / Location'].value_counts().head(15)
            result += job_counts.to_string()
            result += f"\n\nShowing top 15 job locations out of {df['Job Name / Location'].nunique()} total"
            
        elif 'type' in query.lower() and ('near miss' in query.lower() or 'incident' in query.lower()):
            result = f"Incidents by Type of Near Miss:\n"
            type_counts = df['Type of Near Miss'].value_counts()
            result += type_counts.to_string()
            result += f"\n\nTotal incidents: {len(df)} across {len(type_counts)} incident types"
            
        elif 'aggregate' in query.lower() or 'summary' in query.lower():
            # Comprehensive aggregation
            result = f"=== SAFETY INCIDENT AGGREGATION SUMMARY ===\n\n"
            
            # Business Unit aggregation
            result += f"BY BUSINESS UNIT CODE:\n"
            bu_counts = df['BusinessUnitCode'].value_counts()
            result += bu_counts.to_string()
            
            # Type aggregation
            result += f"\n\nBY INCIDENT TYPE:\n"
            type_counts = df['Type of Near Miss'].value_counts()
            result += type_counts.to_string()
            
            # Job/Location aggregation (top 10)
            result += f"\n\nBY JOB NAME/LOCATION (Top 10):\n"
            job_counts = df['Job Name / Location'].value_counts().head(10)
            result += job_counts.to_string()
            
            # Combined aggregation
            result += f"\n\nCROSS-TABULATION (Business Unit vs Incident Type):\n"
            crosstab = pd.crosstab(df['BusinessUnitCode'], df['Type of Near Miss'])
            result += crosstab.to_string()
            
            result += f"\n\nOVERALL TOTALS:"
            result += f"\n- Total incidents: {len(df)}"
            result += f"\n- Business units: {df['BusinessUnitCode'].nunique()}"
            result += f"\n- Incident types: {df['Type of Near Miss'].nunique()}"
            result += f"\n- Unique job locations: {df['Job Name / Location'].nunique()}"
            
        elif 'count' in query.lower() or 'number' in query.lower():
            if 'department' in query.lower():
                result = f"Incidents by Business Unit Code:\n{df['BusinessUnitCode'].value_counts().to_string()}"
            else:
                result = f"Total safety incidents: {len(df)}\n"
                result += f"\nBy Business Unit:\n{df['BusinessUnitCode'].value_counts().to_string()}"
                result += f"\n\nBy Incident Type:\n{df['Type of Near Miss'].value_counts().to_string()}"
            
        elif 'trend' in query.lower() or 'time' in query.lower() or 'month' in query.lower():
            df['Month'] = df['Date Identified'].dt.to_period('M')
            
            # Monthly trends by business unit
            result = f"Monthly Incident Trends by Business Unit:\n"
            monthly_bu = df.groupby(['Month', 'BusinessUnitCode']).size().unstack(fill_value=0)
            result += monthly_bu.to_string()
            
            # Overall monthly trend
            result += f"\n\nOverall Monthly Trends:\n"
            monthly_total = df.groupby('Month').size()
            result += monthly_total.to_string()
            
        elif 'status' in query.lower():
            result = f"Incident Status by Business Unit:\n"
            status_bu = pd.crosstab(df['BusinessUnitCode'], df['Status'])
            result += status_bu.to_string()
            result += f"\n\nOverall Status Summary:\n{df['Status'].value_counts().to_string()}"
            
        elif 'open' in query.lower():
            open_incidents = df[df['Status'] == 'Open']
            result = f"Open Incidents by Business Unit:\n"
            result += open_incidents['BusinessUnitCode'].value_counts().to_string()
            result += f"\n\nOpen Incidents by Type:\n"
            result += open_incidents['Type of Near Miss'].value_counts().to_string()
            result += f"\n\nTotal Open: {len(open_incidents)} out of {len(df)}"
            
        else:
            # Default aggregated summary
            result = f"Safety Incident Analysis Summary:\n"
            result += f"Total incidents: {len(df)}\n\n"
            
            result += f"By Business Unit Code:\n"
            result += df['BusinessUnitCode'].value_counts().to_string()
            
            result += f"\n\nBy Incident Type:\n"
            result += df['Type of Near Miss'].value_counts().to_string()
            
            result += f"\n\nTop Job Locations (Top 8):\n"
            result += df['Job Name / Location'].value_counts().head(8).to_string()
            
        # Ensure we always return something
        if not result or result.strip() == "":
            result = f"Basic analysis: Found {len(df)} safety incidents in the dataset."
            
        return result
        
    except Exception as e:
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
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "safety_session_1"}}
    
    # Test with aggregated statistical queries
    print("=== Testing Business Unit Aggregation ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Show me incidents by business unit code"}]},
        config=config
    )
    print(response["messages"][-1].content)
    
    print("\n=== Testing Comprehensive Aggregation ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Give me a complete aggregation summary of safety incidents"}]},
        config=config
    )
    print(response["messages"][-1].content)
    
    print("\n=== Testing Semantic Query ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the safety procedures for chemical handling?"}]},
        config=config
    )
    print(response["messages"][-1].content)