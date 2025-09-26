
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-haiku-20240307")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# Test the chatbot with tools
if __name__ == "__main__":
    print("🤖 Testing the chatbot with Tavily search tool...")
    
    # Test with a question that should trigger a search
    user_input = "What's the weather like in San Francisco today?"
    
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    print("\n" + "="*60)
    print("🔍 Interactive Chat with Search (type 'quit' to exit)")
    print("="*60)
    
    # Interactive chat loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
            
        events = graph.stream(
            {"messages": [("user", user_input)]}, 
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()