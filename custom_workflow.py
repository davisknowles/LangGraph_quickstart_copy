
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

# Add memory/checkpointer
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Test the chatbot with memory
if __name__ == "__main__":
    # Configuration for memory - each thread maintains separate conversation history
    config = {"configurable": {"thread_id": "1"}}
    
    print("� Testing the chatbot with memory and tools...")
    
    # First interaction
    print("\n" + "="*50)
    print("First interaction:")
    print("="*50)
    
    user_input = "Hi! I'm Bob. What's the weather like in San Francisco today?"
    
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        config=config,
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    # Inspect the state (show conversation history)
    print("\n" + "="*50)
    print("📊 Current conversation state:")
    print("="*50)
    current_state = graph.get_state(config)
    print(f"Messages in memory: {len(current_state.values['messages'])}")
    for i, msg in enumerate(current_state.values['messages']):
        print(f"{i+1}. {type(msg).__name__}: {msg.content[:100]}...")
    
    # Second interaction to test memory
    print("\n" + "="*50)
    print("Second interaction (testing memory):")
    print("="*50)
    
    user_input_2 = "Do you remember my name? And what did you tell me about the weather?"
    
    events = graph.stream(
        {"messages": [("user", user_input_2)]}, 
        config=config,
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    # Show updated state
    print("\n" + "="*50)
    print("📊 Updated conversation state:")
    print("="*50)
    updated_state = graph.get_state(config)
    print(f"Messages in memory: {len(updated_state.values['messages'])}")
    
    print("\n" + "="*60)
    print("🔍 Interactive Chat with Memory (type 'quit' to exit)")
    print("Each conversation remembers the full history!")
    print("="*60)
    
    # Interactive chat loop with memory
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            print(f"\nFinal conversation had {len(graph.get_state(config).values['messages'])} messages in memory.")
            break
            
        events = graph.stream(
            {"messages": [("user", user_input)]}, 
            config=config,
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()