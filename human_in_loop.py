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

# Add memory/checkpointer and human-in-the-loop
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()

# Compile with interrupt_before tools - this will pause before executing tools
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

# Test the chatbot with human-in-the-loop
if __name__ == "__main__":
    # Configuration for memory - each thread maintains separate conversation history
    config = {"configurable": {"thread_id": "1"}}
    
    print("👤 Testing Human-in-the-Loop functionality...")
    print("The bot will pause before using tools and ask for approval!")
    
    # First interaction that should trigger a tool call
    print("\n" + "="*60)
    print("Step 1: Initial request (will pause before search)")
    print("="*60)
    
    user_input = "What's the weather like in San Francisco today?"
    
    # Stream until interruption
    events = list(graph.stream(
        {"messages": [("user", user_input)]}, 
        config=config,
        stream_mode="values"
    ))
    
    # Print the conversation so far
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    # Check the current state - should be interrupted before tools
    current_state = graph.get_state(config)
    print(f"\n🛑 INTERRUPTED! Current state: {current_state.next}")
    print("📋 The AI wants to use tools. Checking what it plans to do...")
    
    # Show the tool calls that are pending
    if current_state.values.get("messages"):
        last_message = current_state.values["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"🔍 Pending tool call: {last_message.tool_calls[0]['name']}")
            print(f"📝 Query: {last_message.tool_calls[0]['args']}")
    
    # Human approval step
    print("\n" + "="*60)
    print("Step 2: Human approval")
    print("="*60)
    
    approval = input("Do you approve this tool use? (y/n): ").lower().strip()
    
    if approval in ['y', 'yes']:
        print("✅ Approved! Continuing execution...")
        
        # Resume execution
        events = list(graph.stream(
            None,  # No new input, just resume
            config=config,
            stream_mode="values"
        ))
        
        # Print the results
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
                
    else:
        print("❌ Rejected! The tool call has been blocked.")
        print("You can manually update the state or provide alternative instructions.")
    
    print("\n" + "="*60)
    print("🔄 Interactive Human-in-the-Loop Chat")
    print("Tools will be paused for approval before execution!")
    print("Type 'quit' to exit")
    print("="*60)
    
    # Interactive loop with human-in-the-loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        # Stream the response
        events = list(graph.stream(
            {"messages": [("user", user_input)]}, 
            config=config,
            stream_mode="values"
        ))
        
        # Show the response (might be interrupted)
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
        
        # Check if we're interrupted (waiting for tool approval)
        current_state = graph.get_state(config)
        if current_state.next:
            print(f"\n🛑 INTERRUPTED at: {current_state.next}")
            
            # Show pending tool calls
            if current_state.values.get("messages"):
                last_message = current_state.values["messages"][-1]
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        print(f"🔍 Tool: {tool_call['name']}")
                        print(f"📝 Args: {tool_call['args']}")
            
            # Get human approval
            approval = input("Approve tool use? (y/n): ").lower().strip()
            
            if approval in ['y', 'yes']:
                print("✅ Continuing...")
                
                # Resume execution
                events = list(graph.stream(
                    None,  # Resume with no new input
                    config=config,
                    stream_mode="values"
                ))
                
                for event in events:
                    if "messages" in event:
                        event["messages"][-1].pretty_print()
            else:
                print("❌ Tool execution blocked!")