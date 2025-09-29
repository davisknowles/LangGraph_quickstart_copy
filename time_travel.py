from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model
# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-haiku-20240307")

from typing import Annotated
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the graph
graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile with memory checkpointer for time travel
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

print("="*60)
print("LANGGRAPH TIME TRAVEL DEMONSTRATION")
print("="*60)

# Configuration for conversation thread
config = {"configurable": {"thread_id": "time_travel_demo"}}

print("\n1. FIRST CONVERSATION - Creating checkpoints...")
print("-" * 40)

# First conversation
events = graph.stream(
    {"messages": [{"role": "user", "content": "What is LangGraph?"}]},
    config,
    stream_mode="values",
)

message_count = 0
for event in events:
    if "messages" in event:
        message_count += 1
        print(f"Message {message_count}: {event['messages'][-1].type} message")

print(f"\nFirst conversation completed. Total messages: {message_count}")

print("\n2. SECOND CONVERSATION - Adding more checkpoints...")
print("-" * 40)

# Second conversation
events = graph.stream(
    {"messages": [{"role": "user", "content": "How can I build agents with it?"}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        message_count += 1
        print(f"Message {message_count}: {event['messages'][-1].type} message")

print(f"\nBoth conversations completed. Total messages: {message_count}")

print("\n3. TIME TRAVEL - Exploring state history...")
print("-" * 40)

# Explore state history
checkpoint_count = 0
checkpoints_info = []

for state in graph.get_state_history(config):
    checkpoint_count += 1
    info = {
        'checkpoint': checkpoint_count,
        'messages': len(state.values["messages"]), 
        'next': state.next,
        'config': state.config
    }
    checkpoints_info.append(info)
    print(f"Checkpoint {checkpoint_count}: {info['messages']} messages, Next: {info['next']}")

print(f"\nFound {checkpoint_count} checkpoints in the conversation history!")

print("\n4. TIME TRAVEL - Resuming from a previous checkpoint...")
print("-" * 40)

# Find a checkpoint from the middle of the conversation
target_checkpoint = None
for info in checkpoints_info:
    if info['messages'] == 2:  # After first exchange
        target_checkpoint = info
        break

if target_checkpoint:
    print(f"Time traveling to checkpoint with {target_checkpoint['messages']} messages...")
    print(f"Next node to execute: {target_checkpoint['next']}")
    print(f"Checkpoint ID: {target_checkpoint['config']['configurable']['checkpoint_id']}")
    
    print("\nResuming execution from this past state:")
    print("(This demonstrates 'rewinding' the graph to a previous point)")
    
    # Resume from the checkpoint - this is the time travel!
    resumed_events = graph.stream(None, target_checkpoint['config'], stream_mode="values")
    
    resumed_count = 0
    for event in resumed_events:
        if "messages" in event:
            resumed_count += 1
            last_message = event["messages"][-1]
            print(f"Resumed message {resumed_count}: {last_message.type}")
            if hasattr(last_message, 'content'):
                content_preview = str(last_message.content)[:100]
                print(f"  Content preview: {content_preview}...")
    
    print(f"\nTime travel complete! Resumed {resumed_count} messages from the past.")
else:
    print("No suitable checkpoint found for time travel demonstration.")

print("\n" + "="*60)
print("TIME TRAVEL CONCEPTS DEMONSTRATED:")
print("="*60)
print("✓ Automatic checkpointing of every graph state")
print("✓ get_state_history() to browse all past states")
print("✓ Resuming execution from any checkpoint using config")
print("✓ 'Time travel' - rewinding and exploring different paths")
print("✓ Useful for debugging, experimentation, and interactive apps")
print("\nTime travel in LangGraph enables powerful features like:")
print("- Undoing actions and trying different approaches")
print("- Debugging by examining intermediate states")
print("- A/B testing different conversation paths")
print("- Building interactive applications with branching narratives")
print("="*60)