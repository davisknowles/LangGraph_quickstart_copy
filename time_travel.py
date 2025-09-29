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
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Add a second conversation to create more checkpoints
print("\n" + "="*50)
print("ADDING SECOND CONVERSATION")
print("="*50)

events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

print("\n" + "="*50)
print("REPLAYING STATE HISTORY")
print("="*50)

# Now replay the full state history
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state

print("\n" + "="*50)
print("TIME TRAVEL - RESUMING FROM CHECKPOINT")
print("="*50)

# Resume from the selected checkpoint
if to_replay:
    print("Resuming from checkpoint with config:")
    print("Next:", to_replay.next)
    print("Config:", to_replay.config)
    print(f"\nCheckpoint ID: {to_replay.config['configurable']['checkpoint_id']}")
    print("\nResuming execution from this checkpoint...")
    print("This demonstrates 'time travel' - we're going back to a previous state")
    print("and continuing execution from that point!")
    print("-" * 80)
    
    # The checkpoint_id in the to_replay.config corresponds to a state we've persisted to our checkpointer.
    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
else:
    print("No checkpoint with 6 messages found to replay from.")

print("\n" + "="*50)
print("TIME TRAVEL DEMONSTRATION COMPLETE")
print("="*50)
print("Key concepts demonstrated:")
print("1. Every step in the graph is automatically checkpointed")
print("2. get_state_history() lets you browse all past states")
print("3. You can resume execution from any checkpoint using its config")
print("4. This enables 'time travel' - going back and exploring different paths")
print("5. Useful for debugging, experimentation, and interactive applications")

print("\n" + "="*50)
print("ADDITIONAL TIME TRAVEL EXAMPLE")
print("="*50)
print("Let's try a different approach - resume from a different checkpoint...")

# Find a checkpoint with exactly 2 messages (earlier in the conversation)
different_checkpoint = None
for state in graph.get_state_history(config):
    if len(state.values["messages"]) == 2:
        different_checkpoint = state
        break

if different_checkpoint:
    print(f"Found checkpoint with 2 messages, next: {different_checkpoint.next}")
    print("This would be right after the first tool call...")
    print("If we resumed from here, we could take the conversation in a different direction!")
    print(f"Checkpoint ID: {different_checkpoint.config['configurable']['checkpoint_id']}")
    
    # Optional: Uncomment the lines below to actually resume from this earlier checkpoint
    # print("\nResuming from this earlier checkpoint:")
    # print("-" * 80)
    # for event in graph.stream(None, different_checkpoint.config, stream_mode="values"):
    #     if "messages" in event:
    #         event["messages"][-1].pretty_print()
else:
    print("No checkpoint with 2 messages found.")