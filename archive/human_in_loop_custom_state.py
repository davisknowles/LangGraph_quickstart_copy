from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-haiku-20240307")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
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

# Test the chatbot with customized state and human assistance
if __name__ == "__main__":
    # Configuration for memory - each thread maintains separate conversation history
    config = {"configurable": {"thread_id": "1"}}
    
    print("👤 Testing Customized State with Human Assistance...")
    print("The bot will ask for help to verify personal information!")
    
    # First interaction - ask for personal info that should trigger human assistance
    print("\n" + "="*60)
    print("Step 1: Asking for personal information")
    print("="*60)
    
    user_input = "Hi! My name is Alice and my birthday is January 15th, 1990. Can you remember this information?"
    
    # Initialize state with empty name and birthday
    initial_state = {
        "messages": [("user", user_input)],
        "name": "",
        "birthday": ""
    }
    
    # Stream the conversation
    events = list(graph.stream(initial_state, config=config, stream_mode="values"))
    
    # Print the conversation so far
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    
    # Check if we were interrupted by human assistance
    current_state = graph.get_state(config)
    if current_state.next:
        print(f"\n🛑 INTERRUPTED for human assistance!")
        print("📋 The AI is asking for help to verify information...")
        
        # Get the interrupt data
        if hasattr(current_state, 'tasks') and current_state.tasks:
            interrupt_data = current_state.tasks[0].interrupts[0].value
            print(f"❓ Question: {interrupt_data.get('question', 'N/A')}")
            print(f"� Extracted Name: {interrupt_data.get('name', 'N/A')}")
            print(f"🎂 Extracted Birthday: {interrupt_data.get('birthday', 'N/A')}")
            
            # Get human response
            print("\n" + "="*60)
            print("Human Verification")
            print("="*60)
            
            correct = input("Is the extracted information correct? (y/n): ").lower().strip()
            
            human_response = {"correct": correct}
            
            if not correct.startswith('y'):
                name = input("Please provide the correct name: ")
                birthday = input("Please provide the correct birthday: ")
                human_response["name"] = name
                human_response["birthday"] = birthday
            
            # Resume with human response
            print("✅ Continuing with human feedback...")
            events = list(graph.stream(
                None,  # No new input, just resume
                config=config,
                stream_mode="values",
                input=human_response
            ))
            
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
    
    # Show final state
    final_state = graph.get_state(config)
    print(f"\n📊 Final State:")
    print(f"👤 Name in memory: {final_state.values.get('name', 'Not set')}")
    print(f"🎂 Birthday in memory: {final_state.values.get('birthday', 'Not set')}")
    
    print("\n" + "="*60)
    print("🔄 Interactive Chat with Custom State")
    print("The bot remembers your name and birthday!")
    print("Type 'quit' to exit")
    print("="*60)
    
    # Interactive loop with custom state and human assistance
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            final_state = graph.get_state(config)
            print("Goodbye! 👋")
            print(f"Final conversation had {len(final_state.values.get('messages', []))} messages")
            print(f"Remembered name: {final_state.values.get('name', 'Not set')}")
            print(f"Remembered birthday: {final_state.values.get('birthday', 'Not set')}")
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
        
        # Check if we're interrupted for human assistance
        current_state = graph.get_state(config)
        if current_state.next and hasattr(current_state, 'tasks') and current_state.tasks:
            print(f"\n🛑 HUMAN ASSISTANCE requested!")
            
            interrupt_data = current_state.tasks[0].interrupts[0].value
            print(f"❓ {interrupt_data.get('question', 'Verification needed')}")
            print(f"👤 Name: {interrupt_data.get('name', 'N/A')}")
            print(f"🎂 Birthday: {interrupt_data.get('birthday', 'N/A')}")
            
            correct = input("Is this information correct? (y/n): ").lower().strip()
            human_response = {"correct": correct}
            
            if not correct.startswith('y'):
                name = input("Correct name: ")
                birthday = input("Correct birthday: ")
                human_response["name"] = name
                human_response["birthday"] = birthday
            
            # Resume with human response
            events = list(graph.stream(
                None,
                config=config,
                stream_mode="values",
                input=human_response
            ))
            
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()