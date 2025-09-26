from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = init_chat_model("anthropic:claude-3-haiku-20240307")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Test the chatbot
if __name__ == "__main__":
    # Test with a simple message
    print("🤖 Testing the basic chatbot...")
    
    result = graph.invoke({"messages": [("user", "Hi there! What can you do?")]})
    print("Bot response:", result["messages"][-1].content)
    
    print("\n" + "="*50)
    print("🎯 Interactive Chat (type 'quit' to exit)")
    print("="*50)
    
    # Interactive chat loop
    messages = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
            
        # Add user message and get response
        messages.append(("user", user_input))
        result = graph.invoke({"messages": messages})
        
        # Extract the bot's response
        bot_response = result["messages"][-1].content
        print(f"Bot: {bot_response}")
        
        # Update messages with the full conversation
        messages = result["messages"]