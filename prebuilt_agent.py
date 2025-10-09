from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    conditions: str

checkpointer = InMemorySaver()

model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0
)

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    checkpointer=checkpointer,
    response_format=WeatherResponse  
)


# You need to provide a config with thread_id when using checkpointer
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config=config
)

print("Full response:", response)
print("*************")
print("Structured response:", response.get("structured_response"))