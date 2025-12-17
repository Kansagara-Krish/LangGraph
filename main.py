# Import load_dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Import typing tools for type hints
from typing import Annotated, Literal

# Import LangGraph components:
# - StateGraph: creates a graph/workflow for our chatbot
# - START, END: special nodes marking the beginning and end of the graph
from langgraph.graph import StateGraph, START, END

# Import add_messages: a helper function that properly merges messages in the state
from langgraph.graph.message import add_messages

# Import ChatGoogleGenerativeAI: the AI model from Google (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# Import pydantic tools for data validation (not strictly needed here but good practice)
from pydantic import BaseModel, Field

# Import TypedDict: defines the structure/type of our state dictionary
from typing_extensions import TypedDict

# Import os module to read environment variables
import os

# Load environment variables from .env file (where we store API keys)
load_dotenv()

# Get the Gemini API key from environment variables (.env file)
api_key = os.getenv("GEMINI_API_KEY")

# Check if API key exists - if not, raise an error to tell the user to add it
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")

# Initialize the Gemini AI model with our API key
# This is the "brain" that will answer our questions
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Using Google's Gemini 2.0 Flash model (free and fast)
    api_key=api_key  # Pass the API key for authentication
)

class Messageclassfier(BaseModel):
    message_type: Literal["logical", "emotional"] = Field(
        ..., description="Classification if the message is logical or emotional"
    )
    
# Define the State TypedDict - this is the data structure that flows through our graph
# It contains all the information our chatbot needs to track
class State(TypedDict):
    # 'messages' key holds a list of all messages in the conversation
    # add_messages is a helper that properly combines old and new messages
    messages: Annotated[list, add_messages]  # user messages
    
    message_type: str | None
    next: str | None

# Create the graph builder after State is defined
graph_builder = StateGraph(State)
    
# Create a StateGraph with our State structure
# This graph defines the workflow/steps of our chatbot
# Define the chatbot node - this is a function that processes the state
# It receives the current state and returns updated state

def classify_message(state:State):
    last_message = state["messages"][-1]
    classify_llm = llm.with_structured_output(Messageclassfier)

    # Safely get text from the last message whether it's a dict or an object
    def _get_content(msg):
        return getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)

    result = classify_llm.invoke([
        {
            "role":"system",
            "content":"""Classify the user message as either:
            - 'emotional' : if it askes for emotional support, therapy, deals with feelings, deals with feeling or personal problem
            - 'logical' : if it asks for facts, information logical analysis or practical solution 
            """
        },
        {"role":"user","content": _get_content(last_message)}
    ])
    return {"message_type": result.message_type}

def router(state:State):
    message_type= state.get("message_type","logical")
    if message_type == "emotional":
        return {"next":"therapist"}
    
    return {"next":"logical"}
    
def therapist_agent(state:State):
    last_message = state["messages"][-1]
    
    # helper to extract content
    def _get_content(msg):
        return getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)

    messages = [{
        "role":"system","content":"""You are the compassionate therapist. Focus on the emotional aspects of the users message
        Show empathy, valuable their fellings amd help them process their emotional, Ask thoghtful questions to help them explore their feelings more deelpy.
        Avoid giving logical solutions unless explacitly asked. """
        
    },
                {
                    "role":"user",
                    "content": _get_content(last_message)
                }
                ]
    
    # Invoke the model with the constructed messages
    reply = llm.invoke(messages)
    # Ensure we return under the `messages` key to match State
    return {"messages": [{"role": "assistant", "content": getattr(reply, "content", None) or (reply.get("content") if isinstance(reply, dict) else None)}]}

def logical_agent(state:State):
    last_message = state["messages"][-1]
    
    def _get_content(msg):
        return getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)

    messages = [{
        "role":"system","content":"""You are a purely logical assistant. Focus only on facts and information.
        Provide clear, concise answers based in the logic and evidence.
        Do not address emotions or provide emotional support.
        Be direct and Straghtforward in your responses
        """        
    },
    {
    "role":"user",
    "content": _get_content(last_message)
    }
    ]
    
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": getattr(reply, "content", None) or (reply.get("content") if isinstance(reply, dict) else None)}]}
    

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    path_map={"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist",END)
graph_builder.add_edge("logical",END)

# Compile the graph - convert it into a runnable workflow
# This prepares the graph for execution
graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [],"message_type":None}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit","quit"]:
            print("Exiting chat. Goodbye!")
            break

        state["messages"].append({"role":"user","content":user_input})
        state = graph.invoke(state)
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print("AI:",state["messages"][-1].content)


if __name__ == "__main__":    
    run_chatbot()
# Define the chatbot node - this is a function that processes the state