import streamlit as st
import requests
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv

# ---------------- Load environment variables ----------------
load_dotenv()
search = DuckDuckGoSearchRun()

# ---------------- Perplexity API key ----------------
if "PERPLEXITY_API_KEY" not in os.environ or not os.environ["PERPLEXITY_API_KEY"]:
    api_input = st.text_input("Enter your API key:", type="password")
    if api_input:
        os.environ["PERPLEXITY_API_KEY"] = api_input

api_key = os.environ.get("PERPLEXITY_API_KEY")
if not api_key:
    st.warning("Please enter your API key to use this app.")
    st.stop()

# ---------------- State definition ----------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# ---------------- Perplexity Sonar API call ----------------
def call_sonar_api(prompt: str) -> str:
    """Call Perplexity Sonar API to generate a response."""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------- Chatbot logic ----------------
def chatbot(state: State, search_enabled: bool = True):
    last_message = state["messages"][-1]

    # Safely get user input
    if isinstance(last_message, HumanMessage):
        user_question = last_message.content
    elif isinstance(last_message, dict):
        user_question = last_message.get("content", "")
    else:
        user_question = str(last_message)

    # Determine if this is a casual query
    casual_keywords = ["hi", "hello", "how are you", "what's up", "hey", "joke", "python"]
    is_casual = any(word in user_question.lower() for word in casual_keywords)

    if is_casual or not search_enabled:
        # All casual queries or when search is off go to Sonar
        response_text = call_sonar_api(user_question)
    else:
        # Only informational queries with search toggle on
        search_result = search.invoke(user_question)
        SYSTEM_PROMPT = f"""You are an Expert Web Research Assistant.

SEARCH RESULTS: {search_result}

USER QUESTION: {user_question}

Provide a well-structured, informative response based on the search results."""
        response_text = call_sonar_api(SYSTEM_PROMPT)

    # Append AI response to state
    state["messages"].append({"role": "assistant", "content": response_text})
    return {"messages": [{"role": "assistant", "content": response_text}]}

# ---------------- Build the graph ----------------
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# ---------------- Streamlit UI ----------------
st.title("🔍 AI Assistant Aaji")
st.markdown("Chat casually or search for information. Toggle web search in the sidebar.")

# Sidebar: toggle search
with st.sidebar:
    st.header("Settings")
    search_enabled = st.checkbox("Enable web search for queries", value=True)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history safely
for msg in st.session_state.messages:
    if isinstance(msg, (HumanMessage, AIMessage)):
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        content = msg.content
    elif isinstance(msg, dict):
        role = msg.get("role", "user")
        content = msg.get("content", "")
    else:
        role = "user"
        content = str(msg)

    with st.chat_message(role):
        st.markdown(content)

# Accept user input
if user_input := st.chat_input("What would you like to know?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                result = graph.invoke({
                    "messages": [{"role": "user", "content": user_input}],
                    "search_enabled": search_enabled
                })
                # Always access content safely
                ai_response_msg = result["messages"][-1]
                ai_response = (
                    ai_response_msg.content
                    if isinstance(ai_response_msg, AIMessage)
                    else ai_response_msg.get("content", str(ai_response_msg))
                )
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---------------- Sidebar Info ----------------
with st.sidebar:
    st.header("About")
    st.markdown("""
This AI assistant can respond conversationally and optionally search the web for answers.

**Features:**
- Interactive chat for casual conversation
- Web search enabled for informational queries
- Powered by Perplexity Sonar for reliable AI responses

**How to use:**
1. Type a question or message in the chat input
2. Toggle 'Enable web search' for information lookup
3. Chat casually without web search if toggle is off
              
      Made by Abhinav Singh
""")
