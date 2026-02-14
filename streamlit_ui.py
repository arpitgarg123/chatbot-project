import streamlit as st
from langgraph_backend import chatbot,retrieve_all_threads
import langgraph_backend as backend
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid

# utily functions
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['messages_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )

    # Safe handling for empty chats
    if state is None:
        return []

    if not hasattr(state, "values"):
        return []

    return state.values.get("messages", [])

# sessions set up
if "messages_history" not in st.session_state:
    st.session_state['messages_history'] = []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])
    
CONFIG = {
    'configurable' : {'thread_id': st.session_state['thread_id']},
    "metadata" : {
        "thread_id" : st.session_state['thread_id']
    },
    "run_name" : "chat_turn"
    }

# side bar ui 
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button('New Chat'):
    reset_chat()
#upload file for RAG
uploaded_file = st.sidebar.file_uploader("Choose a file")
if "rag_store" not in st.session_state:
    st.session_state.rag_store = None
if uploaded_file is not None:
    path = f"./{uploaded_file.name}"
    
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.session_state.rag_store = backend.ingest_pdf(path)
    backend.rag_store = st.session_state.rag_store
    st.sidebar.success("Document indexed!")
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message,HumanMessage):
                 role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": message.content})
        st.session_state['messages_history'] = temp_messages

# we are loading the messages history from the session state, if it exists
for message in st.session_state['messages_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])
        
user_input = st.chat_input("Type here...")


if user_input:
    # Show user's message
    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)
    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["messages_history"].append(
        {"role": "assistant", "content": ai_message}
    )