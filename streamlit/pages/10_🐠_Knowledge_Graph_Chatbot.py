import streamlit as st
from utils import write_message
#from agent import generate_response

# Page Config
st.set_page_config("Biology KS4", page_icon=":tropical_fish:")

st.write("## Biology KS4 Curriculum Assistant")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm you guide to the Biology KS4 curriculum! How can I help you?"}
    ]

# Submit handler
def handle_submit(message):
    with st.spinner('Thinking...'):
        # Call the agent
        response = message
        #response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("Ask your question..."):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
