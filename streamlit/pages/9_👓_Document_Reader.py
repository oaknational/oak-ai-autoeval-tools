import streamlit as st
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

vector_store_id= os.getenv("OPEN_AI_VECTORSTORE_ID")
# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md", "pdf",'pptx')
)

# Ask the user for a question via `st.text_area`.
question = 'Explain each slides content.'


if uploaded_file and question:

    # Add a button to continue
    if st.button('Continue'):
            
        my_assistant = client.beta.assistants.update(
            assistant_id='asst_lbS2rr4CfjgtTFb89w0G01yT',
            tool_resources={'file_search': {'vector_store_ids': [vector_store_id]}}
        )

        # Upload the user-provided file to OpenAI
        message_file = client.files.create(
            file=uploaded_file, purpose="assistants"
        )
        
        # Create a thread and attach the file to the message
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                    # Attach the new file to the message.
                    "attachments": [
                        {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
                    ],
                }
            ]
        )
        
        # The thread now has a vector store with that file in its tool resources.
        print(thread.tool_resources.file_search)
    
        # Use the create and poll SDK helper to create a run and poll the status of the run until it's in a terminal state.
        if my_assistant is not None:
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id, assistant_id=my_assistant.id
            )

            messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

            message_content = messages[0].content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
                if file_citation := getattr(annotation, "file_citation", None):
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}")

            st.write(message_content.value)
            st.write("\n".join(citations))

        # Delete the message file
        client.files.delete(message_file.id)
        
