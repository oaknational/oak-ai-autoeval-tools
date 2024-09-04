""" Streamlit page for creating new prompt tests in the AutoEval app.
    Enables creation of new subsets of lesson plans to run evaluations on.
    It allows users to either create new prompts from scratch with guidance 
    or to create new prompts by using existing prompts as a template.

Functionality:

- Creates new prompt tests from scratch.
- Creates new prompt tests by modifying existing prompts (using them as 
    a template). 
    
Note:
    Prompts cannot be overwritten. 'Create Prompt' will warn the user if
    a prompt with the same title already exists. 'Modify Prompt' 
    will save the new prompt as a new version of the existing prompt.
"""

import streamlit as st

from utils.common_utils import (
    clear_all_caches, 
)

from utils.prompt_utils import (
    create_new_prompt, 
    modify_existing_prompt,
)




# Set page configuration
st.set_page_config(page_title="Create Prompt Tests", page_icon="📝")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

st.title("📝 Create Prompt Tests")
st.write("""
Welcome to the Prompt Test Creator! Here you can:
- **Create a new prompt**: Start from scratch and build a completely new prompt.
- **Modify an existing prompt**: Select an existing prompt to create a new version, tweaking the wording or changing the output format to improve performance.
""")
action = st.selectbox(
    "What would you like to do?",
    [" ", "Create a new prompt", "Modify an existing prompt"],
)
if action == "Create a new prompt":
    create_new_prompt()
elif action == "Modify an existing prompt":
    modify_existing_prompt()
