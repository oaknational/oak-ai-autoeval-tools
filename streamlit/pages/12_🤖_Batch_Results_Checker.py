""" 
Streamlit page for checking batches of evaluations have completed 
processing by OpenAI.
    
Functionality:
- xxxx.

- xxxx.
"""
import io
import json

import pandas as pd
import streamlit as st
from openai import OpenAI
from openai import BadRequestError, AuthenticationError, APIError


from utils.common_utils import (
    clear_all_caches,
    log_message,
    get_env_variable,
    render_prompt
)
from utils.formatting import (
    generate_experiment_placeholders,
    lesson_plan_parts_at_end,
    display_at_end_score_criteria,
    display_at_end_boolean_criteria,
    decode_lesson_json,
    process_prompt
)
from utils.db_scripts import (
    get_batches,
    get_prompts,
    get_samples,
    get_teachers,
    add_experiment,
    get_lesson_plans_by_id,
    get_prompt
)
from utils.constants import (
    OptionConstants,
    ColumnLabels,
    LessonPlanParameters
)

# Initialize the OpenAI client
client = OpenAI()

# Set page configuration
st.set_page_config(page_title="Batch Results", page_icon="🤖")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

# Page and sidebar headers
st.markdown("# 🤖 Batch Results Checker")
st.write(
    """
    This page allows you to check whether batches of evaluations have completed 
    processing by OpenAI.
    """
)

# Fetching data
batches_data = get_batches()

# Order batches_data by created_at
batches_data = batches_data.sort_values(by="created_at", ascending=False)

batches_data["batches_options"] = (
    batches_data["batch_ref"]
    + " -- "
    + batches_data["batch_description"]
    + " -- "
    + batches_data["created_by"]
)
batches_options = batches_data["batches_options"].tolist()
batches_options.insert(0, " ")


# Batch selection section
st.subheader("Batch selection")
selected_batch = st.selectbox(
    "Select pending batch to check status:",
    batches_options
)

# Function to check the status of the batch job
def check_batch_status(batch_ref):
    try:
        # Retrieve batch details using the OpenAI client library
        #batch = client.beta.threads.retrieve(batch_ref)
        batch_details = client.batches.retrieve(batch_ref)
        st.write("Batch details:\n\n", batch_details)
        
        # Extract the status from the batch details
        status = batch_details.status
        output_file_id = batch_details.output_file_id
        error_file_id = batch_details.error_file_id
        return status, output_file_id, error_file_id
    
    except BadRequestError as e:
        st.error(f"Invalid batch reference: {str(e)}")
    except AuthenticationError as e:
        st.error(f"Authentication failed. Check your API key: {str(e)}")
    except APIError as e:
        st.error(f"API error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    return None

# Assuming batch_ref has been selected as per previous example
if selected_batch != " ":
    batch_ref = selected_batch.split(" -- ")[0]  # Extract the batch_ref part
    status, output_file_id, error_file_id = check_batch_status(batch_ref)
    
    if status:
        st.write(f"The status of batch job {batch_ref} is: {status}")
        # Access batch results
        if status == 'completed':
            try:
                file_response = client.files.content(output_file_id)
                st.write(file_response.text)
            except:
                st.error(f"An unexpected error occurred: No output file.")
                file_response = client.files.content(error_file_id)
                st.write(file_response.text)
    else:
        st.write("Could not retrieve the batch status.")
else:
    st.write("No batch selected yet.")
    
# Add in stuff from handle_inference in utils/inference.py


'''
# Retrieve all batches
batches = client.batches.list()
st.write(len(batches.data))

# Loop through all batches and cancel each one
for batch in batches.data:
    batch_id = batch.id
    batch_status = batch.status
    
    # Only cancel batches that are in progress or pending
    if batch_status in ["in_progress", "pending"]:
        try:
            client.batches.cancel(batch_id)
            st.write(f"Cancelled batch: {batch_id}")
        except APIError as e:
            st.write(f"Failed to cancel batch {batch_id}: {e}")

batches = client.batches.list()
st.write(len(batches.data))
'''
