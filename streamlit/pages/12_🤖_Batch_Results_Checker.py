""" 
Streamlit page for checking batches of evaluations have completed 
processing by OpenAI.
"""
import re
import json

import streamlit as st
from openai import OpenAI
from openai import BadRequestError, AuthenticationError, APIError

from utils.common_utils import (
    clear_all_caches
)
from utils.db_scripts import (
    get_batches,
    add_results,
    update_status,
    update_batch_status
)

# Function to check the status of the batch job
def check_batch_status(batch_ref):
    try:
        # Retrieve batch details using the OpenAI client library
        batch_details = client.batches.retrieve(batch_ref)
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


def add_batch_results(output):
    st.warning("Please do not close the page until the results have been retrieved.")
    # Split the JSON string into separate JSON objects
    # using a regex to detect the start of each object
    json_strings = re.findall(r'(\{.*?\})(?=\s*\{|\s*$)', output)
    
    for json_str in json_strings:
        try:
            # Convert the main JSON object
            st.write('HERE')
            st.write(json_str)
            data = json.loads(json_str)
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            cleaned_content = content.replace('json\n', '', 1).strip()
            cleaned_content = cleaned_content.strip('`').strip()
            inner_data = json.loads(cleaned_content)

            # Access specific fields as needed
            experiment_id, prompt_id, lesson_plan_id = data["custom_id"].split('+')
            batch_ref = data["id"]
            justification = inner_data.get("justification", "")
            result = inner_data.get("result", "")
            status = "SUCCESS"

            # Add result to the results table
            add_results(
                experiment_id,
                prompt_id,
                lesson_plan_id,
                result,
                justification,
                status
            )
            # Update status of experiment and batch
            status = "COMPLETE"
            update_status(experiment_id, status)
            update_batch_status(batch_ref, status)

        except json.JSONDecodeError as e:
            st.write("Failed to parse JSON:", e)
        except KeyError as e:
            st.write("Key missing in JSON:", e)


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

# Assuming batch_ref has been selected
if selected_batch != " ":
    batch_ref = selected_batch.split(" -- ")[0]  # Extract the batch_ref part
    status, output_file_id, error_file_id = check_batch_status(batch_ref)
    if status:
        st.write(f"The status of batch job {batch_ref} is: {status}")
        # Access batch results
        if status == 'completed':
            try:
                file_response = client.files.content(output_file_id)
                add_batch_results(file_response.text)
                st.success(
                    "Results retrieved and can be reviewed on the Visualise "
                    "Results page."
                )
            except:
                st.error(f"An unexpected error occurred: No output file.")
                file_response = client.files.content(error_file_id)
                st.write(file_response.text)
    else:
        st.write("Could not retrieve the batch status.")
