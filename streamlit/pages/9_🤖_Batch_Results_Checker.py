""" 
Streamlit page for checking batches of evaluations have completed 
processing by OpenAI.
"""
import re
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from openai import BadRequestError, AuthenticationError, APIError
import psycopg2
from psycopg2.extras import execute_values
from utils.common_utils import (
    clear_all_caches, log_message
)
from utils.db_scripts import (
    get_batches,
    get_db_connection,
    update_status,
    update_batch_status,
    
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


def insert_batch_results(batch_data):
    """
    Insert batch results into the m_results table using batch inserts.
    
    Args:
        batch_data (list of tuples): Each tuple contains the following:
            experiment_id (str), prompt_id (str), lesson_plan_id (str), score (float), 
            justification (str), status (str)
    
    Returns:
        bool: True if the insert was successful, False otherwise.
    """
    
    # Prepare the SQL query without conflict handling
    insert_query = """
        INSERT INTO m_results (
            created_at, updated_at, experiment_id, prompt_id, 
            lesson_plan_id, result, justification, status
        ) VALUES %s
    """
    
    # Get the database connection
    conn = get_db_connection()
    if not conn:
        log_message("error", "Failed to establish database connection")
        return False

    try:
        with conn:
            with conn.cursor() as cur:
                # Use psycopg2's execute_values for efficient batch inserts
                execute_values(
                    cur,
                    insert_query,
                    batch_data,  # List of tuples for batch insert
                    template="(now(), now(), %s, %s, %s, %s, %s, %s)"  # Template matching number of columns
                )
        return True
    
    except (psycopg2.DatabaseError) as db_err:
        log_message("error", f"Database error occurred: {db_err}")
        conn.rollback()
        return False

    except Exception as e:
        log_message("error", f"Unexpected error executing query: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()

    


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
batches_data
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
            file_response = client.files.content(output_file_id)
            #save file_response.text a txt file
            lines = file_response.text.splitlines()
            json_lines = [line.strip() for line in lines if line.startswith('{"id": "batch_req')]
            messages = []
            justifications = []
            scores = []
            experiment_ids = []
            prompt_ids = []
            lesson_plan_ids = []
            statuses=[]
            experiment_id = None
            
            for line in json_lines:
                try:
                    json_obj = json.loads(line)
                    message_content = json_obj['response']['body']['choices'][0]['message']['content']
                    messages.append(message_content)

                    # Extract 'custom_id' from the main json_obj instead of message_content (which is a string)
                    custom_id = json_obj['custom_id']
                    experiment_id, prompt_id, lesson_plan_id = custom_id.split('+')
                     
                    experiment_ids.append(experiment_id)
                    prompt_ids.append(prompt_id)
                    lesson_plan_ids.append(lesson_plan_id)

                    # Extract the justification using regex
                    justification_match = re.search(r'"justification":\s*"(.*?)",\s*"result":', message_content, re.DOTALL)
                    justification = justification_match.group(1) if justification_match else None
                    justifications.append(justification)

                    # Extract the result using regex
                    score_match = re.search(r'"result":\s*"(.*?)"\s*}', message_content, re.DOTALL)
                    score = score_match.group(1) if score_match else None
                    scores.append(score)

                    status = "SUCCESS"
                    statuses.append(status)
                    # log_message("info", f"Attempting to insert: {experiment_id}, {prompt_id}, {lesson_plan_id}, {score}, {justification}, {status}")

    


                except (KeyError, json.JSONDecodeError):
                    messages.append(None)
                    justifications.append(None)
                    score.append(None)
                    experiment_ids.append(None)
                    prompt_ids.append(None)
                    lesson_plan_ids.append(None)

            # Create a DataFrame with multiple columns
            df = pd.DataFrame({
                'experiment_id': experiment_ids,
                'prompt_id': prompt_ids,
                'lesson_plan_id': lesson_plan_ids,
                'result': scores,
                'justification': justifications,
                'status': statuses
            })


            st.dataframe(df)
            # Add a button to insert batch results into the database
            if st.button("Insert Batch Results into Database"):
                # Insert batch results into the database
                success = True
                batch_data = []

                for idx, row in df.iterrows():
                    if row['result'] is not None and row['result'] != "":
                        try:
                            row['result'] = float(row['result'])
                        except ValueError:
                            score_lower = row['result'].lower()
                            if score_lower == "true":
                                row['result'] = 1.0
                            elif score_lower == "false":
                                row['result'] = 0.0
                    batch_data.append((
                        row['experiment_id'],
                        row['prompt_id'],
                        row['lesson_plan_id'],
                        row['result'],
                        row['justification'],
                        row['status']
                    ))

                # Once all the rows are collected, perform the batch insert
                if insert_batch_results(batch_data):
                    st.success("All batch results inserted successfully!")
                    status = "COMPLETE"
                    update_status(experiment_id, status)
                    update_batch_status(experiment_id, status)
                else:
                    st.error("There was an error inserting some batch results.")

            
    else:
        st.write("Could not retrieve the batch status.")


