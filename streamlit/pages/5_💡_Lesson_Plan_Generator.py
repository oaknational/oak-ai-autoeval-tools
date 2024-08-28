import streamlit as st
import psycopg2
import pandas as pd
import os 
from dotenv import load_dotenv
import plotly.express as px
import numpy as np
import json
import re
import openai
from openai import OpenAI
from dataeditor import * 
import plotly.graph_objects as go
from utils import  log_message, get_db_connection, insert_single_lesson_plan
from constants import ErrorMessages
import requests

# Load environment variables
load_dotenv()



def execute_single_query(query, params):
    try:
        connection = get_db_connection()  # Assuming this function gets a database connection
        cursor = connection.cursor()
        cursor.execute(query, params)
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        log_message("error", f"Unexpected error executing query: {e}")
        return False
    





def fetch_lesson_plan_sets(limit=None):
    """
    Fetch the contents of the lesson_plan_sets table and load into a pandas DataFrame.

    Args:
        limit (int or None): The maximum number of rows to retrieve. If None or 0, fetch all rows.

    Returns:
        pd.DataFrame: DataFrame containing the lesson_plan_sets data.
    """
    try:
        conn = get_db_connection()  # Assuming this is a function that returns a connection object
        if limit and limit > 0:
            query = "SELECT * FROM lesson_plan_sets LIMIT %s;"
            df = pd.read_sql_query(query, conn, params=[limit])
        else:
            query = "SELECT * FROM lesson_plan_sets;"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def fetch_sample_sets(limit=None):
    """
    Fetch the contents of the lesson_plan_sets table and load into a pandas DataFrame.

    Args:
        limit (int or None): The maximum number of rows to retrieve. If None or 0, fetch all rows.

    Returns:
        pd.DataFrame: DataFrame containing the lesson_plan_sets data.
    """
    try:
        conn = get_db_connection()  # Assuming this is a function that returns a connection object
        if limit and limit > 0:
            query = """SELECT DISTINCT ON (subject)
                            lesson_number, 
                            subject, 
                            key_stage, 
                            lesson_title
                        FROM public.lesson_plan_sets
                        ORDER BY subject, key_stage, lesson_number LIMIT %s;"""
            df = pd.read_sql_query(query, conn, params=[limit])
        else:
            query = """SELECT DISTINCT ON (subject)
                            lesson_number, 
                            subject, 
                            key_stage, 
                            lesson_title
                        FROM public.lesson_plan_sets
                        ORDER BY subject, key_stage, lesson_number;"""
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define the clean_response function
def clean_response(content):
    try:
        # Assuming content is a JSON string, try to parse it
        content_json = json.loads(content)
        status = "SUCCESS" if content_json else "FAILURE"
        return content_json, status
    except json.JSONDecodeError:
        return content, "FAILURE"

# Function to get environment variable
def get_env_variable(var_name):
    try:
        return os.getenv(var_name)
    except KeyError:
        raise RuntimeError(f"Environment variable '{var_name}' not found")
    

def run_agent_llama_inference(prompt, llm_model, llm_model_temp):

        try:
            # Define the headers for the request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credential}"
            }

            # Create the payload with the data you want to send to the model
            data = {
                "messages": [
                    {"role": "user", "content": prompt},   # Adjust this structure based on API requirements
                ],
                "temperature": llm_model_temp,
                # 'temperature': llm_model_temp,
            }

            # Make the POST request to the model endpoint
            response = requests.post(endpoint, headers=headers, data=json.dumps(data))
            

            # Check the response status and content
            if response.status_code == 200:
                response_data = response.json()
                message = response_data['choices'][0]['message']['content']
                cleaned_content, status = clean_response(message)
                return {
                    "response": cleaned_content  # Add the elapsed time to the return value
                }
            else:
                log_message("error", f"Failed with status code {response.status_code}: {response.text}")
                return {
                    "response": {
                        "result": None,
                        "justification": f"Failed with status code {response.status_code}: {response.text}",
                    },
                    "status": "FAILURE" 
                }

        except Exception as e:
            log_message("error", f"Unexpected error during inference: {e}")
            return {
                "response": {
                    "result": None,
                    "justification": f"An error occurred: {e}",
                },
                "status": "FAILURE" # Include elapsed time even in case of failure
            }
        
def run_agent_openai_inference(prompt, llm_model, llm_model_temp,top_p=1, timeout=150):
    client = OpenAI( api_key= os.environ.get("OPENAI_API_KEY"), timeout=timeout)

    
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_model_temp,
            seed=42,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
        message = response.choices[0].message.content
        # print(message)
        cleaned_content, status = clean_response(message)
        return {
            "response": cleaned_content
        }

    except Exception as e:
        log_message("error", f"Unexpected error during inference: {e}")
        return {
            "response": {
                "result": None,
                "justification": f"An error occurred: {e}",
            },
            "status": "FAILURE",
        }
    
selection = st.selectbox('Select a lesson plan set to generate lesson plans with:', ['HB_Test_Set','Model_Compare_Set_10'])
# Fetch the data and load it into a DataFrame

if selection == 'HB_Test_Set':
    lessons_df = fetch_lesson_plan_sets(0)
    lessons_df['key_stage'] = lessons_df['key_stage'].replace(['KS1', 'KS2', 'KS3', 'KS4'], ['Key Stage 1', 'Key Stage 2', 'Key Stage 3', 'Key Stage 4'])

    st.write(lessons_df)
elif selection == 'Model_Compare_Set_10':
    lessons_df = fetch_sample_sets(0)
    lessons_df['key_stage'] = lessons_df['key_stage'].replace(['KS1', 'KS2', 'KS3', 'KS4'], ['Key Stage 1', 'Key Stage 2', 'Key Stage 3', 'Key Stage 4'])

    st.write(lessons_df)
else:
    st.error("Invalid selection. Please select a valid lesson plan set.")





if 'llm_model' not in st.session_state: 
    st.session_state.llm_model = 'gpt-4o-2024-05-13'
if 'llm_model_temp' not in st.session_state:
    st.session_state.llm_model_temp = 0.1


llm_model_options = ['llama','gpt-4o-mini-2024-07-18',
                     'gpt-4o-2024-05-13','gpt-4o-2024-08-06','chatgpt-4o-latest',
                     'gpt-4-turbo-2024-04-09','gpt-4-0125-preview','gpt-4-1106-preview']


st.session_state.llm_model = st.multiselect(
    'Select models for lesson plan generation:',
    llm_model_options,
    default=[st.session_state.llm_model] if isinstance(st.session_state.llm_model, str) else st.session_state.llm_model
)
st.session_state.llm_model

# todo: add number of lesson plans that will be generated for each model 



st.session_state.llm_model_temp = st.number_input(
    'Enter temperature for the model:',
    min_value=0.0, max_value=2.00,
    value=st.session_state.llm_model_temp,
    help='Minimum value is 0.0, maximum value is 2.00.'
)

response = None

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current script's directory
base_dir = os.path.dirname(script_dir)

# Define the file path for prompt_raw.txt in the data directory
prompt_file_path = os.path.join(base_dir, 'data', 'big_lp_generator_prompt.txt')


# Check if the file exists
if not os.path.exists(prompt_file_path):
    st.error(f"File not found: {prompt_file_path}")
else:
    # Read the prompt from data/prompt_raw.txt
    with open(prompt_file_path, 'r') as file:
        prompt_template = file.read()

    st.write('Review the Prompt for generations')
    with st.expander("Prompt Template", expanded=False):
        st.text_area("Generation Prompt", prompt_template, height=600)

llm_models = st.session_state.llm_model  # This will be a list of selected models from the multiselect
llm_model_temp = st.session_state.llm_model_temp


if 'top_p' not in st.session_state:
    st.session_state.top_p = 1.0  # Ensure this is a float


st.session_state.top_p = st.number_input(
    'Enter top_p for the model:',
    min_value=0.0, max_value=1.0,  # These should be floats
    value=float(st.session_state.top_p),  # Convert value to float
    step=0.01,  # You may need to specify the step value, e.g., 0.01
    help='Minimum value is 0.0, maximum value is 1.00.'
)




endpoint = get_env_variable("ENDPOINT")
username = get_env_variable("USERNAME")
credential = get_env_variable("CREDENTIAL")

# Usage in Streamlit form
with st.form(key='generation_form'):
    if st.form_submit_button('Start Generation'):
        for llm_model in llm_models:
            for index, row in lessons_df.iterrows():
                # Replace placeholders with actual values in the prompt
                prompt = prompt_template.replace("{{key_stage}}", row['key_stage'])
                prompt = prompt.replace("{{subject}}", row['subject'])
                prompt = prompt.replace("{{lesson_title}}", row['lesson_title'])

                if llm_model != 'llama':
                    response = run_agent_openai_inference(prompt, llm_model, llm_model_temp,st.session_state.top_p)
                else:
                    response = run_agent_llama_inference(prompt, llm_model, llm_model_temp)

                st.write(f"Response for {row['key_stage']} - {row['subject']} - {row['lesson_title']} with model {llm_model}:")
                
                # Extract the 'response' field from the API response
                response = response['response']
                
                # Convert the response to a JSON string
                response = json.dumps(response)
                
                # Clean up the response by removing escape characters and line breaks
                response_cleaned = re.sub(r'\\n|\\r', '', response)
                
                lesson_id = selection +'_'+ str(row['lesson_number'])+'_'+ 'gpt-4o_Comparison_Set'
                # st.write(f'Lesson ID: {lesson_id}')
                # st.write(f'llm_model: {llm_model}')
                # st.write(f'llm_model_temp: {llm_model_temp}')
                # st.write(f'top_p: {st.session_state.top_p}')
                # st.write(f"Selection: {selection}")
                generation_details_value = llm_model + '_' + str(llm_model_temp) + '_' + selection + '_' + str(st.session_state.top_p)
                st.write(f"Generation Details: {generation_details_value}")
                # Insert the generated lesson plan into the database
                lesson_plan_id = insert_single_lesson_plan(response_cleaned,lesson_id, row['key_stage'], row['subject'],  generation_details_value)
                # Display the lesson plan ID in the Streamlit app
                st.write(f"Lesson Plan ID: {lesson_plan_id}")