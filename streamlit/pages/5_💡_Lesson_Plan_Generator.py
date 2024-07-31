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
    





def fetch_lesson_plan_sets():
    """
    Fetch the contents of the lesson_plan_sets table and load into a pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the lesson_plan_sets data.
    """
    try:
        conn = get_db_connection()
        query = "SELECT * FROM lesson_plan_sets;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


selection = st.selectbox('Select a lesson plan set to generate lesson plans with:', ['HB_Test_Set'])
# Fetch the data and load it into a DataFrame

if selection == 'HB_Test_Set':
    lessons_df = fetch_lesson_plan_sets()
    lessons_df['key_stage'] = lessons_df['key_stage'].replace(['KS1', 'KS2', 'KS3', 'KS4'], ['Key Stage 1', 'Key Stage 2', 'Key Stage 3', 'Key Stage 4'])

    st.write(lessons_df)





if 'llm_model' not in st.session_state: 
    st.session_state.llm_model = 'gpt-4o-mini'
if 'llm_model_temp' not in st.session_state:
    st.session_state.llm_model_temp = 0.3


llm_model_options = ['llama','gpt-4o-mini','gpt-4o','gpt-4',  'gpt-4-turbo']

st.session_state.llm_model = st.selectbox(
    'Select a model for lesson plan generation:',
    llm_model_options,
    index=llm_model_options.index(st.session_state.llm_model)
)


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

    llm_model = st.session_state.llm_model
    llm_model_temp = st.session_state.llm_model_temp

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
        
    endpoint = get_env_variable("ENDPOINT")
    username = get_env_variable("USERNAME")
    credential = get_env_variable("CREDENTIAL")

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
    def run_agent_openai_inference(prompt, llm_model, llm_model_temp, timeout=150):
        client = OpenAI( api_key= os.environ.get("OPENAI_API_KEY"), timeout=timeout)

        
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_model_temp,
                top_p=1,
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
    # Usage in Streamlit form
    with st.form(key='generation_form'):
        if st.form_submit_button('Start Generation'):
            for index, row in lessons_df.iterrows():
                # Replace placeholders with actual values in the prompt
                prompt = prompt_template.replace("{{key_stage}}", row['key_stage'])
                prompt = prompt.replace("{{subject}}", row['subject'])
                prompt = prompt.replace("{{lesson_title}}", row['lesson_title'])

                if llm_model != 'llama':
                    response = run_agent_openai_inference(prompt, llm_model, llm_model_temp)
                else:
                    response = run_agent_llama_inference(prompt, llm_model, llm_model_temp)

                st.write(f"Response for {row['key_stage']} - {row['subject']} - {row['lesson_title']}:")
                # Make response a string
                response = response['response']
                # response
                response = json.dumps(response)  # Ensure the response is correctly formatted as JSON
                # response
                # Remove escape characters and line breaks using regex
                response_cleaned = re.sub(r'\\n|\\r', '', response)
                # response_cleaned
                # Insert lesson plan into database
                lesson_id = 'HB_' + str(row['lesson_number'])
                generation_details_value =  llm_model+'_'+str(llm_model_temp)+'_'+'HB_Test_Set'
                lesson_plan_id = insert_single_lesson_plan(response_cleaned, row['key_stage'], row['subject'], lesson_id, generation_details_value)
                st.write(lesson_plan_id)