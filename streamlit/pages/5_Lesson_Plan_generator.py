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
import uuid
from utils import  log_message
from constants import ErrorMessages

# Load environment variables
load_dotenv()
def get_db_connection():
    """ Establish a connection to the PostgreSQL database.

    Returns:
        conn: connection object to interact with the database.
    """
    try:
        conn = psycopg2.connect(
            dbname=get_env_variable("DB_NAME"),
            user=get_env_variable("DB_USER"),
            password=get_env_variable("DB_PASSWORD"),
            host=get_env_variable("DB_HOST"),
            port=get_env_variable("DB_PORT")
        )
        return conn
    except psycopg2.Error as e:
        log_message("error", f"Error connecting to the database: {e}")
        return None

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
    

def insert_lesson_plan(json_data, key_stage, subject, lesson_id, geneartion_details):
    try:
        id_value = str(uuid.uuid4())
        lesson_id_value = lesson_id
        json_value = json_data
        generation_details_value = geneartion_details
        key_stage_value = key_stage
        subject_value = subject

        query = """
            INSERT INTO lesson_plans (
                id, lesson_id, json, generation_details, created_at,
                key_stage, subject)
            VALUES (%s, %s, %s, %s, now(), %s, %s);
        """
        params = (
            id_value, lesson_id_value, json_value, generation_details_value,
            key_stage_value, subject_value
        )

        success = execute_single_query(query, params)
        return (
            "Lesson plan inserted successfully." if success else 
            "Unexpected error occurred while inserting the lesson plan."
        )
    except Exception as e:
        log_message("error", f"Unexpected error occurred while inserting the lesson plan: {e}")
        return "Unexpected error occurred while inserting the lesson plan."





response = None



#read csv lessons.csv

# Define the file path for lessons.csv
lessons_path = os.path.join(os.path.dirname(__file__), 'lessons.csv')

# Check if the lessons file exists
if not os.path.exists(lessons_path):
    st.error(f"File not found: {lessons_path}")
else:
    # Read lessons.csv into a dataframe
    lessons_df = pd.read_csv(lessons_path)

    #replave KS1, KS2, KS3, KS4 with Key Stage 1, Key Stage 2, Key Stage 3, Key Stage 4
    lessons_df['Key Stage'] = lessons_df['Key Stage'].replace(['KS1', 'KS2', 'KS3', 'KS4'], ['Key Stage 1', 'Key Stage 2', 'Key Stage 3', 'Key Stage 4'])
    
    st.write(lessons_df)



    

file_path = os.path.join(os.path.dirname(__file__), 'prompt_raw.txt')


# Check if the file exists
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
else:
    # Read the prompt from data/prompt_raw.txt
    with open(file_path, 'r') as file:
        prompt_template = file.read()



    llm_model = "gpt-4o-mini"
    llm_model_temp = 0.3
        
    def log_message(level, message):
        """ Log a message using Streamlit's log functions based on the level.

        Args:
            level (str): Log level ('error', 'warning', 'info').
            message (str): Message to log.
            
        Returns:
            None
        """
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        elif level == "info":
            st.info(message)
        elif level == "success":
            st.success(message)
        else:
            st.write(message)

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

    def run_agent_inference(prompt, llm_model, llm_model_temp, timeout=150):
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
    with st.form(key='experiment_form'):
        if st.form_submit_button('Run Agent'):
            for index, row in lessons_df.iterrows():
                # Replace placeholders with actual values in the prompt
                prompt = prompt_template.replace("{{key_stage}}", row['Key Stage'])
                prompt = prompt.replace("{{subject}}", row['Subject'])
                prompt = prompt.replace("{{lesson_title}}", row['Lesson Title'])

                response = run_agent_inference(prompt, llm_model, llm_model_temp)
                st.write(f"Response for {row['Key Stage']} - {row['Subject']} - {row['Lesson Title']}:")
                # Make response a string
                response = response['response']
                # response
                response = json.dumps(response)  # Ensure the response is correctly formatted as JSON
                # response
                # Remove escape characters and line breaks using regex
                response_cleaned = re.sub(r'\\n|\\r', '', response)
                # response_cleaned
                # Insert lesson plan into database
                lesson_id = 'HB_' + str(row['Lesson number'])
                generation_details_value =  llm_model+'_'+str(llm_model_temp)+'_'+'HB_Test_Set'
                result = insert_lesson_plan(response_cleaned, row['Key Stage'], row['Subject'], lesson_id, generation_details_value)
                st.write(result)