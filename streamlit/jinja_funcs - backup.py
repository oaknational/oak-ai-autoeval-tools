import os
import re
import psycopg2
import json 
import time
import openai
import numpy as np
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape


import streamlit as st
import hashlib
from dotenv import load_dotenv  

import pandas as pd

def log_message(level, message):
    if level == 'error':
        st.error(message)
    elif level == 'warning':
        st.warning(message)
    elif level == 'info':
        st.info(message)
    else:
        st.write(message)


load_dotenv()

# os.chdir(os.getenv('FILE_PATH'))
jinja_path = os.getenv('JINJA_TEMPLATE_PATH')

def get_db_connection():
    # Replace these with your actual database connection details
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')

    # Connect to the database
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

    return conn

def execute_query(query):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query)
    data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()
    conn.close()
    return data

def json_to_html(json_obj, indent=0):
    html = ""
    indent_space = "&nbsp;&nbsp;" * indent

    if isinstance(json_obj, dict):
        html += f"{indent_space}{{<br>"
        for key, value in json_obj.items():
            html += f"{indent_space}&nbsp;&nbsp;&nbsp;&nbsp;<strong>{key}</strong>: "
            if isinstance(value, dict) or isinstance(value, list):
                html += "<br>" + json_to_html(value, indent + 1)
            else:
                html += f"{value},<br>"
        html += f"{indent_space}}}<br>"
    elif isinstance(json_obj, list):
        html += f"{indent_space}[<br>"
        for item in json_obj:
            if isinstance(item, dict) or isinstance(item, list):
                html += json_to_html(item, indent + 1)
            else:
                html += f"{indent_space}&nbsp;&nbsp;&nbsp;&nbsp;{item},<br>"
        html += f"{indent_space}]<br>"
    else:
        html += f"{indent_space}{json_obj}<br>"

    return html

def get_light_experiment_data():
    query_light = """
    SELECT 
        ex.id as experiment_id, 
        ex.experiment_name as experiment_name,
        ex.created_by as created_by, 
        ex.tracked as tracked, 
        ex.created_at as run_date, 
        t.name as teacher_name
    FROM 
        public.m_experiments ex 
    INNER JOIN m_teachers t ON t.id::text = ex.created_by
    WHERE ex.tracked = true
    ORder by ex.created_at desc;
    """
    return execute_query(query_light)

def get_full_experiment_data(selected_experiment_id):
    query_full = f"""
    SELECT 
        r.id as result_id, 
        r.experiment_id, 
        r.result, 
        r.justification, 
        r.prompt_id, 
        r.lesson_plan_id, 
        lp.lesson_id,
        lp.key_stage as key_stage_slug, 
        lp.subject as subject_slug,
        r.status as result_status,
        ex.sample_id as sample_id, 
        s.sample_title as sample_title, 
        ex.experiment_name as experiment_name,
        ex.created_by as created_by, 
        t.name as teacher_name, 
        ex.objective_id, 
        ex.llm_model, 
        ex.llm_model_temp, 
        ex.tracked as tracked, 
        ex.llm_max_tok, 
        ex.description as experiment_description, 
        r.created_at as run_date, 
        p.prompt_title, 
        p.output_format as prompt_output_format, 
        p.prompt_hash, 
        p.lesson_plan_params as prompt_lp_params
    FROM 
        public.m_results r 
    INNER JOIN m_experiments ex ON ex.id = r.experiment_id
    INNER JOIN m_prompts p ON p.id = r.prompt_id
    INNER JOIN m_teachers t ON t.id::text = ex.created_by
    INNER JOIN lesson_plans lp ON lp.id  = r.lesson_plan_id
    INNER JOIN m_sample_lesson_plans slp ON slp.lesson_plan_id = lp.id
    INNER JOIN m_samples s ON s.id = slp.sample_id
    WHERE ex.id = '{selected_experiment_id}' AND ex.tracked = true;
    """
    return execute_query(query_full)

def get_data(query):
    """Execute a query and return the results as a DataFrame."""
    try:
        conn = get_db_connection()
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs

# Queries (as strings) are provided in their respective functions
# @st.cache_data
def get_prompts():
    query = """
    WITH RankedPrompts AS (
        SELECT 
            id, 
            prompt_objective, 
            lesson_plan_params, 
            output_format, 
            rating_criteria, 
            prompt_title, 
            experiment_description, 
            objective_title,
            objective_desc, 
            version,
            created_at,
            ROW_NUMBER() OVER (PARTITION BY prompt_title ORDER BY version DESC) AS row_num
        FROM 
            public.m_prompts 
        WHERE
           
            id != '6a5acf94-f678-4dac-8460-efd58709f09f'
    )
    SELECT
        id, 
        prompt_objective, 
        lesson_plan_params, 
        output_format, 
        rating_criteria, 
        prompt_title, 
        experiment_description, 
        objective_title,
        objective_desc, 
        version
    FROM 
        RankedPrompts
    WHERE 
        row_num = 1;
    """
    return get_data(query)

# @st.cache_data
def get_samples():
    query = """
    SELECT
        m.id,
        m.sample_title,
        m.created_at,
        COUNT(l.lesson_plan_id) AS number_of_lessons
    FROM
        m_samples m
    LEFT JOIN
        m_sample_lesson_plans l ON m.id = l.sample_id
    GROUP BY
        m.id, m.sample_title, m.created_at
		order by m.created_at desc;
    """
    return get_data(query)

# @st.cache_data
def get_teachers():
    query = """
    SELECT id, name FROM m_teachers;
    """
    return get_data(query)


def get_samples_data(add_query):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(add_query)
    lesson_plans = cur.fetchall()
    cur.close()
    conn.close()
    return lesson_plans


def get_lesson_plans(limit):
    conn = get_db_connection()

    cur = conn.cursor()
    # get 200 lessons
    query = f"""
    SELECT * FROM lesson_plans {limit};
    """

    cur.execute(query)
    conn.commit()
    lesson_plans = cur.fetchall()
    cur.close()
    conn.close()
    return lesson_plans

def get_lesson_plans_by_id(sample_id, limit=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        query = """
        SELECT lp.id, lp.lesson_id, lp.json 
        FROM lesson_plans lp
        JOIN m_sample_lesson_plans slp ON lp.id::text = slp.lesson_plan_id::text
        WHERE slp.sample_id::text = %s
        """
        
        params = [str(sample_id)]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        cur.execute(query, params)
        lesson_plans = cur.fetchall()
        
        cur.close()
        conn.close()
        return lesson_plans
    except Exception as e:
        st.error(f"Error fetching lesson plans: {e}")
        return []


def add_experiment(experiment_name, sample_ids, created_by, tracked, llm_model='gpt-4', llm_model_temp=0.5, description='None', status='PENDING'):
    
    conn = get_db_connection()
    cur = conn.cursor()

    # mlflow.log_params({"LLM model": llm_model, "LLM Model Temp": llm_model_temp})
    sample_ids = ','.join(sample_ids)
    insert_query = f"""
    INSERT INTO m_experiments (created_at, updated_at, experiment_name, sample_id, llm_model, llm_model_temp, description, created_by, status, tracked) VALUES (now(), now(), '{experiment_name}', '{sample_ids}', '{llm_model}', '{llm_model_temp}', '{description}', '{created_by}', '{status}', '{tracked}')
    RETURNING id;
    """

    cur.execute(insert_query)
    experiment_id = cur.fetchone()[0]
    conn.commit()

    cur.close()
    conn.close()

    return experiment_id

def fix_json_format(json_string):
    try:
        # Try to load the JSON string to see if it's valid
        json.loads(json_string)
        return json_string
    except ValueError:
        # If it's not valid, try to fix common issues
        json_string = re.sub(r'\\\\"', r'"', json_string)  # Fix escaped quotes
        json_string = re.sub(r"'", r'"', json_string)      # Replace single quotes with double quotes
        json_string = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', json_string)  # Add quotes around keys if missing
        
        try:
            # Try to load the fixed JSON string
            json.loads(json_string)
            return json_string
        except ValueError:
            # If it still fails, return an empty JSON string
            return '{}'

def get_prompt(prompt_id):
    conn = get_db_connection()
    cur = conn.cursor()
    query = f"""
    SELECT id, prompt_objective, lesson_plan_params, output_format, rating_criteria, general_criteria_note, rating_instruction, prompt_title, experiment_description, objective_title, objective_desc
    FROM m_prompts
    WHERE id = '{prompt_id}';
    """
    cur.execute(query)
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        # Clean the rating_criteria before returning the result
        clean_rating_criteria = fix_json_format(result[4])
        return {
            'prompt_id': result[0],
            'prompt_objective': result[1],
            'lesson_plan_params': result[2],
            'output_format': result[3],
            'rating_criteria': clean_rating_criteria,
            'general_criteria_note': result[5],
            'rating_instruction': result[6],
            'prompt_title': result[7],
            'experiment_description': result[8],
            'objective_title': result[9],
            'objective_desc': result[10],
            # 'prompt_created_by': result[11]
        }
    return None

def process_prompt(prompt_details):
            
    if isinstance(prompt_details.get('rating_criteria'), str):
        try:
            # Clean the string by removing escape characters
            cleaned_criteria = prompt_details['rating_criteria'].replace('\\"', '"')
            prompt_details['rating_criteria'] = json.loads(cleaned_criteria)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            prompt_details['rating_criteria'] = {}

    # Ensure lesson_plan_params is a list
    if isinstance(prompt_details.get('lesson_plan_params'), str):
        try:
            prompt_details['lesson_plan_params'] = json.loads(prompt_details['lesson_plan_params'])
        except json.JSONDecodeError:
            prompt_details['lesson_plan_params'] = []

    prompt_details.setdefault('prompt_objective', '')
    prompt_details.setdefault('output_format', 'Boolean')
    prompt_details.setdefault('general_criteria_note', '')
    prompt_details.setdefault('rating_instruction', '')
    prompt_details.setdefault('prompt_title', '')
    prompt_details.setdefault('experiment_description', '')
    prompt_details.setdefault('objective_title', '')
    prompt_details.setdefault('objective_desc', '')

    return prompt_details

def render_prompt(lesson_plan, prompt_details):
    jinja_env = Environment(loader=FileSystemLoader(jinja_path), autoescape=select_autoescape(['html', 'xml']))
    
    template = jinja_env.get_template('prompt.jinja')
    if not template:
        return "Template could not be loaded."

    # Debug output to check the processed prompt details
    # print("Processed prompt details:", prompt_details)

    return template.render(
        lesson=lesson_plan, 
        prompt_objective=prompt_details['prompt_objective'],
        lesson_plan_params=prompt_details['lesson_plan_params'],
        output_format=prompt_details['output_format'],
        rating_criteria=prompt_details['rating_criteria'],
        general_criteria_note=prompt_details['general_criteria_note'],
        rating_instruction=prompt_details['rating_instruction'],
        prompt_title=prompt_details.get('prompt_title'),  
        experiment_description=prompt_details.get('experiment_description'),
        objective_title=prompt_details.get('objective_title'),
        objective_desc=prompt_details.get('objective_desc')
    )

def run_inference(lesson_plan, prompt_id, llm_model, llm_model_temp, timeout=15): 

    if set(lesson_plan.keys()) == {"title", "topic", "subject", "keyStage"}:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data is missing for this check."
            },
            "status": "ABORTED"
        }
    
    if not lesson_plan:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data is missing for this check."
            },
            "status": "ABORTED"
        }
    prompt_details = get_prompt(prompt_id)
    cleaned_prompt_details = process_prompt(prompt_details)
    prompt = render_prompt(lesson_plan, cleaned_prompt_details)
    
    if "Prompt details are missing" in prompt or "Missing data" in prompt:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data missing for this check."
            },
            "status": "ABORTED"
        }
    
    client = wrap_openai(openai.Client())
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=llm_model_temp,
            timeout=timeout,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        try:
            # Clean the response content by removing leading non-JSON text and control characters
            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith('```json'):
                raw_content = raw_content[7:].strip()
            if raw_content.endswith('```'):
                raw_content = raw_content[:-3].strip()
            cleaned_content = re.sub(r'[\n\r\t\\]', '', raw_content)
            
            success_response = {
                "response": json.loads(cleaned_content),
                "status": "SUCCESS"
            }
            return success_response
        except json.JSONDecodeError as e:
            print(response.choices[0].message.content)
            error_position = e.pos
            json_str = response.choices[0].message.content
            start_snippet = max(0, error_position - 40)
            end_snippet = min(len(json_str), error_position + 40)
            snippet = json_str[start_snippet:end_snippet]
            error_response = {
                "response": {
                    "result": None,
                    "justification": f"An error occurred: {e}. Problematic snippet: {repr(snippet)}"
                },
                "status": "FAILURE"
            }
            return error_response

    except Exception as e:
        error_response = {
            "response": {
                "result": None,
                "justification": "An error occurred: " + str(e)
            },
            "status": "FAILURE"
        }
        
        return error_response
    
def add_results(experiment_id, prompt_id, lesson_plan_id, score, justification, status):

    try:
        # Convert score to float, with fallback to boolean if necessary
        if score is not None and score != '':
            try:
                score = float(score)
            except ValueError:
                # make lowercase 
                score = score.lower()
                if score == 'true':
                    score = 1.0
                elif score == 'false':
                    score = 0.0
        else:

            print(f"Score: {score}")
            print(f"NONE TYPE prompt_id: {prompt_id}")
            print(f"NONE TYPE lesson_plan_id: {lesson_plan_id}")
            print(f'NONE TYPE justification: {justification}')
            print(f'NONE TYPE status: {status}')   

        # Get the database connection and cursor
        conn = get_db_connection()
        cur = conn.cursor()

        # Prepare the SQL query
        insert_query = """
        INSERT INTO m_results (created_at, updated_at, experiment_id, prompt_id, lesson_plan_id, result, justification, status)
        VALUES (now(), now(), %s, %s, %s, %s, %s, %s);
        """

        # Execute the query using parameterized SQL to prevent SQL injection
        cur.execute(insert_query, (experiment_id, prompt_id, lesson_plan_id, score, justification, status))

        # Commit the transaction
        conn.commit()

    except (psycopg2.DatabaseError, psycopg2.OperationalError) as db_err:
        # Log the error and rollback transaction
        log_message('error', f"Error executing query: {db_err}")
        conn.rollback()  # Rollback the transaction to avoid partial data entry

    except Exception as e:
        # Catch any other exceptions
        log_message('error', f"Error executing query: {e}")


        if conn:
            conn.rollback()

    finally:
        # Always close the cursor and connection to avoid resource leaks
        if cur:
            cur.close()
        if conn:
            conn.close()


def run_test(sample_id, prompt_id, experiment_id, limit, llm_model, llm_model_temp, timeout=15):
    lesson_plans = get_lesson_plans_by_id(sample_id, limit)
    total_lessons = len(lesson_plans)    
    log_message('info', f'Total lessons{total_lessons}')
    progress = st.progress(0)
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    # response = None
    # output = None
    
    for i, lesson in enumerate(lesson_plans):
        lesson_plan_id = lesson[0]
        lesson_id = lesson[1]
        lesson_json_str = lesson[2]
        
        try:
            if not lesson_json_str:
                log_message('error',f'Lesson JSON is None for lesson index {i}')
                continue
            content = json.loads(lesson_json_str)
        except json.JSONDecodeError as e:
            # Log detailed information when there is an error
            error_position = e.pos
            json_str = lesson_json_str
            start_snippet = max(0, error_position - 40)
            end_snippet = min(len(json_str), error_position + 40)
            snippet = json_str[start_snippet:end_snippet]
            log_message('error', f"Error decoding JSON for lesson index {i}:")
            log_message('error', f"Lesson Plan ID: {lesson_plan_id}")
            log_message('error', f"Lesson ID: {lesson_id}")
            log_message('error', f"Error Message: {e}")
            log_message('error', f"Problematic snippet: {repr(snippet)}")

            continue
        
        output = None  # Initialize output variable


        try:
            output = run_inference(content, prompt_id, llm_model, llm_model_temp, timeout=timeout)

            # Diagnostic print statement for all cases
            
            
            response = output.get('response')

            if 'status' not in output:
                log_message('error', f"Key 'status' missing in output: {output}")
                continue

            # Check if the response is organized into cycles
            if isinstance(response, dict) and all(isinstance(v, dict) for v in response.values()):
                # Process each cycle
                for cycle, cycle_data in response.items():
                    result = cycle_data.get('result')
                    justification = cycle_data.get('justification', '').replace("'", "")

                    add_results(
                        experiment_id,
                        prompt_id,
                        lesson_plan_id,
                        result,
                        justification,
                        output['status']
                    )
            else:
                # Handle response without cycles
                result = response.get('result')
                justification = response.get('justification', '').replace("'", "")

                add_results(
                    experiment_id,
                    prompt_id,
                    lesson_plan_id,
                    result,
                    justification,
                    output['status']
                )
                with placeholder1.container():
                    st.write(f'Inference Status: {output["status"]}')
                with placeholder2.container():
                    st.write(response) 
                    log_message('info', f"""
                    result = {output.get('response')}
                    status = {output.get('status')}
                    lesson_plan_id = {lesson_plan_id}
                    experiment_id = {experiment_id}
                    prompt_id = {prompt_id}
                    """)

        except KeyError as e:
            log_message('error', f"KeyError: Missing key in output: {e}")
            log_message('error', f"Output structure: {output}")
        except Exception as e:
            log_message('error', f"Unexpected error when adding results: {e}")
            log_message('error', f"Lesson Plan ID: {lesson_plan_id}, Prompt ID: {prompt_id}, Output: {output}")
        
        progress.progress((i + 1) / total_lessons)
    #remove placeholders

    placeholder1.empty()
    placeholder2.empty()


def update_status(experiment_id, status): 
    conn = get_db_connection()

    cur = conn.cursor()
    insert_query = f"""
    UPDATE m_experiments SET status = '{status}' WHERE id = '{experiment_id}';
    """
    cur.execute(insert_query)
    conn.commit()

    cur.close()
    conn.close()


def start_experiment(experiment_name, exp_description, sample_ids, created_by, prompt_ids, limit, llm_model,tracked, llm_model_temp=0.5):
    
    experiment_id = add_experiment(experiment_name, sample_ids, created_by,tracked,llm_model,llm_model_temp, description=exp_description)
    
    st.success('Experiment details saved with ID: {}'.format(experiment_id))

        # mlflow.set_experiment(experiment_name)
    for index, sample_id in enumerate(sample_ids):
        total_samples = len(sample_ids)
        st.write(f"Working on sample {index + 1} of {total_samples}")
        # Run experiment for each prompt
        for index, prompt_id in enumerate(prompt_ids):
            total_prompts = len(prompt_ids)
            st.write(f"Working on prompt {index + 1} of {total_prompts}")
            
            run_test(sample_id, prompt_id, experiment_id, limit, llm_model, llm_model_temp)

        
        st.write('Sample Completed!')
        update_status(experiment_id, 'COMPLETE')


def to_prompt_metadata_db(prompt_objective, lesson_plan_params, output_format, rating_criteria, general_criteria_note, rating_instruction, prompt_title, experiment_description, objective_title, objective_desc, prompt_created_by, version):
    conn = get_db_connection()
    cur = conn.cursor()

    unique_prompt_details = prompt_objective + json.dumps(lesson_plan_params) + output_format + json.dumps(rating_criteria) + general_criteria_note + rating_instruction
    prompt_hash = hashlib.sha256(unique_prompt_details.encode('utf-8')).digest()

    duplicates_check = "SELECT id FROM m_prompts WHERE prompt_hash = %s;"
    try:
        cur.execute(duplicates_check, (psycopg2.Binary(prompt_hash),))
        duplicates = cur.fetchall()
    except Exception as e:
        print(f"Error checking duplicates: {e}")

    if len(duplicates) == 0:
        insert_query = """
            INSERT INTO m_prompts (created_at, updated_at, prompt_objective, lesson_plan_params, output_format, rating_criteria, general_criteria_note, rating_instruction, prompt_hash, prompt_title, experiment_description, objective_title, objective_desc, created_by, version)
            VALUES (now(), now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """
        cur.execute(insert_query, (prompt_objective, lesson_plan_params, output_format, json.dumps(rating_criteria), general_criteria_note, rating_instruction, prompt_hash, prompt_title, experiment_description, objective_title, objective_desc, prompt_created_by, version))
        conn.commit()

        returned_id = cur.fetchone()[0]
    else:
        return duplicates[0][0]

    return returned_id

def generate_experiment_placeholders(model_name, temperature, limit, prompt_count, sample_count, teacher_name):
    # Placeholder name with LLM model, temperature, prompt and sample counts, limit, and teacher name
    placeholder_name = (f"{model_name}-temp:{temperature}-prompts:{prompt_count}-samples:{sample_count}-limit:{limit}-created:{teacher_name}")

    # Placeholder description
    placeholder_description = (f"{model_name} Evaluating with temperature {temperature}, "
                               f"using {prompt_count} prompts on {sample_count} samples, "
                               f"with a limit of {limit} lesson plans per sample. "
                               f"Run by {teacher_name}.")

    return placeholder_name, placeholder_description