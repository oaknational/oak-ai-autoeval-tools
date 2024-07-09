""" Utility and helper functions.

Functions:

- log_message
- get_db_connection
- execute_query
- get_data
- json_to_html
- get_light_experiment_data
- get_full_experiment_data
- get_prompts
- get_samples
- get_teachers
- get_samples_data
- get_lesson_plans
- get_lesson_plans_by_id
- add_experiment
- fix_json_format
- get_prompt
- process_prompt
- render_prompt
- run_inference
- add_results
- run_test
- update_status
- start_experiment
- to_prompt_metadata_db
- generate_experiment_placeholders
"""

# Import the required libraries and modules
import os
import re
import json
import time
import openai
import hashlib
import psycopg2
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from jinja2 import Environment, FileSystemLoader, select_autoescape

#import numpy as np
#from langsmith import traceable
#from jinja2 import Template

# Load environment variables from .env file
load_dotenv()

# Set Jinja template path from environment variable
jinja_path = os.getenv('JINJA_TEMPLATE_PATH')


def log_message(level, message):
    """
    Log a message using Streamlit's log functions based on the level.

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
    else:
        st.write(message)


def get_db_connection():
    """Establish a connection to the PostgreSQL database.

    Returns:
        conn: connection object to interact with the database.
    """
    # Retrieve environment variables for database connection
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")

    # Connect to the database
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    return conn


def execute_query(query):
    """
    Execute a SQL query and returns the results as a Pandas DataFrame.

    Args:
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame containing the query results.
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()
    
    # Execute query and return results as a DataFrame
    cur = conn.cursor()
    cur.execute(query)
    data = pd.DataFrame(
        cur.fetchall(), columns=[desc[0] for desc in cur.description]
    )
    cur.close()
    conn.close()
    return data


# IS THIS FUNCTION USED???? SAME AS ABOVE ???? NEEDS COMMENTING
def get_data(query):
    """
    Execute a query and returns the results as a Pandas DataFrame.

    Args:
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame containing the query results.
    """
    try:
        conn = get_db_connection()
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs


def json_to_html(json_obj, indent=0):
    """
    Convert a JSON object to an HTML-formatted string recursively.

    Args:
        json_obj (dict or list): JSON object to convert.
        indent (int): Current level of indentation for formatting.

    Returns:
        str: HTML-formatted string representing the JSON object.
    """
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
    """
    Retrieve light experiment data from the database.

    Returns:
        pd.DataFrame: DataFrame with light experiment data.
    """
    query_light = (
        """
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
        ORDER by ex.created_at DESC;
        """
    )
    return execute_query(query_light)


def get_full_experiment_data(selected_experiment_id):
    """ Retrieve full data for a selected experiment ID from the database.

    Args:
    - selected_experiment_id (str): ID of experiment to fetch data for.

    Returns:
        pd.DataFrame: DataFrame with full experiment data.
    """
    query_full = (
        f"""
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
        WHERE 
            ex.id = '{selected_experiment_id}' 
            AND ex.tracked = true;
        """
    )
    return execute_query(query_full)



# @st.cache_data
def get_prompts():
    """ Retrieve prompts data from the database.

    Returns:
        pd.DataFrame: DataFrame with prompts data.
    """
    query = (
        """
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
                ROW_NUMBER() OVER (
                    PARTITION BY prompt_title 
                    ORDER BY version DESC
                ) AS row_num
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
    )
    return get_data(query)


# @st.cache_data
def get_samples():
    """ Retrieve samples data from the database.

    Returns:
        pd.DataFrame: DataFrame with samples data.
    """
    query = (
        """
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
    )
    return get_data(query)


# @st.cache_data
def get_teachers():
    """ Retrieve teachers data from the database.

    Returns:
        pd.DataFrame: DataFrame with teachers data.
    """
    query = "SELECT id, name FROM m_teachers;"
    return get_data(query)


def get_samples_data(add_query):
    """
    Retrieves lesson plans data from the database based on an additional 
    query.

    Args:
        add_query (str): Additional query to execute.

    Returns:
        list: List of lesson plans fetched from the database.
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()
    
    # Execute query and return results
    cur = conn.cursor()
    cur.execute(add_query)
    lesson_plans = cur.fetchall()
    cur.close()
    conn.close()
    return lesson_plans


def get_lesson_plans(limit):
    """ Retrieve lesson plans data from the database with a specified 
    limit.

    Args:
        limit (str): Limitation condition for the query.

    Returns:
        list: List of lesson plans fetched from the database.
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Get limited number of lessons
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM m_lesson_plans ORDER BY created_at DESC LIMIT %s;",
        (limit,)
    )
    conn.commit()
    lesson_plans = cur.fetchall()
    cur.close()
    conn.close()
    return lesson_plans


def get_lesson_plans_by_id(sample_id, limit=None):
    """ Retrieve lesson plans based on a sample ID.

    Args:
        sample_id (str): ID of the sample.
        limit (int, optional): Maximum number of records to fetch.

    Returns:
        list: List of tuples representing lesson plans.
    """
    try:
        # Establish connection to the PostgreSQL database
        conn = get_db_connection()
        
        # Get lessons plans by sample ID
        cur = conn.cursor()
        query = (
            """
            SELECT lp.id, lp.lesson_id, lp.json 
            FROM lesson_plans lp
            JOIN m_sample_lesson_plans slp 
                ON lp.id::text = slp.lesson_plan_id::text
            WHERE slp.sample_id::text = %s
            """
        )
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


def add_experiment(
    experiment_name, sample_ids, created_by, tracked,
    llm_model="gpt-4", llm_model_temp=0.5, description="None",
    status="PENDING"
):
    """ Add a new experiment to the database.

    Args:
        experiment_name (str): Name of the experiment.
        sample_ids (list): List of sample IDs associated with the experiment.
        created_by (str): Name of the user who created the experiment.
        tracked (bool): Flag indicating whether the experiment is tracked.
        llm_model (str, optional): Name of the LLM model used.
        llm_model_temp (float, optional): Temperature parameter for LLM.
        description (str, optional): Description of the experiment.
        status (str, optional): Status of the experiment.

    Returns:
        int: ID of the newly added experiment.
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # mlflow.log_params({"LLM model": llm_model, "LLM Model Temp": llm_model_temp})
    
    # Add a new experiment to the `m_experiments` table
    sample_ids = ",".join(sample_ids)
    insert_query = (
        f"""
        INSERT INTO m_experiments (created_at, updated_at, 
            experiment_name, sample_id, llm_model, llm_model_temp, 
            description, created_by, status, tracked) VALUES (now(), now(), 
            '{experiment_name}', '{sample_ids}', '{llm_model}', 
            '{llm_model_temp}', '{description}', '{created_by}', '{status}', 
            '{tracked}')
        RETURNING id;
        """
    )
    cur = conn.cursor()
    cur.execute(insert_query)
    experiment_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return experiment_id


def fix_json_format(json_string):
    """ Fix JSON formatting issues in a given JSON string.

    Args:
        json_string (str): JSON string to fix.

    Returns:
        str: Fixed JSON string or an empty JSON object if fixing fails.
    """
    try:
        # Try to load the JSON string to see if it's valid
        json.loads(json_string)
        return json_string
    
    except ValueError:
        # If it's not valid, try to fix common issues
        
        # Fix escaped quotes
        json_string = re.sub(r'\\\\"', r'"', json_string)  
        
        # Replace single quotes with double quotes
        json_string = re.sub(
            r"'", r'"', json_string
        )
        # Add quotes around keys if missing
        json_string = re.sub(
            r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', json_string
        )

        try:
            # Retry to load the JSON string to see if it's valid
            json.loads(json_string)
            return json_string
        
        except ValueError:
            # If it still fails, return an empty JSON string
            return "{}"


def get_prompt(prompt_id):
    """ Retrieve prompt details based on a prompt ID.

    Args:
        Prompt_id (str): ID of the prompt.

    Returns:
        dict: Dictionary containing prompt details, 
        or None if prompt is not found.
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()
    
    # Get prompt details by prompt ID
    query = (
        f"""
        SELECT id, prompt_objective, lesson_plan_params, output_format, 
            rating_criteria, general_criteria_note, rating_instruction, 
            prompt_title, experiment_description, objective_title, 
            objective_desc
        FROM m_prompts
        WHERE id = '{prompt_id}';
        """
    )
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result:
        # Clean the rating_criteria before returning the result
        clean_rating_criteria = fix_json_format(result[4])
        return {
            "prompt_id": result[0],
            "prompt_objective": result[1],
            "lesson_plan_params": result[2],
            "output_format": result[3],
            "rating_criteria": clean_rating_criteria,
            "general_criteria_note": result[5],
            "rating_instruction": result[6],
            "prompt_title": result[7],
            "experiment_description": result[8],
            "objective_title": result[9],
            "objective_desc": result[10],
            # 'prompt_created_by': result[11]
        }
    return None

"""
GOT TO HERE ........
"""


def process_prompt(prompt_details):

    if isinstance(prompt_details.get("rating_criteria"), str):
        try:
            # Clean the string by removing escape characters
            cleaned_criteria = prompt_details["rating_criteria"].replace('\\"', '"')
            prompt_details["rating_criteria"] = json.loads(cleaned_criteria)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            prompt_details["rating_criteria"] = {}

    # Ensure lesson_plan_params is a list
    if isinstance(prompt_details.get("lesson_plan_params"), str):
        try:
            prompt_details["lesson_plan_params"] = json.loads(
                prompt_details["lesson_plan_params"]
            )
        except json.JSONDecodeError:
            prompt_details["lesson_plan_params"] = []

    prompt_details.setdefault("prompt_objective", "")
    prompt_details.setdefault("output_format", "Boolean")
    prompt_details.setdefault("general_criteria_note", "")
    prompt_details.setdefault("rating_instruction", "")
    prompt_details.setdefault("prompt_title", "")
    prompt_details.setdefault("experiment_description", "")
    prompt_details.setdefault("objective_title", "")
    prompt_details.setdefault("objective_desc", "")

    return prompt_details


def render_prompt(lesson_plan, prompt_details):
    jinja_env = Environment(
        loader=FileSystemLoader(jinja_path),
        autoescape=select_autoescape(["html", "xml"]),
    )

    template = jinja_env.get_template("prompt.jinja")
    if not template:
        return "Template could not be loaded."

    # Debug output to check the processed prompt details
    # print("Processed prompt details:", prompt_details)

    return template.render(
        lesson=lesson_plan,
        prompt_objective=prompt_details["prompt_objective"],
        lesson_plan_params=prompt_details["lesson_plan_params"],
        output_format=prompt_details["output_format"],
        rating_criteria=prompt_details["rating_criteria"],
        general_criteria_note=prompt_details["general_criteria_note"],
        rating_instruction=prompt_details["rating_instruction"],
        prompt_title=prompt_details.get("prompt_title"),
        experiment_description=prompt_details.get("experiment_description"),
        objective_title=prompt_details.get("objective_title"),
        objective_desc=prompt_details.get("objective_desc"),
    )


def run_inference(lesson_plan, prompt_id, llm_model, llm_model_temp, timeout=15):

    if set(lesson_plan.keys()) == {"title", "topic", "subject", "keyStage"}:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data is missing for this check.",
            },
            "status": "ABORTED",
        }

    if not lesson_plan:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data is missing for this check.",
            },
            "status": "ABORTED",
        }
    prompt_details = get_prompt(prompt_id)
    cleaned_prompt_details = process_prompt(prompt_details)
    prompt = render_prompt(lesson_plan, cleaned_prompt_details)

    if "Prompt details are missing" in prompt or "Missing data" in prompt:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data missing for this check.",
            },
            "status": "ABORTED",
        }

    client = wrap_openai(openai.Client())
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_model_temp,
            timeout=timeout,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        try:
            # Clean the response content by removing leading non-JSON text and control characters
            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:].strip()
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3].strip()
            cleaned_content = re.sub(r"[\n\r\t\\]", "", raw_content)

            success_response = {
                "response": json.loads(cleaned_content),
                "status": "SUCCESS",
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
                    "justification": f"An error occurred: {e}. Problematic snippet: {repr(snippet)}",
                },
                "status": "FAILURE",
            }
            return error_response

    except Exception as e:
        error_response = {
            "response": {
                "result": None,
                "justification": "An error occurred: " + str(e),
            },
            "status": "FAILURE",
        }

        return error_response


def add_results(experiment_id, prompt_id, lesson_plan_id, score, justification, status):

    try:
        # Convert score to float, with fallback to boolean if necessary
        if score is not None and score != "":
            try:
                score = float(score)
            except ValueError:
                # make lowercase
                score = score.lower()
                if score == "true":
                    score = 1.0
                elif score == "false":
                    score = 0.0
        else:

            print(f"Score: {score}")
            print(f"NONE TYPE prompt_id: {prompt_id}")
            print(f"NONE TYPE lesson_plan_id: {lesson_plan_id}")
            print(f"NONE TYPE justification: {justification}")
            print(f"NONE TYPE status: {status}")

        # Get the database connection and cursor
        conn = get_db_connection()
        cur = conn.cursor()

        # Prepare the SQL query
        insert_query = """
        INSERT INTO m_results (created_at, updated_at, experiment_id, prompt_id, lesson_plan_id, result, justification, status)
        VALUES (now(), now(), %s, %s, %s, %s, %s, %s);
        """

        # Execute the query using parameterized SQL to prevent SQL injection
        cur.execute(
            insert_query,
            (experiment_id, prompt_id, lesson_plan_id, score, justification, status),
        )

        # Commit the transaction
        conn.commit()

    except (psycopg2.DatabaseError, psycopg2.OperationalError) as db_err:
        # Log the error and rollback transaction
        log_message("error", f"Error executing query: {db_err}")
        conn.rollback()  # Rollback the transaction to avoid partial data entry

    except Exception as e:
        # Catch any other exceptions
        log_message("error", f"Error executing query: {e}")

        if conn:
            conn.rollback()

    finally:
        # Always close the cursor and connection to avoid resource leaks
        if cur:
            cur.close()
        if conn:
            conn.close()


def run_test(
    sample_id, prompt_id, experiment_id, limit, llm_model, llm_model_temp, timeout=15
):
    lesson_plans = get_lesson_plans_by_id(sample_id, limit)
    total_lessons = len(lesson_plans)
    log_message("info", f"Total lessons{total_lessons}")
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
                log_message("error", f"Lesson JSON is None for lesson index {i}")
                continue
            content = json.loads(lesson_json_str)
        except json.JSONDecodeError as e:
            # Log detailed information when there is an error
            error_position = e.pos
            json_str = lesson_json_str
            start_snippet = max(0, error_position - 40)
            end_snippet = min(len(json_str), error_position + 40)
            snippet = json_str[start_snippet:end_snippet]
            log_message("error", f"Error decoding JSON for lesson index {i}:")
            log_message("error", f"Lesson Plan ID: {lesson_plan_id}")
            log_message("error", f"Lesson ID: {lesson_id}")
            log_message("error", f"Error Message: {e}")
            log_message("error", f"Problematic snippet: {repr(snippet)}")

            continue

        output = None  # Initialize output variable

        try:
            output = run_inference(
                content, prompt_id, llm_model, llm_model_temp, timeout=timeout
            )

            # Diagnostic print statement for all cases

            response = output.get("response")

            if "status" not in output:
                log_message("error", f"Key 'status' missing in output: {output}")
                continue

            # Check if the response is organized into cycles
            if isinstance(response, dict) and all(
                isinstance(v, dict) for v in response.values()
            ):
                # Process each cycle
                for cycle, cycle_data in response.items():
                    result = cycle_data.get("result")
                    justification = cycle_data.get("justification", "").replace("'", "")

                    add_results(
                        experiment_id,
                        prompt_id,
                        lesson_plan_id,
                        result,
                        justification,
                        output["status"],
                    )
            else:
                # Handle response without cycles
                result = response.get("result")
                justification = response.get("justification", "").replace("'", "")

                add_results(
                    experiment_id,
                    prompt_id,
                    lesson_plan_id,
                    result,
                    justification,
                    output["status"],
                )
                with placeholder1.container():
                    st.write(f'Inference Status: {output["status"]}')
                with placeholder2.container():
                    st.write(response)
                    log_message(
                        "info",
                        f"""
                    result = {output.get('response')}
                    status = {output.get('status')}
                    lesson_plan_id = {lesson_plan_id}
                    experiment_id = {experiment_id}
                    prompt_id = {prompt_id}
                    """,
                    )

        except KeyError as e:
            log_message("error", f"KeyError: Missing key in output: {e}")
            log_message("error", f"Output structure: {output}")
        except Exception as e:
            log_message("error", f"Unexpected error when adding results: {e}")
            log_message(
                "error",
                f"Lesson Plan ID: {lesson_plan_id}, Prompt ID: {prompt_id}, Output: {output}",
            )

        progress.progress((i + 1) / total_lessons)
    # remove placeholders

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


def start_experiment(
    experiment_name,
    exp_description,
    sample_ids,
    created_by,
    prompt_ids,
    limit,
    llm_model,
    tracked,
    llm_model_temp=0.5,
):

    experiment_id = add_experiment(
        experiment_name,
        sample_ids,
        created_by,
        tracked,
        llm_model,
        llm_model_temp,
        description=exp_description,
    )

    st.success("Experiment details saved with ID: {}".format(experiment_id))

    # mlflow.set_experiment(experiment_name)
    for index, sample_id in enumerate(sample_ids):
        total_samples = len(sample_ids)
        st.write(f"Working on sample {index + 1} of {total_samples}")
        # Run experiment for each prompt
        for index, prompt_id in enumerate(prompt_ids):
            total_prompts = len(prompt_ids)
            st.write(f"Working on prompt {index + 1} of {total_prompts}")

            run_test(
                sample_id, prompt_id, experiment_id, limit, llm_model, llm_model_temp
            )

        st.write("Sample Completed!")
        update_status(experiment_id, "COMPLETE")


def to_prompt_metadata_db(
    prompt_objective,
    lesson_plan_params,
    output_format,
    rating_criteria,
    general_criteria_note,
    rating_instruction,
    prompt_title,
    experiment_description,
    objective_title,
    objective_desc,
    prompt_created_by,
    version,
):
    conn = get_db_connection()
    cur = conn.cursor()

    unique_prompt_details = (
        prompt_objective
        + json.dumps(lesson_plan_params)
        + output_format
        + json.dumps(rating_criteria)
        + general_criteria_note
        + rating_instruction
    )
    prompt_hash = hashlib.sha256(unique_prompt_details.encode("utf-8")).digest()

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
        cur.execute(
            insert_query,
            (
                prompt_objective,
                lesson_plan_params,
                output_format,
                json.dumps(rating_criteria),
                general_criteria_note,
                rating_instruction,
                prompt_hash,
                prompt_title,
                experiment_description,
                objective_title,
                objective_desc,
                prompt_created_by,
                version,
            ),
        )
        conn.commit()

        returned_id = cur.fetchone()[0]
    else:
        return duplicates[0][0]

    return returned_id


def generate_experiment_placeholders(
    model_name, temperature, limit, prompt_count, sample_count, teacher_name
):
    # Placeholder name with LLM model, temperature, prompt and sample counts, limit, and teacher name
    placeholder_name = f"{model_name}-temp:{temperature}-prompts:{prompt_count}-samples:{sample_count}-limit:{limit}-created:{teacher_name}"

    # Placeholder description
    placeholder_description = (
        f"{model_name} Evaluating with temperature {temperature}, "
        f"using {prompt_count} prompts on {sample_count} samples, "
        f"with a limit of {limit} lesson plans per sample. "
        f"Run by {teacher_name}."
    )

    return placeholder_name, placeholder_description
