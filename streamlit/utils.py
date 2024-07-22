""" Utility and helper functions for managing database operations, 
processing data, rendering templates, and running experiments.

This module provides the following functions:

- get_env_variable:
    Fetch environment variables with a fallback mechanism.
- log_message: 
    Logs messages with different severity levels.
- clear_all_caches:
    Clears the cache for Streamlit.
- get_db_connection: 
    Establishes a connection to the PostgreSQL database.
- execute_single_query:
    Executes a single SQL query.
- execute_multi_query:
    Executes multiple SQL queries.
- convert_to_json:
    Converts text to JSON format.
- json_to_html: 
    Converts a JSON object to an HTML-formatted string.
- get_light_experiment_data: 
    Retrieves light experiment data from the database.
- get_full_experiment_data: 
    Retrieves full data for a selected experiment ID.
- get_prompts: 
    Retrieves prompts data from the database.
- get_samples: 
    Retrieves samples data from the database.
- get_teachers: 
    Retrieves teachers data from the database.
- get_lesson_plans_by_id: 
    Retrieves lesson plans based on a sample ID.
- add_experiment: 
    Adds a new experiment to the database.
- fix_json_format: 
    Fixes JSON formatting issues in a given JSON string.
- get_prompt: 
    Retrieves prompt details based on a prompt ID.
- process_prompt: 
    Processes prompt details, ensuring correct formatting.
- render_prompt: 
    Renders a prompt template using lesson plan and prompt details.
- clean_response:
    Cleans JSON response by removing extraneous characters and decoding 
    the JSON content.
- run_inference: 
    Runs inference using a lesson plan and a prompt ID.
- add_results: 
    Adds results of an experiment to the database.
- decode_lesson_json:
    Decodes JSON string and logs errors if any.
- handle_inference:
    Runs inference and adds results to the database.
- run_test: 
    Runs a test for each lesson plan associated with a sample and adds
    results to the database.
- update_status: 
    Updates the status of an experiment in the database.
- start_experiment: 
    Starts a new experiment, runs tests for each sample and prompt, and
    updates status.
- to_prompt_metadata_db: 
    Adds or retrieves prompt metadata in the database.
- generate_experiment_placeholders: 
    Generates placeholders for an experiment based on specified parameters.
"""

# Import the required libraries and modules
import os
import re
import json
import hashlib

import pandas as pd
import psycopg2
from jinja2 import Environment, FileSystemLoader, select_autoescape
import openai
from openai import OpenAI
import streamlit as st

from constants import ErrorMessages


def get_env_variable(var_name, default_value=None):
    """ Retrieve the value of an environment variable with an optional 
    default, or raises an error.

    Args:
        var_name (str): The name of the environment variable to retrieve.
            default_value (any, optional): The value to return if the 
            environment variable is not set. Defaults to None.

    Returns:
        any: The value of the environment variable, or the default value 
            if the environment variable is not set.

    Raises:
        EnvironmentError: If the environment variable is not set and no 
            default value is provided.
    """
    value = os.getenv(var_name, default_value)
    if value is None:
        log_message("error", f"Environment variable {var_name} not found")
        if default_value is None:
            raise EnvironmentError(
                f"Missing mandatory environment variable: {var_name}"
            )
    return value


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


def clear_all_caches():
    """ Clear all caches in the application."""
    st.cache_data.clear()
    st.cache_resource.clear()


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


def execute_single_query(query, params=None, return_dataframe=False):
    """ Execute a given SQL query using a PostgreSQL database connection.

    This function handles the connection, cursor management, and error
    handling for executing a SQL query. It supports both parameterized
    queries and returning results as DataFrames for SELECT queries.

    Args:
        query (str): The SQL query to be executed.
        params (tuple, optional): A tuple of parameters to be passed to the
            SQL query. Defaults to None if no parameters are required.
        return_dataframe (bool, optional): If True, returns results as a
            DataFrame. Defaults to False.

    Returns:
        pd.DataFrame or bool: If return_dataframe is True, returns a
            DataFrame containing the query results. If return_dataframe is
            False, returns True if the query was executed and committed
            successfully, or False if an error occurred.
    """
    conn = get_db_connection()
    if not conn:
        log_message("error", "Failed to establish database connection")
        return pd.DataFrame() if return_dataframe else False

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                if return_dataframe:
                    return pd.DataFrame(
                        cur.fetchall(),
                        columns=[desc[0] for desc in cur.description]
                    )
                elif cur.description:
                    return cur.fetchall()
                return True

    except (psycopg2.DatabaseError) as db_err:
        log_message("error", f"Error executing query: {db_err}")
        conn.rollback()
    except Exception as e:
        log_message("error", f"Unexpected error executing query: {e}")
        conn.rollback()
    finally:
        conn.close()
        
    return pd.DataFrame() if return_dataframe else False


def execute_multi_query(queries_and_params, return_results=False):
    """ Execute a list of SQL queries using a PostgreSQL database 
    connection with rollback on failure.

    This function handles the connection, cursor management, and error
    handling for executing a list of SQL queries. It supports both 
    parameterized queries and fetching results from SELECT queries.

    Args:
        queries_and_params (list of tuples): A list where each element 
            is a tuple containing the SQL query as a string and the 
            parameters as a tuple.
        return_results (bool, optional): If True, returns the results 
            of the SELECT queries. Defaults to False.

    Returns:
        list or bool: Returns a list of results if return_results is 
            True, otherwise True if all queries were executed and 
            committed successfully, or False if an error occurred.

    Raises:
        psycopg2.Error: If a database error occurs during the execution 
            of the query.
        Exception: If any other unexpected error occurs during the 
            execution of the query.
    """
    results = []
    for query, params in queries_and_params:
        result = execute_single_query(
            query, params, return_dataframe=return_results
        )
        if return_results:
            results.append(result)
    return results if return_results else True


def convert_to_json(text):
    """
    Convert text to JSON format.

    If the text is already in JSON format, it is returned as a dictionary. 
    If the text is not in JSON format or an error occurs during parsing, 
    the text is converted to a JSON object with the text stored under the 
    key 'text'.

    Args:
        text (str): The input text to be converted to JSON.

    Returns:
        dict: A dictionary representing the JSON object. If the input 
            text is valid JSON, it returns the parsed JSON. If the input 
            is not valid JSON, it returns a dictionary with the original 
            text under the key 'text'. If the input is NaN, it returns 
            None.
    """
    if pd.isna(text):
        return None
    try:
        json_data = json.loads(text)
    except json.JSONDecodeError:
        json_data = {"text": text}
    except TypeError as e:
        st.error(f"TypeError: {e} - Value: {text}")
        json_data = {"text": str(text)}
    return json_data


def json_to_html(json_obj, indent=0):
    """ Convert a JSON object to an HTML-formatted string recursively.

    Args:
        json_obj (dict or list): JSON object to convert.
        indent (int): Current level of indentation for formatting.

    Returns:
        str: HTML-formatted string representing the JSON object.
    """
    def dict_to_html(d, indent):
        """Convert a dictionary to an HTML-formatted string."""
        if not d:
            return f"{get_indent(indent)}{{}}"
        html = f"{get_indent(indent)}{{<br>"
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            html += f"{get_indent(indent + 1)}<strong>{key}</strong>: "
            html += convert_to_html(value, indent + 1)
            if i < len(items) - 1:
                html += ","
            html += "<br>" if i < len(items) - 1 else ""
        html += f"{get_indent(indent)}}}"
        return html

    def list_to_html(lst, indent):
        """Convert a list to an HTML-formatted string."""
        if not lst:
            return f"{get_indent(indent)}[]"
        html = f"{get_indent(indent)}[<br>"
        for i, item in enumerate(lst):
            html += convert_to_html(item, indent + 1)
            if i < len(lst) - 1:
                html += ","
            html += "<br>" if i < len(lst) - 1 else ""
        html += f"{get_indent(indent)}]"
        return html

    def get_indent(indent):
        """Return a string of HTML spaces for indentation."""
        return "&nbsp;&nbsp;" * indent

    def convert_to_html(obj, indent):
        """Convert a JSON object to an HTML-formatted string."""
        if isinstance(obj, dict):
            return dict_to_html(obj, indent)
        elif isinstance(obj, list):
            return list_to_html(obj, indent)
        else:
            return f"{get_indent(indent)}{obj}"

    return convert_to_html(json_obj, indent)


def get_light_experiment_data():
    """ Retrieve light experiment data from the database.

    Returns:
        pd.DataFrame: DataFrame with light experiment data.
    """
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
        ORDER by ex.created_at DESC;
    """
    return execute_single_query(query_light, return_dataframe=True)


def get_full_experiment_data(selected_experiment_id):
    """ Retrieve full data for a selected experiment ID from the database.

    Args:
        selected_experiment_id (str): ID of experiment to fetch data for.

    Returns:
        pd.DataFrame: DataFrame with full experiment data.
    """
    query_full = """
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
            ex.id = %s 
            AND ex.tracked = true;
    """
    return execute_single_query(
        query_full, (selected_experiment_id,), return_dataframe=True
    )


def get_prompts():
    """ Retrieve prompts data from the database.

    Returns:
        pd.DataFrame: DataFrame with prompts data.
    """
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
    return execute_single_query(query, return_dataframe=True)


def get_samples():
    """ Retrieve samples data from the database.

    Returns:
        pd.DataFrame: DataFrame with samples data.
    """
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
    return execute_single_query(query, return_dataframe=True)


def get_teachers():
    """ Retrieve teachers data from the database.

    Returns:
        pd.DataFrame: DataFrame with teachers data.
    """
    query = "SELECT id, name FROM m_teachers;"
    return execute_single_query(query, return_dataframe=True)


def get_lesson_plans_by_id(sample_id, limit=None):
    """ Retrieve lesson plans based on a sample ID.

    Args:
        sample_id (str): ID of the sample.
        limit (int, optional): Maximum number of records to fetch.

    Returns:
        list: List of tuples representing lesson plans.
    """
    query = """
        SELECT lp.id, lp.lesson_id, lp.json 
        FROM lesson_plans lp
        JOIN m_sample_lesson_plans slp 
        ON lp.id::text = slp.lesson_plan_id::text
        WHERE slp.sample_id::text = %s
    """
    params = [str(sample_id)]
    if limit:
        query += " LIMIT %s"
        params.append(int(limit))

    return execute_single_query(query, params)


def add_experiment(experiment_name, sample_ids, created_by, tracked,
        llm_model="gpt-4", llm_model_temp=0.5, description="None",
        status="PENDING"):
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
    sample_ids_str = ",".join(map(str, sample_ids))
    insert_query = """
        INSERT INTO m_experiments (
            created_at, updated_at, experiment_name, sample_id, llm_model, 
            llm_model_temp, description, created_by, status, tracked) 
        VALUES (
            now(), now(), %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    params = (
        experiment_name, sample_ids_str, llm_model, llm_model_temp,
        description, created_by, status, tracked
    )

    try:
        result = execute_single_query(insert_query, params)
        if result:
            return result[0][0]
        else:
            return None
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return None


def fix_json_format(json_string):
    """ Fix JSON formatting issues in a given JSON string.

    Args:
        json_string (str): JSON string to fix.

    Returns:
        str: Fixed JSON string or an empty JSON object if fixing fails.
    """
    try:
        json.loads(json_string)
        return json_string
    except ValueError:
        pass

    json_string = json_string.replace('\\\\\\"', '"')
    json_string = json_string.replace("'", '"')
    json_string = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', json_string)
    try:
        json.loads(json_string)
        return json_string
    except ValueError:
        return "{}"


def get_prompt(prompt_id):
    """ Retrieve prompt details based on a prompt ID.

    Args:
        Prompt_id (str): ID of the prompt.

    Returns:
        dict: Dictionary containing prompt details, 
            or None if prompt is not found.
    """
    query = """
        SELECT id, prompt_objective, lesson_plan_params, output_format, 
            rating_criteria, general_criteria_note, rating_instruction, 
            prompt_title, experiment_description, objective_title, 
            objective_desc
        FROM m_prompts
        WHERE id = %s;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (prompt_id,))
            result = cur.fetchone()

    if result:
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
        }
    return None


def process_prompt(prompt_details):
    """ Process prompt details, ensuring correct formatting.

    Args:
        prompt_details (dict): Dictionary containing prompt details.

    Returns:
        dict: Processed prompt details.
    """
    if isinstance(prompt_details.get("rating_criteria"), str):
        try:
            cleaned_criteria = (
                prompt_details["rating_criteria"].replace('\\"', '"')
            )
            prompt_details["rating_criteria"] = json.loads(cleaned_criteria)
        except json.JSONDecodeError as e:
            log_message("error", f"Error decoding JSON: {e}")
            prompt_details["rating_criteria"] = {}

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
    """ Render a prompt template using lesson plan and prompt details.

    Args:
        lesson_plan (dict): Dictionary containing lesson plan details.
        prompt_details (dict): Dictionary containing prompt details.

    Returns:
        str: Rendered prompt template or error message if template 
            cannot be loaded.
    """
    jinja_path = get_env_variable('JINJA_TEMPLATE_PATH')
    jinja_env = Environment(
        loader=FileSystemLoader(jinja_path),
        autoescape=select_autoescape(["html", "xml"]),
    )

    template = jinja_env.get_template("prompt.jinja")
    if not template:
        return "Template could not be loaded."

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


def clean_response(response_text):
    """ Clean and process a JSON response text by removing extraneous 
    characters and decoding the JSON content.

    The function strips the input text, removes enclosing triple 
    backticks if they exist, and then removes any newline, 
    carriage return, tab, and backslash characters. It attempts to 
    decode the cleaned text into a JSON object. If successful, it 
    returns the decoded JSON object and a success status. If a JSON 
    decoding error occurs, it identifies the error position, extracts a 
    snippet around the problematic area, and returns an error message 
    and failure status.

    Args:
        response_text (str): The raw response text to be cleaned and 
            decoded.

    Returns:
        Tuple[Union[Dict, Any], str]: A tuple containing the cleaned and 
            decoded JSON object or an error message in case of failure, 
            and a status string ("SUCCESS" or "FAILURE").
    """
    try:
        raw_content = response_text.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:].strip()
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3].strip()
        cleaned_content = re.sub(r"[\n\r\t\\]", "", raw_content)
        return json.loads(cleaned_content), "SUCCESS"
    except json.JSONDecodeError as e:
        error_position = e.pos
        start_snippet = max(0, error_position - 40)
        end_snippet = min(len(json_str), error_position + 40)
        snippet = response_text[start_snippet:end_snippet]
        return {
            "result": None,
            "justification": (
                f"{ErrorMessages.UNEXPECTED_ERROR}: {e}. "
                f"Problematic snippet: {repr(snippet)}"
            )
        }, "FAILURE"


def run_inference(lesson_plan, prompt_id, llm_model, llm_model_temp,
        timeout=15):
    """ Run inference using a lesson plan and a prompt ID.

    Args:
        lesson_plan (dict): Dictionary containing lesson plan details.
        prompt_id (str): ID of the prompt to use.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for the LLM.
        timeout (int, optional): Timeout duration for inference.

    Returns:
        dict: Inference result or error response.
    """
    required_keys = ["title", "topic", "subject", "keyStage"]
    if not all(k in lesson_plan for k in required_keys):
        return {
            "response": {
                "result": None, 
                "justification": "Lesson data is missing for this check."
            },
            "status": "ABORTED",
        }

    prompt_details = get_prompt(prompt_id)
    if not prompt_details:
        return {
            "response": {
                "result": None,
                "justification": "Prompt details not found for the given ID."
            },
            "status": "ABORTED",
        }

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

    openai.api_key = get_env_variable("OPENAI_API_KEY")
    client = OpenAI()
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

        cleaned_content, status = clean_response(
            response.choices[0].message.content
        )
        return {
            "response": cleaned_content,
            "status": status,
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


def add_results(experiment_id, prompt_id, lesson_plan_id, score,
        justification, status):
    """ Add results of an experiment to the database.

    Args:
        experiment_id (int): ID of the experiment.
        prompt_id (str): ID of the prompt used.
        lesson_plan_id (str): ID of the lesson plan.
        score (float or str): Score or boolean as float (1.0 or 0.0).
        justification (str): Justification for the result.
        status (str): Status of the result (e.g., 'COMPLETE', 'ABORTED').
        
    Returns:
        None
    """
    try:
        if score is not None and score != "":
            try:
                score = float(score)
            except ValueError:
                score_lower = score.lower()
                if score_lower == "true":
                    score = 1.0
                elif score_lower == "false":
                    score = 0.0
        else:
            log_message("error", f"Invalid score: {score}")
            return

        insert_query = """
            INSERT INTO m_results (
                created_at, updated_at, experiment_id, prompt_id, 
                lesson_plan_id, result, justification, status)
            VALUES (now(), now(), %s, %s, %s, %s, %s, %s);
        """
        params = (
            experiment_id, prompt_id, lesson_plan_id, score, justification,
            status
        )

        success = execute_multi_query([(insert_query, params)])
        if not success:
            log_message("error", "Failed to insert result")
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")


def decode_lesson_json(lesson_json_str, lesson_plan_id, lesson_id, index):
    """Decode JSON string and log errors if any.

    Args:
        lesson_json_str (str): JSON string of the lesson.
        lesson_plan_id (str): ID of the lesson plan.
        lesson_id (str): ID of the lesson.
        index (int): Index of the lesson in the list.

    Returns:
        dict: Decoded JSON content or None if decoding fails.
    """
    if not lesson_json_str:
        log_message("error", f"Lesson JSON is None for lesson index {index}")
        return None
    
    try:
        return json.loads(lesson_json_str)
    except json.JSONDecodeError as e:
        error_position = e.pos
        start_snippet = max(0, error_position - 40)
        end_snippet = min(len(lesson_json_str), error_position + 40)
        snippet = lesson_json_str[start_snippet:end_snippet]
        log_message("error", f"Error decoding JSON for lesson index {index}:")
        log_message("error", f"Lesson Plan ID: {lesson_plan_id}")
        log_message("error", f"Lesson ID: {lesson_id}")
        log_message("error", f"Error Message: {e}")
        log_message("error", f"Problematic snippet: {repr(snippet)}")
        return None


def handle_inference(content, prompt_id, llm_model, llm_model_temp, timeout,
        experiment_id, lesson_plan_id):
    """Run inference and add results to the database.

    Args:
        content (dict): Content to run inference on.
        prompt_id (str): ID of the prompt.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for LLM.
        timeout (int): Timeout duration for inference.
        experiment_id (int): ID of the experiment.
        lesson_plan_id (str): ID of the lesson plan.

    Returns:
        dict: Inference output.
    """
    try:
        output = run_inference(
            content, prompt_id, llm_model, llm_model_temp, timeout=timeout
        )
        response = output.get("response")

        if "status" not in output:
            log_message("error", f"Key 'status' missing in output: {output}")
            return None

        if isinstance(response, dict) and all(
            isinstance(v, dict) for v in response.values()
        ):
            for _, cycle_data in response.items():
                result = cycle_data.get("result")
                justification = cycle_data.get(
                    "justification", "").replace("'", "")
                add_results(
                    experiment_id, prompt_id, lesson_plan_id, result, 
                    justification, output["status"]
                )
        else:
            result = response.get("result")
            justification = response.get("justification", "").replace("'", "")
            add_results(
                experiment_id, prompt_id, lesson_plan_id, result, 
                justification, output["status"]
            )
        return output

    except KeyError as e:
        log_message("error", f"KeyError: Missing key in output: {e}")
        log_message("error", f"Output structure: {output}")
        return None

    except Exception as e:
        log_message("error", f"Unexpected error when adding results: {e}")
        log_message(
            "error",
            f"""
            Lesson Plan ID: {lesson_plan_id}, 
            Prompt ID: {prompt_id}, 
            Output: {output}
            """
        )
        return None
    

def run_test(sample_id, prompt_id, experiment_id, limit, llm_model,
        llm_model_temp, timeout=15):
    """ Run a test for each lesson plan associated with a sample and add 
    results to the database.

    Args:
        sample_id (str): ID of the sample.
        prompt_id (str): ID of the prompt.
        experiment_id (int): ID of the experiment.
        limit (int): Maximum number of records to fetch.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for LLM.
        timeout (int, optional): Timeout duration for inference.

    Returns:
        None
    """
    lesson_plans = get_lesson_plans_by_id(sample_id, limit)
    total_lessons = len(lesson_plans)
    log_message("info", f"Total lessons: {total_lessons}")
    
    progress = st.progress(0)
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    for i, lesson in enumerate(lesson_plans):
        lesson_plan_id = lesson[0]
        lesson_id = lesson[1]
        lesson_json_str = lesson[2]

        content = decode_lesson_json(lesson_json_str, lesson_plan_id, lesson_id, i)
        if content is None:
            continue

        output = handle_inference(content, prompt_id, llm_model, llm_model_temp, timeout, experiment_id, lesson_plan_id)
        if output is None:
            continue

        response = output.get("response")
        with placeholder1.container():
            st.write(f'Inference Status: {output["status"]}')
        with placeholder2.container():
            st.write(response)
            log_message(
                "info",
                f"""
                result = {output.get('response')},
                status = {output.get('status')},
                lesson_plan_id = {lesson_plan_id},
                experiment_id = {experiment_id},
                prompt_id = {prompt_id}
                """
            )

        progress.progress((i + 1) / total_lessons)

    placeholder1.empty()
    placeholder2.empty()


def update_status(experiment_id, status):
    """ Update the status of an experiment in the database.

    Args:
        experiment_id (int): ID of the experiment.
        status (str): New status to update.
        
    Returns:
        bool: True if the status was updated successfully, False otherwise.
    """
    query = """
        UPDATE m_experiments SET status = %s
        WHERE id = %s;
    """
    params = (status, experiment_id)

    try:
        success = execute_multi_query([(query, params)])
        if not success:
            log_message("error", "Failed to update status")
            return False
        return True
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return False


def start_experiment(experiment_name, exp_description, sample_ids, created_by,
        prompt_ids, limit, llm_model, tracked, llm_model_temp=0.5):
    """ Start a new experiment, run tests for each sample and prompt, 
    and update status.

    Args:
        experiment_name (str): Name of the experiment.
        exp_description (str): Description of the experiment.
        sample_ids (list): List of sample IDs.
        created_by (str): Name of the creator of the experiment.
        prompt_ids (list): List of prompt IDs to use in the experiment.
        limit (int): Maximum number of records to fetch.
        llm_model (str): Name of the LLM model.
        tracked (bool): Flag indicating whether the experiment is tracked.
        llm_model_temp (float, optional): Temperature parameter for LLM.

    Returns:
        bool: True if the experiment completes successfully, False otherwise.
    """
    experiment_id = add_experiment(
        experiment_name, sample_ids, created_by, tracked, llm_model,
        llm_model_temp, description=exp_description
    )

    if not experiment_id:
        log_message("error", "Failed to create experiment")
        return False

    st.success(f"Experiment details saved with ID: {experiment_id}")

    total_samples = len(sample_ids)
    total_prompts = len(prompt_ids)

    try:
        for sample_index, sample_id in enumerate(sample_ids):
            st.write(
                f"Working on sample {sample_index + 1} of {total_samples}"
            )

            for prompt_index, prompt_id in enumerate(prompt_ids):
                st.write(
                    f"Working on prompt {prompt_index + 1} of {total_prompts}"
                )
                run_test(
                    sample_id, prompt_id, experiment_id, limit, llm_model,
                    llm_model_temp
                )
            st.write(f"Sample {sample_index + 1} Completed!")

        if update_status(experiment_id, "COMPLETE"):
            st.write("Experiment Completed!")
            return True
        else:
            return False
    except Exception as e:
        log_message("error", f"An error occurred during the experiment: {e}")
        update_status(experiment_id, "FAILED")
        return False


def to_prompt_metadata_db(prompt_objective, lesson_plan_params, output_format,
        rating_criteria, general_criteria_note, rating_instruction,
        prompt_title, experiment_description, objective_title, objective_desc,
        prompt_created_by, version,):
    """ Add or retrieve prompt metadata in the database.

    Args:
        prompt_objective (str): Objective of the prompt.
        lesson_plan_params (dict): Parameters related to lesson plans.
        output_format (str): Output format specification.
        rating_criteria (dict): Criteria used for rating.
        general_criteria_note (str): General note on criteria.
        rating_instruction (str): Instructions for rating.
        prompt_title (str): Title of the prompt.
        experiment_description (str): Description of the experiment.
        objective_title (str): Title of the objective.
        objective_desc (str): Description of the objective.
        prompt_created_by (str): Creator of the prompt.
        version (str): Version of the prompt metadata.

    Returns:
        int: ID of the prompt metadata in the database.
    """
    unique_prompt_details = (
        prompt_objective
        + json.dumps(lesson_plan_params)
        + output_format
        + json.dumps(rating_criteria)
        + general_criteria_note
        + rating_instruction
    )
    prompt_hash = hashlib.sha256(
        unique_prompt_details.encode("utf-8")
    ).digest()

    duplicates_check = "SELECT id FROM m_prompts WHERE prompt_hash = %s;"

    try:
        results = execute_multi_query(
            [(duplicates_check, (psycopg2.Binary(prompt_hash),))],
            return_results=True
        )
        duplicates = results[0] if results else []

        if len(duplicates) == 0:
            insert_query = """
                INSERT INTO m_prompts (
                    created_at, updated_at, prompt_objective, 
                    lesson_plan_params, output_format, rating_criteria, 
                    general_criteria_note, rating_instruction, prompt_hash, 
                    prompt_title, experiment_description, objective_title, 
                    objective_desc, created_by, version
                )
                VALUES (
                    now(), now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s
                ) 
                RETURNING id;
            """
            params = (
                prompt_objective, lesson_plan_params, output_format,
                json.dumps(rating_criteria), general_criteria_note,
                rating_instruction, prompt_hash, prompt_title,
                experiment_description, objective_title, objective_desc,
                prompt_created_by, version
            )
            results = execute_multi_query(
                [(insert_query, params)], return_results=True
            )
            return results[0][0][0] if results else None
        return duplicates[0][0]

    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return None


def generate_experiment_placeholders(model_name, temperature, limit,
        prompt_count, sample_count, teacher_name):
    """ Generate placeholders for an experiment based on specified parameters.

    Args:
        model_name (str): Name of the LLM model.
        temperature (float): Temperature parameter for the LLM.
        limit (int): Limit of lesson plans per sample.
        prompt_count (int): Number of prompts used in the experiment.
        sample_count (int): Number of samples in the experiment.
        teacher_name (str): Name of the teacher who initiated the experiment.

    Returns:
        tuple: placeholder name and description formatted as strings.
    """
    placeholder_name = (
        f"{model_name}-temp:{temperature}-prompts:{prompt_count}-samples:"
        f"{sample_count}-limit:{limit}-created:{teacher_name}"
    )
    placeholder_description = (
        f"{model_name} Evaluating with temperature {temperature}, using "
        f"{prompt_count} prompts on {sample_count} samples, with a limit of "
        f"{limit} lesson plans per sample. Run by {teacher_name}."
    )
    return placeholder_name, placeholder_description
