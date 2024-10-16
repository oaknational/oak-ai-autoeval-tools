
"""

- get_db_connection: 
    Establishes a connection to the PostgreSQL database.
- execute_single_query:
    Executes a single SQL query.
- execute_multi_query:
    Executes multiple SQL queries.
- new sample:
    Creates a new sample in the database.
- add_lesson_plans_to_sample:
    Links lesson plans to a sample in the database.
- add_lesson_plan_to_sample:
    Links a lesson plan to a sample in the database.
- insert_single_lesson_plan:
    Inserts a single lesson plan into the database.
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
- add_batch:
    Adds details of a new batch submission to the database
- get_batches:
    Retrieves pending batches.
- add_experiment: 
    Adds a new experiment to the database.
- get_prompt: 
    Retrieves prompt details based on a prompt ID.
- add_results: 
    Adds results of an experiment to the database.
- update_status: 
    Updates the status of an experiment in the database.
- update_batch_status: 
    Updates the status of a batch in the database.
- start_experiment: 
    Starts a new experiment, runs tests for each sample and prompt, and
    updates status.
- insert_prompt: 
    Add or update prompt in the database.
- fetch_lesson_plan_json:
    Fetches lesson plan JSON data from the database.
- fetch_prompt_objectives_desc:
    Fetches prompt objectives and descriptions from the database.
- fetch_bad_lesson_plans:
    Fetches lesson plans with lowest scores based on selected prompt IDs.
- fetch_result_data:
    Fetches result data based on lesson plan ID, prompt ID, and result.
- fetch_final_data:
    Fetches final result data based on lesson plan ID, prompt ID, and experiment ID.
- delete_created_sample:
    Deletes a created sample from the database.
- delete_lesson_plans_from_sample_lesson_plans:
    Deletes lesson plans from the sample_lesson_plans table.
    """

import uuid
import hashlib
import psycopg2
import json
from datetime import datetime
import pandas as pd
import streamlit as st
from utils.formatting import fix_json_format
from utils.constants import ErrorMessages
from utils.common_utils import log_message, get_env_variable

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

def new_sample(sample_title, created_by):
    """ Create a new sample and insert it into the m_samples table.

    Args:
        sample_title (str): Title of the sample.
        created_by (str): The name of the creator of the sample.

    Returns:
        str: ID of the created sample if successful, None otherwise.
    """
    query = """
        INSERT INTO public.m_samples (
            id, created_at, updated_at, sample_title, created_by)
        VALUES (gen_random_uuid(), NOW(), NOW(), %s, %s)
        RETURNING id;
    """
    params = (sample_title, created_by)
    result = execute_single_query(query, params)
    if result and result[0]:
        sample_id = result[0][0]
        st.session_state.sample_id = sample_id
        st.info(f"Sample created with ID: {sample_id}")
        return sample_id
    else:
        st.error("Failed to create a new sample.")
        return None


def add_lesson_plans_to_sample(sample_id, lesson_plan_ids):
    """ Link lesson plans to a sample in the m_sample_lesson_plans table.

    Args:
        sample_id (str): ID of the sample.
        lesson_plan_ids (list): List of lesson plan IDs to link.

    Returns:
        bool: True if successful, False otherwise.
    """
    queries = [
        (
            """
            INSERT INTO public.m_sample_lesson_plans (
                sample_id, lesson_plan_id
            )
            VALUES (%s, %s);
            """,
            (sample_id, lesson_plan_id)
        ) for lesson_plan_id in lesson_plan_ids
    ]
    return execute_multi_query(queries)

def add_lesson_plan_to_sample(sample_id, lesson_plan_id):
    """ Link a lesson plan to a sample in the m_sample_lesson_plans table.

    Args:
        sample_id (str): ID of the sample.
        lesson_plan_id (str): ID of the lesson plan.

    Returns:
        bool: True if successful, False otherwise.
    """
    query = """
    INSERT INTO public.m_sample_lesson_plans (
        sample_id, lesson_plan_id, created_at, updated_at
    )
    VALUES (%s, %s, %s, %s);
    """
    now = datetime.now()
    return execute_single_query(query, (sample_id, lesson_plan_id, now, now))



def insert_single_lesson_plan(json_data, lesson_id=None,key_stage=None, subject=None,  generation_details=None):
    try:
        id_value = str(uuid.uuid4())
        lesson_id_value = lesson_id
        json_value = json_data
        generation_details_value = generation_details
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
        if success:
            print("Lesson plan inserted successfully.")  
        else: 
            print("Unexpected error occurred while inserting the lesson plan.")
        return id_value
    except Exception as e:
        log_message("error", f"Unexpected error occurred while inserting the lesson plan: {e}")
        return None


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
        WHERE ex.tracked = true and ex.status = 'COMPLETE'
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
            p.version as prompt_version,
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
    """Retrieve all versions of prompts data from the database.

    Returns:
        pd.DataFrame: DataFrame with all versions of prompts data.
    """
    query = """
        WITH RankedPrompts AS (
            SELECT 
                id, 
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
                version,
                preferred,
                created_by,
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
            general_criteria_note,
            rating_instruction,
            prompt_title, 
            experiment_description, 
            objective_title,
            objective_desc, 
            version,
            preferred,
            created_by,
            created_at,
            row_num
        FROM 
            RankedPrompts;
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


def add_batch(batch_ref, experiment_id, batch_description, created_by, status="PENDING"):
    """ Add a new batch to the database.

    Args:
        xxx
        status (str, optional): Status of the experiment.

    Returns:
        int: ID of the newly added batch.
    """
    insert_query = """
        INSERT INTO m_batches (
            created_at, updated_at, batch_ref, batch_description, experiment_id, created_by, status) 
        VALUES (
            now(), now(), %s, %s, %s, %s, %s)
        RETURNING id;
    """
    params = (batch_ref, batch_description, experiment_id, created_by, status)

    try:
        result = execute_single_query(insert_query, params)
        if result:
            return result[0][0]
        else:
            return None
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return None


def get_batches():
    """ Retrieve batches data from the database.

    Returns:
        pd.DataFrame: DataFrame with batches data.
    """
    query = "SELECT batch_ref, batch_description, created_by, created_at FROM m_batches WHERE status = 'PENDING';"
    return execute_single_query(query, return_dataframe=True)


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
    result = execute_single_query(query, (prompt_id,), return_dataframe=True)

    if not result.empty:
        prompt_data = result.iloc[0]
        clean_rating_criteria = fix_json_format(prompt_data["rating_criteria"])

        return {
            "prompt_id": prompt_data["id"],
            "prompt_objective": prompt_data["prompt_objective"],
            "lesson_plan_params": prompt_data["lesson_plan_params"],
            "output_format": prompt_data["output_format"],
            "rating_criteria": clean_rating_criteria,
            "general_criteria_note": prompt_data["general_criteria_note"],
            "rating_instruction": prompt_data["rating_instruction"],
            "prompt_title": prompt_data["prompt_title"],
            "experiment_description": prompt_data["experiment_description"],
            "objective_title": prompt_data["objective_title"],
            "objective_desc": prompt_data["objective_desc"],
        }
    return None


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


def update_batch_status(experiment_id, status):
    """ Update the status of a batch in the database using experiment_id as the key.

    Args:
        experiment_id (str): Reference identifier for the batch.
        status (str): New status to update.
        
    Returns:
        bool: True if the status was updated successfully, False otherwise.
    """
    query = """
        UPDATE m_batches SET status = %s
        WHERE experiment_id = %s;
    """
    params = (status, experiment_id)

    try:
        success = execute_single_query(query, params)
        if not success:
            log_message("error", "Failed to update status")
            return False
        return True
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return False


def start_experiment(experiment_name, exp_description, sample_ids, created_by,
        prompt_ids, limit, llm_model, tracked, llm_model_temp=0.5, top_p=1):
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
    from utils.inference import run_test

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
                    llm_model_temp, top_p
                )
            st.write(f"Sample {sample_index + 1} Completed!")

        if update_status(experiment_id, "COMPLETE"):
            st.write("Experiment Completed!")
            return experiment_id
        else:
            return False
    except Exception as e:
        log_message("error", f"An error occurred during the experiment: {e}")
        update_status(experiment_id, "FAILED")
        return False


def insert_prompt(prompt_objective, lesson_plan_params, output_format,
        rating_criteria, general_criteria_note, rating_instruction,
        prompt_title, experiment_description, objective_title, objective_desc,
        prompt_created_by, version,preferred):
    """ Add or update prompt in the database.

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
        None.
    """
    insert_query = """
        INSERT INTO m_prompts (
            created_at, updated_at, prompt_objective, 
            lesson_plan_params, output_format, rating_criteria, 
            general_criteria_note, rating_instruction, prompt_hash, 
            prompt_title, experiment_description, objective_title, 
            objective_desc, created_by, version, preferred
        )
        VALUES (
            now(), now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s
        ) 
        RETURNING id;
    """
    params = (
        prompt_objective,
        lesson_plan_params,
        output_format,
        json.dumps(rating_criteria),
        general_criteria_note,
        rating_instruction,
        hashlib.sha256(prompt_objective.encode()).digest(),
        prompt_title,
        experiment_description,
        objective_title,
        objective_desc,
        prompt_created_by,
        version,
        preferred,
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
    
def fetch_lesson_plan_json(lesson_plan_id):
    try:
        conn = get_db_connection()
        query = """
        SELECT json
        FROM public.lesson_plans 
        WHERE id = %s
        """
        df = pd.read_sql_query(query, conn, params=(lesson_plan_id,))
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def fetch_prompt_objectives_desc():
    try:
        conn = get_db_connection()
        query = """
        SELECT id, objective_desc
        FROM public.m_prompts
        WHERE output_format = 'Score'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred while fetching prompt details: {e}")
        return None


def fetch_bad_lesson_plans(selected_prompt_ids):
    try:
        conn = get_db_connection()  
        query = """SELECT 
                    r.prompt_id, 
                    r.lesson_plan_id, 
                    lp.generation_details,
                    p.prompt_title,
                    min(CAST(r.result AS numeric)) AS min_result, 
                    max(CAST(r.result AS numeric)) AS max_result,
                    count(r.justification) AS justification_count,
                    COUNT(CASE WHEN CAST(r.result AS numeric) = 1 THEN 1 END) AS score_1_count,
                    COUNT(CASE WHEN CAST(r.result AS numeric) = 2 THEN 1 END) AS score_2_count,
                    COUNT(CASE WHEN CAST(r.result AS numeric) = 3 THEN 1 END) AS score_3_count,
                    COUNT(CASE WHEN CAST(r.result AS numeric) = 4 THEN 1 END) AS score_4_count, 
                    COUNT(CASE WHEN CAST(r.result AS numeric) = 5 THEN 1 END) AS score_5_count
                FROM public.m_results r
                INNER JOIN m_prompts p ON p.id = r.prompt_id
                INNER JOIN lesson_plans lp ON lp.id = r.lesson_plan_id
                WHERE r.status = 'SUCCESS' AND r.result ~ '^[0-9\\.]+$' AND p.output_format = 'Score' 
                AND p.prompt_title <> 'Answers Are Minimally Different'
                AND r.prompt_id IN %s
                GROUP BY r.lesson_plan_id, r.prompt_id, p.prompt_title, lp.generation_details
                ORDER BY lesson_plan_id DESC, justification_count DESC, max_result ASC;"""
        df = pd.read_sql_query(query, conn, params=(selected_prompt_ids,))
        conn.close()
        return df
    
    except Exception as e:
        print(f"An error occurred while fetching lesson plans: {e}")
        return None
    
def fetch_result_data(lesson_plan_id, prompt_id, result):
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            r.prompt_id, r.lesson_plan_id, r.result, r.justification
        FROM public.m_results r
        WHERE r.lesson_plan_id = %s 
        AND r.prompt_id = %s 
        AND CAST(r.result AS numeric) = %s
        AND r.status = 'SUCCESS'
        ORDER BY r.result ASC
        LIMIT 1
        """
        df = pd.read_sql_query(query, conn, params=(lesson_plan_id, prompt_id, result))
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred while fetching result data: {e}")
        return None


def fetch_final_data(lesson_plan_id, prompt_id, experiment_id):
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            r.result, r.justification
        FROM public.m_results r
        WHERE r.lesson_plan_id = %s 
        AND r.prompt_id = %s 
        AND r.experiment_id = %s
        """
        df = pd.read_sql_query(query, conn, params=(lesson_plan_id, prompt_id, experiment_id))
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def delete_created_sample(sample_id):
    try:
        conn = get_db_connection()
        query = """
        DELETE FROM public.m_samples
        WHERE id = %s
        """
        cur = conn.cursor()
        cur.execute(query, (sample_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"An error occurred while deleting the sample: {e}")
        return False
    
 
def delete_lesson_plans_from_sample_lesson_plans(sample_id):
    try:
        conn = get_db_connection()
        query = """
        DELETE FROM public.m_sample_lesson_plans
        WHERE sample_id = %s
        """
        cur = conn.cursor()
        cur.execute(query, (sample_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"An error occurred while deleting the sample lesson plans: {e}")
        return False
    


def get_lesson_plans_for_dataset(keyword=None):
    """ Retrieve lesson plans from the lesson_plans table based on a 
        keyword filter.

    Args:
        keyword (str, optional): Keyword to filter generation details. 
            Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing lesson plan IDs and 
            generation details.
    """
    query = """
        SELECT lp.id, lp.generation_details
        FROM lesson_plans lp
        WHERE 1=1
    """
    params = []
    if keyword:
        query += " AND lp.generation_details LIKE %s"
        params.append(f"%{keyword}%")

    return execute_single_query(query, params, return_dataframe=True)
