""" Database operations to setup PostgreSQL Database for AutoEval.

Functions:

- initialize_database:
This function initializes the database schema and populates it with data
by calling the functions listed below to create tables and rows.

Create new tables in the database:
- new_objectives_table
- new_prompts_table
- new_samples_table
- new_experiments_table
- new_results_table
- new_teachers_table
- new_lesson_plans_table
- new_obj_prompt_table (link objectives with prompts)
- new_samples_lessons_table (link samples with lesson plans)

Create new rows in tables:
- add_teacher
- insert_lesson_plan
- insert_sample_prompt (add sample prompts for experiments from CSV)
"""

import csv
import json
import uuid
import hashlib

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from utils.common_utils import log_message
from utils.db_scripts import execute_single_query, execute_multi_query
from utils.constants import ErrorMessages


load_dotenv()
psycopg2.extras.register_uuid()


def new_objectives_table():
    """ Create a new table `m_objectives` in the database to store
    objectives.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_objectives (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            created_by TEXT, title TEXT,
            description TEXT);
    """
    execute_single_query(query)


def new_prompts_table():
    """ Create a new table `m_prompts` in the database to store prompts.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_prompts (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            prompt_objective TEXT,
            lesson_plan_params TEXT,
            output_format TEXT,
            rating_criteria TEXT,
            general_criteria_note TEXT,
            rating_instruction TEXT,
            prompt_hash bytea,
            prompt_title TEXT,
            experiment_description TEXT,
            objective_title TEXT,
            objective_desc TEXT,
            created_by TEXT,
            version TEXT);
    """
    execute_single_query(query)


def new_obj_prompt_table():
    """ Create a new table 'm_objectives_prompts' in the database to 
    link objectives with prompts.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_objectives_prompts (
            objective_id UUID,
            prompt_id UUID);
    """
    execute_single_query(query)


def new_samples_table():
    """ Create a new table 'm_samples' in the database to store samples.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_samples (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            sample_title TEXT,
            created_by TEXT);
    """
    execute_single_query(query)


def new_experiments_table():
    """ Create a new table 'm_experiments' in the database to store
    experiments.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_experiments (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            experiment_name TEXT,
            objective_id UUID,
            sample_id TEXT,
            llm_model TEXT,
            llm_model_temp FLOAT,
            llm_max_tok INT,
            description TEXT,
            created_by TEXT,
            status TEXT,
            tracked BOOL DEFAULT TRUE);
    """
    execute_single_query(query)


def new_results_table():
    """ Create a new table 'm_results' in the database to store results 
    of experiments.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_results (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            experiment_id UUID,
            prompt_id UUID,
            lesson_plan_id TEXT,
            result TEXT,
            justification TEXT,
            status TEXT);
    """
    execute_single_query(query)


def new_samples_lessons_table():
    """ Create a new table 'm_sample_lesson_plans' in the database to 
    link samples with lesson plans.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_sample_lesson_plans (
            sample_id UUID,
            lesson_plan_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now());
    """
    execute_single_query(query)


def new_teachers_table():
    """ Create a new table 'm_teachers' in the database to store 
    teachers' names.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_teachers (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            name TEXT);
    """
    execute_single_query(query)


def add_teacher(name):
    """ Add a new teacher to the 'm_teachers' table if the teacher does 
    not already exist.

    Args:
        name (str): Name of the teacher to be added.

    Returns:
        str: Success or error message indicating whether the teacher was
        added successfully.
    """
    select_query = """
        SELECT 1 FROM m_teachers WHERE name = %s;
    """
    if execute_single_query(select_query, (name,)):
        return "Teacher already exists."

    insert_query = """
        INSERT INTO m_teachers (name) VALUES (%s);
    """
    execute_single_query(insert_query, (name,))
    return "Teacher added successfully."


def new_lesson_plans_table():
    """ Create a new table 'lesson_plans' in the database to store 
    lesson plans.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS lesson_plans (
            id TEXT,
            lesson_id TEXT,
            json TEXT,
            generation_details TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            key_stage TEXT,
            subject TEXT);
    """
    execute_single_query(query)


def insert_lesson_plan():
    """ Inserts a sample lesson plan into the 'lesson_plans' table from
    a JSON file.

    Returns:
        str: Success message or error message indicating the result of the 
        operation.
    """
    try:
        with open("data/sample_lesson.json", "r", encoding="utf-8") as file:
            json_data = file.read()

        id_value = uuid.uuid4()
        lesson_id_value = None
        json_value = json_data
        generation_details_value = "sample lesson plan"
        key_stage_value = "key-stage-1"
        subject_value = "english"

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

        success = execute_single_query([(query, params)])
        return (
            "Lesson plan inserted successfully." if success else 
            ErrorMessages.UNEXPECTED_ERROR
        )
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return ErrorMessages.UNEXPECTED_ERROR


def insert_sample_prompt(csv_file_path):
    """Insert prompts into the 'm_prompts' table from a CSV file.

    Args:
        csv_file_path (str): CSV file path containing prompts data.

    Returns:
        str: Success message or error message indicating the result of the 
        operation.
    """
    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            queries_and_params = []

            for row in reader:
                prompt_data = json.loads(row["result"])

                prompt_hash = hashlib.sha256(
                    prompt_data["prompt_objective"].encode()
                ).digest()

                query = """
                    INSERT INTO m_prompts (
                        id, prompt_title, prompt_objective,
                        prompt_hash, output_format, lesson_plan_params,
                        rating_criteria, general_criteria_note,
                        rating_instruction, experiment_description,
                        objective_title, objective_desc, created_by,
                        version, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, now(), now());
                """
                params = (
                    prompt_data["id"],
                    prompt_data["prompt_title"],
                    prompt_data["prompt_objective"],
                    prompt_hash,
                    prompt_data["output_format"],
                    prompt_data["lesson_plan_params"],
                    prompt_data["rating_criteria"],
                    prompt_data["general_criteria_note"],
                    prompt_data["rating_instruction"],
                    prompt_data["experiment_description"],
                    prompt_data["objective_title"],
                    prompt_data["objective_desc"],
                    prompt_data["created_by"],
                    prompt_data["version"]
                )

                queries_and_params.append((query, params))

            success = execute_multi_query(queries_and_params)
            return (
                "Sample prompts inserted successfully." if success else 
                ErrorMessages.UNEXPECTED_ERROR
            )
    except Exception as e:
        log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")
        return ErrorMessages.UNEXPECTED_ERROR

def new_lesson_sets_table(csv_file_path):
    """ Create a new table 'lesson_plan_sets' in the database and insert CSV data.

    Args:
        csv_file_path (str): Path to the CSV file containing lesson plan sets.
    """
    # Create table query
    create_table_query = """
        CREATE TABLE IF NOT EXISTS lesson_plan_sets (
        lesson_number TEXT,
        subject VARCHAR(50),
        key_stage VARCHAR(10),
        lesson_title TEXT
    );
    """
    # Execute create table query
    execute_single_query(create_table_query)

    # Read CSV and insert data
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            insert_query = """
                INSERT INTO lesson_plan_sets (lesson_number, subject, key_stage, lesson_title) 
                VALUES (%s, %s, %s, %s);
            """
            execute_single_query(insert_query, tuple(row))
            

def initialize_database(csv_file_path):
    """Initialize the database schema and populate it with data."""
    
    sample_lesson_set_path = csv_file_path + "sample_lesson_set.csv"
    sample_prompts_path = csv_file_path + "sample_prompts.csv"
    new_experiments_table()
    new_results_table()
    new_prompts_table()
    new_objectives_table()
    new_obj_prompt_table()
    new_samples_table()
    new_samples_lessons_table()
    new_teachers_table()
    new_lesson_plans_table()
    insert_lesson_plan()
    add_teacher("John Doe")
    insert_sample_prompt(sample_prompts_path)
    new_lesson_sets_table(sample_lesson_set_path)


if __name__ == "__main__":
    initialize_database("data/")
