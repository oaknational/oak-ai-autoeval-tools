""" Database operations to setup PostgreSQL Database for AutoEval.

Functions:

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
- add_objective
- upload_prompt
- add_to_samples
- add_teacher
- insert_lesson_plan
- insert_sample_prompt (add sample prompts for experiments from CSV)
- add_obj_prompt (link objective with prompt)
"""

# Import the required libraries and modules
import os
import csv
import json
import uuid
import hashlib
import psycopg2
import psycopg2.extras

from dotenv import load_dotenv
from utils import get_db_connection

# Enable direct use of UUID objects in PostgreSQL
psycopg2.extras.register_uuid()

# Load environment variables from .env file into Python environment
load_dotenv()


def new_objectives_table():
    """ Create a new table `m_objectives` in the database to store
    objectives.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table `m_objectives`
    cur = conn.cursor()
    cur.execute(
        """ 
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_objectives (
            id UUID DEFAULT 
            uuid_generate_v4() PRIMARY KEY, 
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
            created_by TEXT, title TEXT, 
            description TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def add_objective(created_by, title, description):
    """ Add a new objective into the `m_objectives` table.

    Args:
        created_by (str): The name of the creator of the objective.
        title (str): Title of the objective.
        description (str): Description of the objective.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Insert a new row into `m_objectives` table
    cur = conn.cursor()
    cur.execute(
        """ 
        INSERT INTO m_objectives (created_by, title, description) 
        VALUES (%s, %s, %s);
        """,
        (created_by, title, description),
    )
    conn.commit()
    cur.close()
    conn.close()


def new_prompts_table():
    """ Create a new table `m_prompts` in the database to store prompts.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table `m_prompts`
    cur = conn.cursor()
    cur.execute(
        """ 
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
            version TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def upload_prompt(
    prompt_title, prompt, prompt_hash, output_format, lesson_plan_params, created_by
):
    """ Upload a prompt for experiments into the `m_prompts` table.

    Args:
        prompt_title (str): Title of the prompt.
        prompt (str): The prompt text.
        prompt_hash (str): SHA256 hash of the prompt text.
        output_format (str): Expected output format of the prompt.
        lesson_plan_params (str): Parameters related to the lesson plan.
        created_by (str): The name of the creator of the prompt.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Insert a new row into the `m_prompts` table
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO m_prompts (
            prompt_title, prompt, prompt_hash, output_format, 
            lesson_plan_params, created_by) 
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (
            prompt_title,
            prompt,
            prompt_hash,
            output_format,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


# WHATS THIS STUFF DOING HERE?

prompt_title = "<PROMPT TITLE>"

prompt = """
        <PROMPT>
    """

prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

output_format = "<SCORE OR TRUE/FALSE>"

lesson_plan_params = "lesson_plan"  # Input params for prompt

created_by = "<NAME>"

# END OF STUFF


def new_obj_prompt_table():
    """ Create a new table 'm_objectives_prompts' in the database to 
    link objectives with prompts.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table `m_objectives_prompts`
    cur = conn.cursor()
    cur.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_objectives_prompts (
            objective_id UUID,
            prompt_id UUID
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def add_obj_prompt(objective_id, prompt_id):
    """ Link an objective with a prompt in the 'm_objectives_prompts' 
    table.

    Args:
        objective_id (str): UUID of the objective.
        prompt_id (str): UUID of the prompt.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Insert a new row into the 'm_objectives_prompts' table
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO m_objectives_prompts (objective_id, prompt_id) 
        VALUES (%s, %s);
        """,
        (objective_id, prompt_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def new_samples_table():
    """ Create a new table 'm_samples' in the database to store samples.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table 'm_samples'
    cur = conn.cursor()
    cur.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_samples (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            sample_title TEXT,
            created_by TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def add_to_samples(sample_table, sample_title, created_by):
    """ Insert a new sample into the 'm_samples' table.

    Args:
        sample_table (str): Table related to the sample.
        sample_title (str): Title of the sample.
        created_by (str): The name of the creator of the sample.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Insert a new row into the 'm_samples' table
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO m_samples (sample_table, sample_title, created_by) 
        VALUES (%s, %s, %s);
        """,
        (sample_table, sample_title, created_by),
    )
    conn.commit()
    cur.close()
    conn.close()


def new_experiments_table():
    """ Create a new table 'm_experiments' in the database to store
    experiments.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table 'm_experiments'
    cur = conn.cursor()
    cur.execute(
        """
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
            tracked BOOL DEFAULT TRUE
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def new_results_table():
    """ Create a new table 'm_results' in the database to store results 
    of experiments.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table 'm_results'
    cur = conn.cursor()
    cur.execute(
        """
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
            status TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def new_samples_lessons_table():
    """ Create a new table 'm_sample_lesson_plans' in the database to 
    link samples with lesson plans.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table 'm_sample_lesson_plans'
    cur = conn.cursor()
    cur.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_sample_lesson_plans (
            sample_id UUID,
            lesson_plan_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def new_teachers_table():
    """ Create a new table 'm_teachers' in the database to store 
    teachers' names.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table 'm_teachers'
    cur = conn.cursor()
    cur.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_teachers (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            name TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def add_teacher(name):
    """ Add a new teacher to the 'm_teachers' table if the teacher does 
    not already exist.

    Args:
        name (str): Name of the teacher to be added.

    Returns:
        str: Success or error message indicating whether the teacher was
        added successfully.
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Check if the teacher already exists
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM m_teachers WHERE name = %s;", (name,))
    if cur.fetchone() is not None:
        print("Teacher already exists.")
        cur.close()
        conn.close()
        return "Teacher already exists."

    # Insert a new row into the 'm_teachers' table
    cur.execute("INSERT INTO m_teachers (name) VALUES (%s);", (name,))
    conn.commit()
    cur.close()
    conn.close()
    return "Teacher added successfully."


def new_lesson_plans_table():
    """ Create a new table 'lesson_plans' in the database to store 
    lesson plans.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    # Create a new table 'lesson_plans'
    cur = conn.cursor()
    cur.execute(
        """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS lesson_plans (
            id TEXT,
            lesson_id TEXT,
            json TEXT,
            generation_details TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            key_stage TEXT,
            subject TEXT
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_lesson_plan():
    """ Inserts a sample lesson plan into the 'lesson_plans' table from
    a JSON file.

    Returns:
        None
    """
    # Establish connection to the PostgreSQL database
    conn = get_db_connection()

    with open("data/sample_lesson.json", "r") as file:
        json_data = file.read()

    # Define the values to insert
    id_value = uuid.uuid4()
    lesson_id_value = None
    json_value = json_data
    generation_details_value = "sample lesson plan"
    key_stage_value = "key-stage-1"
    subject_value = "english"

    # Insert a new row into the 'lesson_plans' table
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO lesson_plans (
            id, lesson_id, json, generation_details, created_at, 
            key_stage, subject
        ) 
        VALUES (%s, %s, %s, %s, now(), %s, %s);
        """,
        (
            id_value,
            lesson_id_value,
            json_value,
            generation_details_value,
            key_stage_value,
            subject_value,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_sample_prompt(csv_file_path):
    """Insert prompts into the 'm_prompts' table from a CSV file.

    Args:
        csv_file_path (str): CSV file path containing prompts data.

    Returns:
        None
    """
    # Connect to the PostgreSQL database
    conn = get_db_connection()

    with open(csv_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt_data = json.loads(row["result"])

            # Extract data from the row's JSON object
            id = prompt_data["id"]
            prompt_title = prompt_data["prompt_title"]
            prompt_objective = prompt_data["prompt_objective"]
            output_format = prompt_data["output_format"]
            lesson_plan_params = prompt_data["lesson_plan_params"]
            rating_criteria = prompt_data["rating_criteria"]
            general_criteria_note = prompt_data["general_criteria_note"]
            rating_instruction = prompt_data["rating_instruction"]
            experiment_description = prompt_data["experiment_description"]
            objective_title = prompt_data["objective_title"]
            objective_desc = prompt_data["objective_desc"]
            created_by = prompt_data["created_by"]
            version = prompt_data["version"]

            # Compute the prompt hash
            prompt_hash = hashlib.sha256(prompt_objective.encode()).digest()

            # Insert a new row into m_prompts
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO m_prompts (
                    id, prompt_title, prompt_objective, prompt_hash, 
                    output_format, lesson_plan_params, rating_criteria, 
                    general_criteria_note, rating_instruction, 
                    experiment_description, objective_title, 
                    objective_desc, created_by, version, created_at, 
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, now(), now()
                );
                """,
                (
                    id,
                    prompt_title,
                    prompt_objective,
                    prompt_hash,
                    output_format,
                    lesson_plan_params,
                    rating_criteria,
                    general_criteria_note,
                    rating_instruction,
                    experiment_description,
                    objective_title,
                    objective_desc,
                    created_by,
                    version,
                ),
            )
            conn.commit()
            cur.close()
    conn.close()


if __name__ == "__main__":
    """Initialize the database schema and populate it with data."""
    csv_file_path = "data/sample_prompts.csv"

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
    insert_sample_prompt(csv_file_path)
