import psycopg2
import psycopg2.extras
import os 
from dotenv import load_dotenv
import hashlib
import json
import uuid
import psycopg2.extras
import csv

# call it in any place of your program
# before working with UUID objects in PostgreSQL
psycopg2.extras.register_uuid()

load_dotenv()

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

def new_objectives_table():
    conn = get_db_connection()
    # Create a new table
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_objectives (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            created_by TEXT,
            title TEXT,
            description TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close() 


def add_objective(created_by, title, description):
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO m_objectives (created_by, title, description) VALUES ('{created_by}', '{title}', '{description}');
    """)

    conn.commit()
    cur.close()
    conn.close()


def new_prompts_table():
    conn = get_db_connection()

    # Create a new table
    cur = conn.cursor()
    cur.execute("""
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
    """)
    conn.commit()
    cur.close()
    conn.close()

def upload_prompt(prompt_title, prompt, prompt_hash, output_format, lesson_plan_params, created_by):
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO m_prompts (prompt_title, prompt, prompt_hash, output_format, lesson_plan_params, created_by) VALUES ('{prompt_title}', '{prompt.replace("'", "")}', '{prompt_hash}', '{output_format}', '{lesson_plan_params}', '{created_by}');
    """)

    conn.commit()
    cur.close()
    conn.close()


prompt_title = "<PROMPT TITLE>" 

prompt = """
        <PROMPT>
    """

prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

output_format = "<SCORE OR TRUE/FALSE>"

lesson_plan_params = "lesson_plan" # Input params for prompt

created_by = "<NAME>"


def new_obj_prompt_table(): 
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_objectives_prompts (
            objective_id UUID,
            prompt_id UUID
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


def add_obj_prompt(objective_id, prompt_id):
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO m_objectives_prompts (objective_id, prompt_id) VALUES ('{objective_id}', '{prompt_id}');
    """)

    conn.commit()
    cur.close()
    conn.close()



def new_samples_table(): 
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_samples (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            sample_title TEXT,
            created_by TEXT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


def add_to_samples(sample_table, sample_title, created_by):
    conn = get_db_connection()
    # Insert a new row
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO m_samples (sample_table, sample_title, created_by) VALUES ('{sample_table}', '{sample_title}', '{created_by}');
    """)
    conn.commit()
    cur.close()
    conn.close()

def new_experiments_table(): 
    conn = get_db_connection()

    cur = conn.cursor()
    cur.execute("""
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
    """)

    conn.commit()
    cur.close()
    conn.close()


def new_results_table():
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute("""
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
    """)

    conn.commit()
    cur.close()
    conn.close()


def new_samples_lessons_table():
    conn = get_db_connection()

    # Insert a new row
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_sample_lesson_plans (
            sample_id UUID,
            lesson_plan_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


def new_teachers_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_teachers (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            name TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def add_teacher(name):
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if the teacher already exists
    cur.execute("""
        SELECT 1 FROM m_teachers WHERE name = %s;
    """, (name,))
    if cur.fetchone() is not None:
        print("Teacher already exists.")
        cur.close()
        conn.close()
        return "Teacher already exists."

    # Insert a new teacher
    cur.execute("""
        INSERT INTO m_teachers (name) VALUES (%s);
    """, (name,))

    conn.commit()
    cur.close()
    conn.close()
    return "Teacher added successfully."
    

def new_lesson_plans_table():
    conn = get_db_connection()

    # Create a new table
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS lesson_plans (
            id TEXT, -- Alternatively: UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            lesson_id TEXT, -- Alternatively UUID
            json TEXT,
            generation_details TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            key_stage TEXT,
            subject TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_lesson_plan():
    conn = get_db_connection()

    with open('data/sample_lesson.json', 'r') as file:
        json_data = file.read()
    # print(json_data)

    # Define the values to insert
    id_value = uuid.uuid4()
    lesson_id_value = None
    json_value = json_data
    generation_details_value = 'sample lesson plan'
    key_stage_value = 'key-stage-1'
    subject_value = 'english'

    # Insert a new row
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO lesson_plans (id, lesson_id, json, generation_details, created_at, key_stage, subject) 
        VALUES (%s, %s, %s, %s, now(), %s, %s);
    """, (id_value, lesson_id_value, json_value, generation_details_value, key_stage_value, subject_value))

    conn.commit()
    cur.close()
    conn.close()



def insert_sample_prompt(csv_file_path):
    conn = get_db_connection()
    
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt_data = json.loads(row['result'])
            
            # Extract data from JSON object
            id = prompt_data['id']
            prompt_title = prompt_data['prompt_title']
            prompt_objective = prompt_data['prompt_objective']
            output_format = prompt_data['output_format']
            lesson_plan_params = prompt_data['lesson_plan_params']
            rating_criteria = prompt_data['rating_criteria']
            general_criteria_note = prompt_data['general_criteria_note']
            rating_instruction = prompt_data['rating_instruction']
            experiment_description = prompt_data['experiment_description']
            objective_title = prompt_data['objective_title']
            objective_desc = prompt_data['objective_desc']
            created_by = prompt_data['created_by']
            version = prompt_data['version']
            
            # Compute the prompt hash
            prompt_hash = hashlib.sha256(prompt_objective.encode()).digest()

            # Insert a new row into m_prompts
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO m_prompts (id, prompt_title, prompt_objective, prompt_hash, output_format, lesson_plan_params, rating_criteria, general_criteria_note, rating_instruction, experiment_description, objective_title, objective_desc, created_by, version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now());
            """, (id, prompt_title, prompt_objective, prompt_hash, output_format, lesson_plan_params, rating_criteria, general_criteria_note, rating_instruction, experiment_description, objective_title, objective_desc, created_by, version))

            conn.commit()
            cur.close()
    
    conn.close()

# Read and parse the JSON file



csv_file_path = 'data/sample_prompts.csv'

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
add_teacher('John Doe')
insert_sample_prompt(csv_file_path)
