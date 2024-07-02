import streamlit as st
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
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

# Function to add a new sample
def new_sample(sample_title, created_by):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO public.m_samples (id, created_at, updated_at, sample_title, created_by)
            VALUES (gen_random_uuid(), NOW(), NOW(), %s, %s)
            RETURNING id;
            """,
            (sample_title, created_by)
        )
        sample_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
        connection.close()
        st.session_state['sample_id'] = sample_id
        st.info(f"Sample created with ID: {sample_id}")
        return sample_id
    except Exception as e:
        st.error(f"Error creating sample: {e}")
        return None

# Function to get lesson plans based on filters
def get_lesson_plans(keyword=None):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = """
            SELECT lp.id, lp.generation_details
            FROM lesson_plans lp
            WHERE 1=1
        """
        
        params = []
        
        if keyword:
            query += " AND lp.generation_details LIKE %s"
            params.append(f"%{keyword}%")
        
        cursor.execute(query, params)
        lesson_plans = cursor.fetchall()
        cursor.close()
        connection.close()
        return lesson_plans
    except Exception as e:
        st.error(f"Error fetching lesson plans: {e}")
        return []

# Function to link lesson plans to a sample
def add_lesson_plans_to_sample(sample_id, lesson_plan_ids):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        for lesson_plan_id in lesson_plan_ids:
            cursor.execute(
                """
                INSERT INTO public.m_sample_lesson_plans (sample_id, lesson_plan_id)
                VALUES (%s, %s);
                """,
                (sample_id, lesson_plan_id)
            )
        connection.commit()
        cursor.close()
        connection.close()
        st.info(f"Lesson plans linked to sample with ID: {sample_id}")
        return True
    except Exception as e:
        st.error(f"Error linking lesson plans to sample: {e}")
        return False

# Function to get unique values for a column
def get_unique_values(column_name):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = f"SELECT DISTINCT {column_name} FROM lesson_plans;"
        cursor.execute(query)
        values = cursor.fetchall()
        cursor.close()
        connection.close()
        return [value[0] for value in values]
    except Exception as e:
        st.error(f"Error fetching unique values: {e}")
        return []

# Set page configuration
st.set_page_config(page_title="Build Datasets", page_icon="🗃️")
st.markdown("# 🗃️ Build Datasets")
st.write("Create a new subset of lesson plans to run evaluations on.")

# Function to clear cache
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

# Initialize session state
if 'sample_id' not in st.session_state:
    st.session_state['sample_id'] = None

if 'lesson_plan_ids' not in st.session_state:
    st.session_state['lesson_plan_ids'] = []

# Fetch unique values for subjects and key stages
unique_subjects = get_unique_values("subject")
unique_key_stages = get_unique_values("key_stage")

# Get user input
sample_title = st.text_input("Enter a dataset title for the Eval UI (e.g. history_ks2):")
created_by = st.text_input("Enter your name: ")



# Keyword search for generation details
keyword = st.text_input("Enter keyword for generation details:")

# Get lesson plans
if st.button("Get Lesson Plans"):
    lesson_plans = get_lesson_plans(keyword)
    if lesson_plans:
        st.write("Lesson Plans:")
        lesson_plans_df = pd.DataFrame(lesson_plans, columns=["id", "generation_details"])
        st.dataframe(lesson_plans_df)
        st.session_state['lesson_plan_ids'] = lesson_plans_df["id"].tolist()
    else:
        st.warning("No lesson plans found with the given filters.")

# Save sample with selected lesson plans
if st.button("Save Sample with Selected Lesson Plans"):
    if sample_title and created_by:
        if st.session_state['sample_id'] is None:
            sample_id = new_sample(sample_title, created_by)
        else:
            sample_id = st.session_state['sample_id']
        
        if sample_id:
            if add_lesson_plans_to_sample(sample_id, st.session_state['lesson_plan_ids']):
                st.success("Sample and lesson plans added successfully!")
            else:
                st.error("Failed to add lesson plans to the sample.")
        else:
            st.error("Failed to create a new sample.")
    else:
        st.warning("Please fill in all the required fields.")
