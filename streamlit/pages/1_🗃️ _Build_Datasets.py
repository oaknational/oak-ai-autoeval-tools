""" Streamlit page for building and managing datasets in the AutoEval app.
    Enables creation of new subsets of lesson plans to run evaluations on. 

Functionality:

- Provides user inputs for dataset title, creator's name, and keyword 
    search for lesson plans.
- Displays the retrieved lesson plans.
- Allows saving selected lesson plans to a new or existing sample.
- Includes a button to clear the cache.
"""

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from utils import clear_all_caches, execute_single_query, execute_multi_query

load_dotenv()


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
    return result[0][0] if result else None


def get_lesson_plans(keyword=None):
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


def get_unique_values(column_name):
    """ Get unique values for a specified column in the lesson_plans 
        table.

    Args:
        column_name (str): Name of the column to fetch unique values from.

    Returns:
        list: List of unique values.
    """
    query = f"SELECT DISTINCT {column_name} FROM lesson_plans;"
    result = execute_single_query(query)
    return [value[0] for value in result] if result else []


# Set page configuration
st.set_page_config(page_title="Build Datasets", page_icon="🗃️")
st.markdown("# 🗃️ Build Datasets")
st.write("Create a new subset of lesson plans to run evaluations on.")

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
sample_title = st.text_input(
    "Enter a dataset title for the Eval UI (e.g. history_ks2):"
)
created_by = st.text_input("Enter your name: ")

# Keyword search for generation details
keyword = st.text_input("Enter keyword for generation details:")

# Get lesson plans
if st.button("Get Lesson Plans"):
    lesson_plans = get_lesson_plans(keyword)
    if not lesson_plans.empty:
        st.write("Lesson Plans:")




        # REVIEW WHETHER I CAN CHANGE THIS:
        lesson_plans_df = pd.DataFrame(
            lesson_plans, columns=["id", "generation_details"]
        )
        st.dataframe(lesson_plans_df)
        # FOR THIS CODE:
        #st.dataframe(lesson_plans)





        st.session_state['lesson_plan_ids'] = lesson_plans_df["id"].tolist()
    else:
        st.warning("No lesson plans found with the given filters.")

# Save sample with selected lesson plans
if st.button("Save Sample with Selected Lesson Plans"):
    if sample_title and created_by:
        if st.session_state['sample_id'] is None:
            sample_id = new_sample(sample_title, created_by)
            st.session_state['sample_id'] = sample_id
        else:
            sample_id = st.session_state['sample_id']

        if sample_id:
            if add_lesson_plans_to_sample(
                sample_id, st.session_state['lesson_plan_ids']
            ):
                st.success("Sample and lesson plans added successfully!")
            else:
                st.error("Failed to add lesson plans to the sample.")
        else:
            st.error("Failed to create a new sample.")
    else:
        st.warning("Please fill in all the required fields.")
