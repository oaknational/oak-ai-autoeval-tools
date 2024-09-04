""" Streamlit page for building and managing datasets in the AutoEval app.
    Enables creation of new subsets of lesson plans to run evaluations on. 

Functionality:

- Provides user inputs for dataset title, creator's name, and keyword 
    search for lesson plans.
- Displays the retrieved lesson plans.
- Allows saving selected lesson plans to a new or existing sample.
- Includes a button to clear the cache.
"""
import pandas as pd
import streamlit as st


from utils import clear_all_caches
from db_scripts import (
execute_single_query, execute_multi_query, new_sample, add_lesson_plans_to_sample
)



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
    st.session_state.sample_id = None

if 'lesson_plan_ids' not in st.session_state:
    st.session_state.lesson_plan_ids = []

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
        st.dataframe(lesson_plans)
        st.session_state.lesson_plan_ids = lesson_plans["id"].tolist()
    else:
        st.warning("No lesson plans found with the given filters.")

# Save sample with selected lesson plans
if st.button("Save Sample with Selected Lesson Plans"):
    if sample_title and created_by:
        st.session_state.sample_id = None
        
        sample_id = new_sample(sample_title, created_by)

        if sample_id:
            if add_lesson_plans_to_sample(
                sample_id, st.session_state.lesson_plan_ids
            ):
                st.success("Sample and lesson plans added successfully!")
            else:
                st.error("Failed to add lesson plans to the sample.")
    else:
        st.warning("Please fill in all the required fields.")
