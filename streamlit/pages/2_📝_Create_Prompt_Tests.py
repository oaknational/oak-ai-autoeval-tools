import streamlit as st
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np
from jinja_funcs import *
import re



st.set_page_config(page_title="Create Prompt Tests", page_icon="📝")
st.markdown("# 📝 Create Prompt Tests")

st.markdown("#### Please select a prompt to modify. As you make modifications on the selected prompt, the changes will be visible on the sidebar.:arrow_left:")
st.markdown("##### Don't forget to click Refresh Prompt if you decide to use another existing prompt.")
st.markdown("##### Once you are satisfied with the changes in each part, click Save Changes to save your modifications.")
st.markdown("##### You can also view the rendered version of the prompt with a sample lesson plan down below.")

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Function to clear cache
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

# def get_all_prompts():
#     conn = psycopg2.connect(
#         dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
#     )
#     query = """
#     SELECT id, prompt_objective, lesson_plan_params, output_format, rating_criteria, general_criteria_note, rating_instruction, encode(prompt_hash, 'hex'), prompt_title, experiment_description, objective_title, objective_desc, created_at, created_by, version
#     FROM public.m_prompts;
#     """
#     cur = conn.cursor()
#     cur.execute(query)
#     data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
#     cur.close()
#     conn.close()
    
#     # Parse JSON fields safely
#     data['rating_criteria'] = data['rating_criteria'].apply(lambda x: json.loads(x) if x else {})
    
#     return data

def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except ValueError:
        return False

def fix_json_format(json_string):
    # Add double quotes around keys if they are missing
    json_string = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', json_string)
    # Ensure property names are quoted properly
    json_string = re.sub(r"'", r'"', json_string)
    return json_string

data = get_all_prompts()

filtered_data = data

prompt_title_options = filtered_data['prompt_title'].unique().tolist()
prompt_title = st.selectbox('Select an existing prompt to modify:', prompt_title_options)

filtered_data = filtered_data[filtered_data['prompt_title'] == prompt_title]

latest_prompt = filtered_data.loc[filtered_data['created_at'].idxmax()]

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if not filtered_data.empty:
    max_index = len(filtered_data) - 1
    if not (0 <= st.session_state.current_index <= max_index):
        st.session_state.current_index = 0
        st.error("Selected prompt index is out of bounds, resetting to first prompt.")

    current_prompt = filtered_data.loc[filtered_data['created_at'].idxmax()]
    prompt_title_unique_checker = current_prompt['prompt_title']
    prompt_id = current_prompt['id']
   
    st.table(current_prompt[['created_at', 'prompt_title', 'prompt_objective', 'lesson_plan_params', 'output_format', 'rating_criteria', 'general_criteria_note', 'rating_instruction', 'experiment_description', 'objective_title', 'objective_desc', 'created_by']])
    
    if 'draft_prompt' not in st.session_state or st.session_state['refresh']:
        st.session_state['draft_prompt'] = current_prompt.copy(deep=True)
        st.session_state['refresh'] = False
    elif st.button('Refresh Prompt'):
        st.session_state['refresh'] = True
        st.experimental_rerun()
        
    prompt_title = st.text_input("Prompt Title", value=current_prompt['prompt_title'])
    prompt_objective = st.text_area("Prompt Objective", value=current_prompt['prompt_objective'], height=200)
    lesson_plan_params = st.text_area("Lesson Plan Parameters", value=current_prompt['lesson_plan_params'], height=100)
    output_format = st.text_input("Output Format", value=current_prompt['output_format'])
    rating_criteria = st.text_area("Rating Criteria", value=current_prompt['rating_criteria'], height=200)
    general_criteria_note = st.text_area("General Criteria Note", value=current_prompt['general_criteria_note'], height=100)
    rating_instruction = st.text_area("Rating Instruction", value=current_prompt['rating_instruction'], height=100)
    experiment_description = st.text_area("Experiment Description", value=current_prompt['experiment_description'], height=100)
    objective_title = st.text_input("Objective Title", value=current_prompt['objective_title'])
    objective_desc = st.text_area("Objective Description", value=current_prompt['objective_desc'], height=100)

    teachers = get_teachers()
    teachers_options = teachers['name'].tolist()
    teacher_option = ['Select a teacher'] + teachers_options
    created_by = st.selectbox('Who is creating the prompt?', teacher_option, index=teacher_option.index(current_prompt['created_by']) if current_prompt['created_by'] in teacher_option else 0)

    if st.button("Save Changes"):
        st.session_state['draft_prompt']['prompt_title'] = prompt_title
        st.session_state['draft_prompt']['prompt_objective'] = prompt_objective
        st.session_state['draft_prompt']['lesson_plan_params'] = lesson_plan_params
        st.session_state['draft_prompt']['output_format'] = output_format
        rating_criteria_fixed = fix_json_format(rating_criteria)
        if is_valid_json(rating_criteria_fixed):
            st.session_state['draft_prompt']['rating_criteria'] = json.loads(rating_criteria_fixed)
        else:
            st.error("Unable to fix JSON format in Rating Criteria")
        st.session_state['draft_prompt']['general_criteria_note'] = general_criteria_note
        st.session_state['draft_prompt']['rating_instruction'] = rating_instruction
        st.session_state['draft_prompt']['experiment_description'] = experiment_description
        st.session_state['draft_prompt']['objective_title'] = objective_title
        st.session_state['draft_prompt']['objective_desc'] = objective_desc
        st.session_state['draft_prompt']['created_by'] = created_by
        
        st.success("Changes saved successfully!")

    if st.button("Save Prompt", help="Save the prompt to the database."):
        prompt_objective = st.session_state['draft_prompt']['prompt_objective']
        lesson_plan_params = st.session_state['draft_prompt']['lesson_plan_params']
        output_format = st.session_state['draft_prompt']['output_format']
        rating_criteria = st.session_state['draft_prompt']['rating_criteria']
        general_criteria_note = st.session_state['draft_prompt']['general_criteria_note']
        rating_instruction = st.session_state['draft_prompt']['rating_instruction']
        prompt_title = st.session_state['draft_prompt']['prompt_title']
        experiment_description = st.session_state['draft_prompt']['experiment_description']
        objective_title = st.session_state['draft_prompt']['objective_title']
        objective_desc = st.session_state['draft_prompt']['objective_desc']
        prompt_created_by = st.session_state['draft_prompt']['created_by']
        
        if prompt_title_unique_checker == prompt_title:
            version = str(int(st.session_state['draft_prompt']['version']) + 1)
        else:
            version = '1'
        
        returned_id = to_prompt_metadata_db(
            prompt_objective, 
            lesson_plan_params, 
            output_format, 
            json.dumps(rating_criteria),  # Convert JSON object back to string
            general_criteria_note, 
            rating_instruction, 
            prompt_title, 
            experiment_description, 
            objective_title, 
            objective_desc, 
            prompt_created_by, 
            version
        )

        st.success(f"Prompt saved successfully! With ID: {returned_id}")

    def pretty_print_json(json_data):
        if json_data:
            return json.dumps(json_data, indent=2, sort_keys=False)
        return "None"

    st.sidebar.header('Draft Prompt Preview')
    st.sidebar.markdown("#### Prompt Title:")
    st.sidebar.markdown(st.session_state['draft_prompt']['prompt_title'])
    st.sidebar.markdown("#### Objective:")
    st.sidebar.markdown(st.session_state['draft_prompt']['prompt_objective'])

    st.sidebar.markdown("#### Relevant Lesson Plan Parts:")
    st.sidebar.markdown(st.session_state['draft_prompt']['lesson_plan_params'])

    st.sidebar.markdown("#### Output Format:")
    st.sidebar.markdown(st.session_state['draft_prompt']['output_format'])

    st.sidebar.markdown("#### Rating Criteria:")
    rating_criteria = st.session_state['draft_prompt']['rating_criteria']
    st.sidebar.code(rating_criteria, language='json')

    st.sidebar.markdown("#### General Criteria Note:")
    st.sidebar.markdown(st.session_state['draft_prompt']['general_criteria_note'])

    st.sidebar.markdown("#### Rating Instruction:")
    st.sidebar.markdown(st.session_state['draft_prompt']['rating_instruction'])
    
    st.sidebar.markdown("#### Experiment Description:")
    st.sidebar.markdown(st.session_state['draft_prompt']['experiment_description'])

    st.sidebar.markdown("#### Objective Title:")
    st.sidebar.markdown(st.session_state['draft_prompt']['objective_title'])

    st.sidebar.markdown("#### Objective Description:")
    st.sidebar.markdown(st.session_state['draft_prompt']['objective_desc'])

    st.sidebar.markdown("#### Created By:")
    st.sidebar.markdown(st.session_state['draft_prompt']['created_by'])

    st.markdown('### Rendered Prompt')
    st.warning('This feature has compatibility issues with edited prompts due to json rating_criteria and might not be working. We are working on fixing it but if you want to use it; one workaround is to edit the prompt rating_criteria directly on the DB table to work as a dictionary.')
    st.markdown('You can review what will be sent to the LLM after the prompt is rendered with the prompt template below.')
    with st.expander("# Click to Expand"):
        file_path = os.path.join(DATA_PATH, 'sample_lesson_empty.json')
        with open(file_path, 'r') as file:
            sample_lesson_dict = json.load(file)

        prompt_details = get_prompt(prompt_id)

        # prompt_details
        

        prompt_details_processed = process_prompt(prompt_details) #our new function
        rendered_prompt = render_prompt(sample_lesson_dict, prompt_details_processed)
        st.markdown(rendered_prompt)

else:
    st.write("No prompts available for the selected title.")
