import os
import re
import streamlit as st
import psycopg2
from dotenv import load_dotenv
from jinja_funcs import *
import mlflow
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Run Auto Evaluations", page_icon="🤖")

# Function to clear cache
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

# Page and sidebar headers
st.markdown("# 🤖 Run Auto Evaluations")
st.sidebar.header("Run Auto Evaluations")
st.write('This page allows you to run evaluations on a dataset using a selected prompts. Results will be stored in the database and can be viewed in the Visualise Results page.')
load_dotenv()

# Initialize session state
if 'llm_model' not in st.session_state:
    st.session_state['llm_model'] = 'gpt-4'
if 'llm_model_temp' not in st.session_state:
    st.session_state['llm_model_temp'] = 0.5
if 'limit' not in st.session_state:
    st.session_state['limit'] = 5
if 'created_by' not in st.session_state:
    st.session_state['created_by'] = 'Select a teacher'
if 'experiment_run' not in st.session_state:
    st.session_state['experiment_run'] = False

# Fetching data
prompts_data = get_prompts()
samples_data = get_samples()
teachers_data = get_teachers()

# Order samples_data by created_at desc
samples_data = samples_data.sort_values(by='created_at', ascending=False)

samples_data['samples_options'] = samples_data['sample_title'] + " (" + samples_data['number_of_lessons'].astype(str) + ")"
samples_options = samples_data['samples_options'].tolist()

prompts_data['prompt_title_with_date'] = prompts_data['prompt_title'] + " (" + prompts_data['output_format'].astype(str) + ")"
st.subheader("Test selection")
prompt_title_options = prompts_data['prompt_title_with_date'].tolist()

# Select prompts
prompt_options = st.multiselect('Select prompts:', prompt_title_options, help='You can select multiple prompts to run evaluations on.')

prompt_data = prompts_data[prompts_data['prompt_title_with_date'].isin(prompt_options)]
prompt_table = pd.DataFrame({'Prompt': prompt_options, 'Description': [prompt_data[prompt_data['prompt_title_with_date'] == prompt]['experiment_description'].iloc[0] for prompt in prompt_options]})
st.dataframe(prompt_table, hide_index=True, use_container_width=True)

prompt_ids = [prompt_data[prompt_data['prompt_title_with_date'] == prompt]['id'].iloc[0] for prompt in prompt_options]

st.subheader("Dataset selection")

# Select dataset
sample_options = st.multiselect('Select datasets to run evaluation on:', samples_options, help='(Number of Lesson Plans in the Sample)')
samples_data = samples_data[(samples_data['samples_options'].isin(sample_options))]

sample_ids = [samples_data[samples_data['samples_options'] == sample]['id'].iloc[0] for sample in sample_options]

samples_table = pd.DataFrame({'Sample': sample_options, 'Number of Lessons': [samples_data[samples_data['samples_options'] == sample]['number_of_lessons'].iloc[0] for sample in sample_options]})
st.dataframe(samples_table, hide_index=True, use_container_width=True)

max_lessons = 5
total_sample_count = 0
total_prompt_count = 0

if samples_table is not None and not samples_table.empty:
    total_sample_count = samples_table['Number of Lessons'].sum()
    max_lessons = samples_table['Number of Lessons'].max()
    st.session_state['limit'] = max_lessons
    
if prompt_table is not None and not prompt_table.empty:
    total_prompt_count = prompt_table.shape[0]

Avg_Latency = 7.78  # seconds
total_time = total_sample_count * total_prompt_count * Avg_Latency
hours = total_time // 3600
total_time = total_time % 3600
minutes = total_time // 60
seconds = total_time % 60
st.warning('A limit is advised to avoid long run times.')
st.warning(f"Estimated time to run evaluations without Limit: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

# Callbacks to update session state
def update_model():
    st.session_state.llm_model = st.session_state.model_select

def update_temp():
    st.session_state.llm_model_temp = st.session_state.temp_input

def update_limit():
    st.session_state.limit = st.session_state.limit_input

def update_created_by():
    st.session_state.created_by = st.session_state.created_by_select

# Set limit on lesson plans
st.session_state['limit'] = st.number_input('Set a limit on the number of lesson plans per sample to evaluate:', min_value=1, max_value=9000, value=max_lessons, help='Minimum value is 1.', key='limit_input', on_change=update_limit)

llm_model_options = ['gpt-4o','gpt-4o-mini','gpt-4', 'gpt-4-turbo','gpt-4-32k']

# Callbacks outside of the form
llm_model = st.selectbox('Select a model:', llm_model_options, index=0, key='model_select', on_change=update_model)
llm_model_temp = st.number_input('Enter temperature:', min_value=0.0, max_value=2.00, value=0.5, key='temp_input', help='Minimum value is 0.0, maximum value is 2.00.', on_change=update_temp)
teachers_options = teachers_data['name'].tolist()
teacher_option = ['Select a teacher'] + teachers_options
created_by = st.selectbox('Who is running the experiment?', teacher_option, index=0, key='created_by_select', on_change=update_created_by)

if created_by != 'Select a teacher':
    teachers_data = teachers_data[teachers_data['name'] == created_by]
    teacher_id = teachers_data['id'].iloc[0]
else:
    teacher_id = None

# Generate placeholders dynamically
placeholder_name, placeholder_description = generate_experiment_placeholders(st.session_state.llm_model, st.session_state.llm_model_temp, st.session_state.limit, len(prompt_ids), len(sample_ids), st.session_state.created_by)

# st.session_state.limit 
# st.session_state.llm_model 
# st.session_state.llm_model_temp
tracked_options = ['True','False']
tracked = st.selectbox('should experiment be tracked?', options=tracked_options)

with st.form(key='experiment_form'):
    st.subheader("Experiment information")
    experiment_name = st.text_input('Enter experiment name:', value=placeholder_name, placeholder=placeholder_name)
    exp_description = st.text_input('Enter experiment description:', value=placeholder_description, placeholder=placeholder_description)
    
    if st.form_submit_button('Run evaluation'):
        st.warning('Please do not close the page until the evaluation is complete.')
        placeholder = st.empty()
        start_experiment(experiment_name, exp_description, sample_ids, teacher_id, prompt_ids, st.session_state.limit, st.session_state.llm_model, tracked, st.session_state.llm_model_temp)
        st.session_state.experiment_run = True  # Set the flag to True when the experiment is completed
        

# Conditionally display the View Insights button based on the experiment run flag
if st.session_state.experiment_run:
    st.write('**Click the button to view insights.**')
    if st.button('View Insights'):
        st.switch_page('pages/4_🔍_Visualise Results.py')
