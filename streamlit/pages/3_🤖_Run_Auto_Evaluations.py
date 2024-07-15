""" 
Streamlit page for running evaluations in the AutoEval app.
    
Functionality:
- Allows running evaluations on a dataset using selected prompts.
- Results are stored in the database and can be viewed in the
    Visualise Results page.
"""

import pandas as pd
from dotenv import load_dotenv
import streamlit as st

from utils import (
    clear_all_caches, get_prompts, get_samples, get_teachers,
    generate_experiment_placeholders, start_experiment
)

load_dotenv()


def update_session_state(key, value):
    """
    Updates the Streamlit session state with a specified key-value pair.

    Args:
        key (str): The key in the session state to update.
        value (any): The new value to set for the specified key.
    """
    st.session_state[key] = value


# Set page configuration
st.set_page_config(page_title="Run Auto Evaluations", page_icon="🤖")

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

# Page and sidebar headers
st.markdown("# 🤖 Run Auto Evaluations")
st.sidebar.header("Run Auto Evaluations")
st.write(
    """
    This page allows you to run evaluations on a dataset using a
    selected prompt. Results will be stored in the database and can be 
    viewed in the Visualise Results page.
    """
)

# Initialize session state
default_session_state = {
    'llm_model': 'gpt-4',
    'llm_model_temp': 0.5,
    'limit': 5,
    'created_by': 'Select a teacher',
    'experiment_run': False
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Fetching data
prompts_data = get_prompts()
samples_data = get_samples()
teachers_data = get_teachers()

# Order samples_data by created_at
samples_data = samples_data.sort_values(by='created_at', ascending=False)

samples_data['samples_options'] = (
    samples_data['sample_title'] + " (" +
    samples_data['number_of_lessons'].astype(str) + ")"
)
samples_options = samples_data['samples_options'].tolist()

prompts_data['prompt_title_with_date'] = (
    prompts_data['prompt_title'] + " (" +
    prompts_data['output_format'].astype(str) + ")"
)
prompt_title_options = prompts_data['prompt_title_with_date'].tolist()

# Test selection section
st.subheader("Test selection")
prompt_options = st.multiselect(
    'Select prompts:', 
    prompt_title_options,
    help='You can select multiple prompts to run evaluations on.'
)

# Filter prompts based on selection
prompt_data = (
    prompts_data[prompts_data['prompt_title_with_date'].isin(prompt_options)]
)
prompt_table = pd.DataFrame({
    'Prompt': prompt_options, 
    'Description': [
        prompt_data[
            prompt_data['prompt_title_with_date'] == prompt
        ]['experiment_description'].iloc[0]
        for prompt in prompt_options
    ]
})
st.dataframe(prompt_table, hide_index=True, use_container_width=True)

prompt_ids = [
    prompt_data[prompt_data['prompt_title_with_date'] == prompt]['id'].iloc[0]
    for prompt in prompt_options
]

# Dataset selection section
st.subheader("Dataset selection")
sample_options = st.multiselect(
    'Select datasets to run evaluation on:',
    samples_options,
    help='(Number of Lesson Plans in the Sample)'
)
samples_data = samples_data[(
    samples_data['samples_options'].isin(sample_options)
)]

# Get sample IDs
sample_ids = [
    samples_data[samples_data['samples_options'] == sample]['id'].iloc[0]
    for sample in sample_options
]

# Create samples table
samples_table = pd.DataFrame({
    'Sample': sample_options, 
    'Number of Lessons': [
        samples_data[
            samples_data['samples_options'] == sample
        ]['number_of_lessons'].iloc[0] for sample in sample_options
    ]
})

st.dataframe(samples_table, hide_index=True, use_container_width=True)


# Calculate time estimates and set limits
MAX_LESSONS = 5
TOTAL_SAMPLE_COUNT = TOTAL_PROMPT_COUNT = 0

max_lessons = (
    samples_table['Number of Lessons'].max()
    if not samples_table.empty else MAX_LESSONS
)
total_sample_count = (
    samples_table['Number of Lessons'].sum()
    if not samples_table.empty else TOTAL_SAMPLE_COUNT
)
total_prompt_count = (
    prompt_table.shape[0]
    if not prompt_table.empty else TOTAL_PROMPT_COUNT
)

AVG_LATENCY = 7.78  # seconds
total_time = total_sample_count * total_prompt_count * AVG_LATENCY
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

st.warning('A limit is advised to avoid long run times.')
st.warning(
    f"""
    Estimated time to run evaluations without Limit: {int(hours)} hours,
    {int(minutes)} minutes, {int(seconds)} seconds
    """
)

# Set limit on lesson plans
st.session_state['limit'] = st.number_input(
    'Set a limit on the number of lesson plans per sample to evaluate:',
    min_value=1,
    max_value=9000,
    value=max_lessons,
    help='Minimum value is 1.',
    key='limit_input',
    on_change=update_session_state,
    args=('limit', st.session_state['limit_input'])
)

llm_model_options = ['gpt-4', 'gpt-4o', 'gpt-4-turbo']

# Model selection section
st.selectbox(
    'Select a model:',
    llm_model_options, index=0, key='model_select',
    on_change=update_session_state, args=(
        'llm_model', st.session_state['model_select'])
)
st.number_input(
    'Enter temperature:', min_value=0.0, max_value=2.00, value=0.5, 
    key='temp_input',
    help='Minimum value is 0.0, maximum value is 2.00.',
    on_change=update_session_state, args=(
        'llm_model_temp', st.session_state['temp_input'])
)
teachers_options = ['Select a teacher'] + teachers_data['name'].tolist()
st.selectbox(
    'Who is running the experiment?',
    teachers_options, index=0, key='created_by_select',
    on_change=update_session_state, args=(
        'created_by', st.session_state['created_by_select'])
)
teacher_id = (
    teachers_data.loc[
        teachers_data['name'] == st.session_state['created_by'], 'id'
    ].iloc[0] if st.session_state['created_by'] != 'Select a teacher' else None
)

# Generate placeholders dynamically
placeholder_name, placeholder_description = generate_experiment_placeholders(
    st.session_state.llm_model,
    st.session_state.llm_model_temp,
    st.session_state.limit,
    len(prompt_ids),
    len(sample_ids),
    st.session_state.created_by
)

tracked_options = ['True','False']
tracked = st.selectbox('should experiment be tracked?', options=tracked_options)

with st.form(key='experiment_form'):
    st.subheader("Experiment information")
    experiment_name = st.text_input(
        'Enter experiment name:',
        value=placeholder_name,
        placeholder=placeholder_name
    )
    exp_description = st.text_input(
        'Enter experiment description:',
        value=placeholder_description,
        placeholder=placeholder_description
    )

    if st.form_submit_button('Run evaluation'):
        st.warning('Please do not close the page until the evaluation is complete.')
        start_experiment(
            experiment_name, exp_description, sample_ids, teacher_id,
            prompt_ids, st.session_state.limit, st.session_state.llm_model,
            tracked, st.session_state.llm_model_temp
        )
        st.session_state.experiment_run = True

# Conditionally display the View Insights button based on the experiment run flag
if st.session_state.experiment_run:
    st.write('**Click the button to view insights.**')
    if st.button('View Insights'):
        st.switch_page('pages/4_🔍_Visualise Results.py')
