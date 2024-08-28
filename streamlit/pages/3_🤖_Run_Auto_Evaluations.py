""" 
Streamlit page for running evaluations in the AutoEval app.
    
Functionality:
- Allows running evaluations on a dataset using selected prompts.
- Results are stored in the database and can be viewed in the
    Visualise Results page.
"""
import pandas as pd
import streamlit as st

from utils import (
    clear_all_caches, get_prompts, get_samples, get_teachers,
    generate_experiment_placeholders, start_experiment
)
from constants import OptionConstants, ColumnLabels


# Set page configuration
st.set_page_config(page_title="Run Auto Evaluations", page_icon="🤖")

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

# Page and sidebar headers
st.markdown("# 🤖 Run Auto Evaluations")
st.write(
    """
    This page allows you to run evaluations on a dataset using a
    selected prompt. Results will be stored in the database and can be 
    viewed in the Visualise Results page.
    """
)

# Initialize session state
if 'llm_model' not in st.session_state: 
    st.session_state.llm_model = 'gpt-4o'
if 'llm_model_temp' not in st.session_state:
    st.session_state.llm_model_temp = 0.5
if 'limit' not in st.session_state:
    st.session_state.limit = 5
if 'created_by' not in st.session_state:
    st.session_state.created_by = OptionConstants.SELECT_TEACHER
if 'experiment_run' not in st.session_state:
    st.session_state.experiment_run = False

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
    ColumnLabels.NUM_LESSONS: [
        samples_data[
            samples_data['samples_options'] == sample
        ]['number_of_lessons'].iloc[0] for sample in sample_options
    ]
})

st.dataframe(samples_table, hide_index=True, use_container_width=True)

# Calculate time estimates and set limits
max_lessons = (
    samples_table[ColumnLabels.NUM_LESSONS].max()
    if not samples_table.empty else 5
)
total_sample_count = (
    samples_table[ColumnLabels.NUM_LESSONS].sum()
    if not samples_table.empty else 0
)
total_prompt_count = (
    prompt_table.shape[0]
    if not prompt_table.empty else 0
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
st.session_state.limit = st.number_input(
    'Set a limit on the number of lesson plans per sample to evaluate:',
    min_value=1,
    max_value=9000,
    value=st.session_state.limit,
    help='Minimum value is 1.'
)

llm_model_options = ['gpt-4o','gpt-4o-mini-2024-07-18',
                     'gpt-4o-2024-05-13','gpt-4o-2024-08-06','chatgpt-4o-latest',
                     'gpt-4-turbo-2024-04-09','gpt-4-0125-preview','gpt-4-1106-preview','llama',]

st.session_state.llm_model = st.selectbox(
    'Select a model:',
    llm_model_options,
    index=llm_model_options.index(st.session_state.llm_model)
)

st.session_state.llm_model_temp = st.number_input(
    'Enter temperature:',
    min_value=0.0, max_value=2.00,
    value=st.session_state.llm_model_temp,
    help='Minimum value is 0.0, maximum value is 2.00.'
)

if 'top_p' not in st.session_state:
    st.session_state.top_p = 1.0  


st.session_state.top_p = st.number_input(
    'Enter top_p for the model:',
    min_value=0.0, max_value=1.0,  
    value=float(st.session_state.top_p),  
    step=0.01,  
    help='Minimum value is 0.0, maximum value is 1.00.'
)

teachers_options = [OptionConstants.SELECT_TEACHER] + teachers_data['name'].tolist()

st.session_state.created_by = st.selectbox(
    'Who is running the experiment?',
    teachers_options,
    index=teachers_options.index(st.session_state.created_by)
)

teacher_id = None
if st.session_state.created_by != OptionConstants.SELECT_TEACHER:
    teacher_id = teachers_data[
        teachers_data['name'] == st.session_state.created_by
    ]['id'].iloc[0]

# Generate placeholders dynamically
placeholder_name, placeholder_description = generate_experiment_placeholders(
    st.session_state.llm_model,
    st.session_state.llm_model_temp,
    st.session_state.limit,
    len(prompt_ids),
    len(sample_ids),
    st.session_state.created_by
)

tracked = st.selectbox(
    'Should experiment be tracked?', options=['True', 'False']
)

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
        st.warning(
            'Please do not close the page until the evaluation is complete.'
        )
        experiment_complete = start_experiment(
            experiment_name, exp_description, sample_ids, teacher_id,
            prompt_ids, st.session_state.limit, st.session_state.llm_model,
            tracked, st.session_state.llm_model_temp, st.session_state.top_p
        )
        
        if experiment_complete:
            st.session_state.experiment_run = True
        else:
            st.error(
                "Experiment failed to complete. "
                "Please check the logs for details."
            )

if st.session_state.experiment_run:
    st.write('**Click the button to view insights.**')
    if st.button('View Insights'):
        st.switch_page('pages/4_🔍_Visualise_Results.py')