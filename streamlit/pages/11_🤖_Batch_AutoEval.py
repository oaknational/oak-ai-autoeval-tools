""" 
Streamlit page for running batches of evaluations in the AutoEval app.
    
Functionality:
- Allows running evaluations on multiple datasets 
    using selected prompts, with 50% lower costs, a separate pool of 
    significantly higher rate limits, and a clear 24-hour turnaround 
    time. For processing jobs that don't require immediate responses.

- Results are stored in the database and can be viewed in the
    Visualise Results page.
"""
import io
import uuid
import json

import pandas as pd
import streamlit as st
from openai import OpenAI
from openai import OpenAIError

from utils.common_utils import (
    clear_all_caches,
    log_message,
    get_env_variable,
    render_prompt
)
from utils.formatting import (
    generate_experiment_placeholders,
    lesson_plan_parts_at_end,
    display_at_end_score_criteria,
    display_at_end_boolean_criteria,
    decode_lesson_json,
    process_prompt
)
from utils.db_scripts import (
    get_prompts,
    get_samples,
    get_teachers,
    add_batch,
    add_experiment,
    get_lesson_plans_by_id,
    get_prompt,
    execute_single_query
)
from utils.constants import (
    OptionConstants,
    ColumnLabels,
    LessonPlanParameters
)

# Initialize the OpenAI client
client = OpenAI()

# Set page configuration
st.set_page_config(page_title="Batch AutoEval", page_icon="🤖")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

# Page and sidebar headers
st.markdown("# 🤖 Batch AutoEval")
st.write(
    """
    This page allows you to run evaluations on multiple datasets using
    multiple prompts in batch mode. Batch submissions have a clear 24-hour 
    turnaround time, and are ideal for processing jobs that don't require 
    immediate responses.
    
    Results will be stored in the database and can be 
    viewed in the Visualise Results page.
    """
)

# Initialize session state
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o"
if "llm_model_temp" not in st.session_state:
    st.session_state.llm_model_temp = 0.5
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0
if "limit" not in st.session_state:
    st.session_state.limit = 5
if "created_by" not in st.session_state:
    st.session_state.created_by = OptionConstants.SELECT_TEACHER
if "evaluations_list" not in st.session_state:
    st.session_state.evaluations_list = []


def new_batches_table():
    """ Create a new table `m_batches` in the database to store batch information.

    Returns:
        None
    """
    query = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS m_batches (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            batch_ref TEXT,
            batch_description TEXT,
            experiment_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            created_by TEXT,
            status TEXT);
    """
    execute_single_query(query)

new_batches_table()


def create_eval(sample_id, prompt_id, experiment_id, limit, llm_model,
        llm_model_temp, top_p=1):
    """ Run a test for each lesson plan associated with a sample and add 
    results to the database.

    Args:
        sample_id (str): ID of the sample.
        prompt_id (str): ID of the prompt.
        experiment_id (int): ID of the experiment.
        limit (int): Maximum number of records to fetch.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for LLM.

    Returns:
        None
    """
    # Convert any int64 values to Python int
    def convert_to_serializable(obj):
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif hasattr(obj, "item"):  # Handles numpy types (e.g., np.int64)
            return obj.item()
        else:
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    prompt_details = get_prompt(prompt_id)
    if not prompt_details:
        return {
            "response": {
                "result": None,
                "justification": "Prompt details not found for the given ID."
            },
            "status": "ABORTED",
        }
    lesson_plans = get_lesson_plans_by_id(sample_id, limit)
    total_lessons = len(lesson_plans)
    log_message("info", f"Total lessons: {total_lessons}")

    for i, lesson in enumerate(lesson_plans):
        lesson_plan_id = lesson[0]
        lesson_id = lesson[1]
        lesson_json_str = lesson[2]

        content = decode_lesson_json(lesson_json_str, lesson_plan_id, lesson_id, i)
        if content is None:
            continue
        
        cleaned_prompt_details = process_prompt(prompt_details)
        prompt = render_prompt(content, cleaned_prompt_details)
        
        if "Prompt details are missing" in prompt or "Missing data" in prompt:
            st.write(f"Skipping lesson {i + 1} of {total_lessons} due to missing prompt data.")
        else:
            # Create a unique custom_id for each evaluation entry
            unique_custom_id = f"{i}-{experiment_id}"
            st.write(unique_custom_id)
            
            # Create the evaluation json
            eval_entry = convert_to_serializable({
                "custom_id": unique_custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": llm_model_temp,
                    "top_p": top_p,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            })
            # Append the dictionary to the evaluations list
            st.session_state.evaluations_list.append(eval_entry)
        

def add_to_batch(
    experiment_name,
    exp_description,
    sample_ids,
    created_by,
    prompt_ids,
    limit,
    llm_model,
    tracked,
    llm_model_temp,
    top_p,
):
    """
    Add evaluations to batch.
    """
    # Create the experiment in the database
    experiment_id = add_experiment(
        experiment_name, sample_ids, created_by, tracked, llm_model,
        llm_model_temp, description=exp_description
    )
    if not experiment_id:
        log_message("error", "Failed to create experiment")
        return False
    st.success(f"Experiment details saved with ID: {experiment_id}")
    
    total_samples = len(sample_ids)
    total_prompts = len(prompt_ids)

    try:
        for sample_index, sample_id in enumerate(sample_ids):
            st.write(
                f"Working on sample {sample_index + 1} of {total_samples}"
            )

            for prompt_index, prompt_id in enumerate(prompt_ids):
                st.write(
                    f"Working on prompt {prompt_index + 1} of {total_prompts}"
                )
                create_eval(
                    sample_id, prompt_id, experiment_id, limit, llm_model,
                    llm_model_temp, top_p
                )

        return experiment_id
        
    except Exception as e:
        log_message("error", f"An error occurred during the experiment: {e}")
        return False


# Fetching data
prompts_data = get_prompts()
samples_data = get_samples()
teachers_data = get_teachers()

# Order samples_data by created_at
samples_data = samples_data.sort_values(by="created_at", ascending=False)

samples_data["samples_options"] = (
    samples_data["sample_title"]
    + " ("
    + samples_data["number_of_lessons"].astype(str)
    + ")"
)
samples_options = samples_data["samples_options"].tolist()

# Initialise lists to store selected prompts and their IDs
selected_prompts_info = []
prompt_ids = []

# Section: Test Selection
st.subheader("Test selection")
prompt_titles = prompts_data["prompt_title"].unique().tolist()
selected_prompt_titles = st.multiselect(
    "Select prompts:",
    prompt_titles,
    help="You can select multiple prompts to run evaluations on.",
)

# Iterate through each selected prompt to allow version selection
for selected_prompt_title in selected_prompt_titles:
    # Filter prompts by selected title
    filtered_prompts = prompts_data.loc[
        prompts_data["prompt_title"] == selected_prompt_title
    ].copy()

    # Filter for the preferred version
    preferred_prompt = filtered_prompts.loc[filtered_prompts["preferred"] == True]

    # Create metadata for display
    filtered_prompts["prompt_version_info"] = (
        "v"
        + filtered_prompts["version"].astype(str)
        + " | "
        + filtered_prompts["output_format"]
        + " | Created by: "
        + filtered_prompts["created_by"]
        + " | Created at: "
        + filtered_prompts["created_at"].astype(str)
    )
    
    # Apply the same for preferred_prompt
    if not preferred_prompt.empty:
        preferred_prompt["prompt_version_info"] = (
            "v"
            + preferred_prompt["version"].astype(str)
            + " | "
            + preferred_prompt["output_format"]
            + " | Created by: "
            + preferred_prompt["created_by"]
            + " | Created at: "
            + preferred_prompt["created_at"].astype(str)
        )

    # Check if multiple versions are available
    if len(filtered_prompts) > 1:
        # Display the preferred version if available, otherwise use the latest version
        if not preferred_prompt.empty:
            st.markdown(f"**Preferred Version for '{selected_prompt_title}':**")
            preferred_prompt_info = preferred_prompt["prompt_version_info"].values[0]
        else:
            st.markdown(f"**Latest Version for '{selected_prompt_title}':**")
            preferred_prompt_info = filtered_prompts.iloc[0]["prompt_version_info"]
        
        st.write(preferred_prompt_info)

        # Show full prompt details for the preferred or latest version
        current_prompt = (
            preferred_prompt.iloc[0]
            if not preferred_prompt.empty
            else filtered_prompts.iloc[0]
        )

        with st.expander("View Full Prompt for Preferred/Latest Version"):
            st.markdown(f'# *{current_prompt["prompt_title"]}* #')
            st.markdown("### Objective:")
            st.markdown(f"{current_prompt['prompt_objective']}")
            output = lesson_plan_parts_at_end(
                current_prompt["lesson_plan_params"],
                LessonPlanParameters.LESSON_PARAMS,
                LessonPlanParameters.LESSON_PARAMS_TITLES,
            )
            st.markdown(output)

            rating_criteria = json.loads(current_prompt["rating_criteria"])
            if current_prompt["output_format"] == "Score":
                display_at_end_score_criteria(rating_criteria, truncated=False)
            elif current_prompt["output_format"] == "Boolean":
                display_at_end_boolean_criteria(rating_criteria, truncated=False)

            st.markdown(f"{current_prompt['general_criteria_note']}")
            st.markdown("### Evaluation Instruction:")
            st.markdown(f"{current_prompt['rating_instruction']}")

        # Allow user to choose a different version
        use_different_version = st.checkbox(
            f"Use a different version for '{selected_prompt_title}'?"
        )

        if use_different_version:
            # Display a multiselect box with all available versions
            selected_versions = st.multiselect(
                f"Choose versions for {selected_prompt_title}:",
                filtered_prompts["prompt_version_info"].tolist(),
                help=f"You can select specific versions of {selected_prompt_title} to run evaluations on.",
            )

            # Show full prompt details for each selected version
            for selected_version in selected_versions:
                version_prompt = filtered_prompts.loc[
                    filtered_prompts["prompt_version_info"] == selected_version
                ].iloc[0]

                with st.expander(f"View Full Prompt for {selected_version}"):
                    st.markdown(f'# *{version_prompt["prompt_title"]}* #')
                    st.markdown("### Objective:")
                    st.markdown(f"{version_prompt['prompt_objective']}")
                    output = lesson_plan_parts_at_end(
                        version_prompt["lesson_plan_params"],
                        LessonPlanParameters.LESSON_PARAMS,
                        LessonPlanParameters.LESSON_PARAMS_TITLES,
                    )
                    st.markdown(output)

                    rating_criteria = json.loads(version_prompt["rating_criteria"])
                    if version_prompt["output_format"] == "Score":
                        display_at_end_score_criteria(rating_criteria, truncated=False)
                    elif version_prompt["output_format"] == "Boolean":
                        display_at_end_boolean_criteria(
                            rating_criteria, truncated=False
                        )

                    st.markdown(f"{version_prompt.get('general_criteria_note', '')}")
                    st.markdown("### Evaluation Instruction:")
                    st.markdown(f"{version_prompt['rating_instruction']}")
        else:
            # Default to the preferred or latest version
            selected_versions = [preferred_prompt_info]
    else:
        # Automatically select the only available version
        selected_versions = filtered_prompts["prompt_version_info"].tolist()

    # Filter the selected versions
    selected_versions_df = filtered_prompts.loc[
        filtered_prompts["prompt_version_info"].isin(selected_versions)
    ]

    # Collect IDs and information of selected prompts
    prompt_ids.extend(selected_versions_df["id"].tolist())

    for _, current_prompt in selected_versions_df.iterrows():
        selected_prompts_info.append(
            {
                "Prompt": f"{current_prompt['prompt_title']} v{current_prompt['version']}",
                "Description": current_prompt["experiment_description"],
            }
        )

# Create and display the prompt table
if selected_prompts_info:
    prompt_table = pd.DataFrame(selected_prompts_info)
else:
    prompt_table = pd.DataFrame(columns=["Prompt", "Description"])

st.dataframe(prompt_table, hide_index=True, use_container_width=True)

# Dataset selection section
st.subheader("Dataset selection")
dataset_selected = st.multiselect(
    "Select datasets to run evaluation on:",
    samples_options,
    help="(Number of Lesson Plans in the Sample)",
)
# Filter samples_data based on the selected datasets
if dataset_selected:
    filtered_samples_data = samples_data[samples_data["samples_options"].isin(dataset_selected)]

    # Get sample IDs
    sample_ids = [
        filtered_samples_data[filtered_samples_data["samples_options"] == sample]["id"].iloc[0]
        for sample in dataset_selected
    ]

    # Create samples table for the selected datasets
    samples_table = pd.DataFrame(
        {
            "Sample": dataset_selected,
            ColumnLabels.NUM_LESSONS: [
                filtered_samples_data[filtered_samples_data["samples_options"] == sample]["number_of_lessons"].iloc[0]
                for sample in dataset_selected
            ],
        }
    )

    # Display the samples table
    st.dataframe(samples_table, hide_index=True, use_container_width=True)
else:
    st.warning("Please select at least one dataset to proceed.")

# Set parameters for batch processing
st.session_state.limit = (
    samples_table[ColumnLabels.NUM_LESSONS].max() if not samples_table.empty else 5
)

llm_model_options = [
    "gpt-4o-2024-05-13",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o",
    "gpt-4o-mini"
]

st.session_state.llm_model = st.selectbox(
    'Select a model:',
    llm_model_options,
    index=llm_model_options.index(st.session_state.llm_model)
)

st.session_state.llm_model_temp = st.number_input(
    "Enter temperature:",
    min_value=0.0,
    max_value=2.00,
    value=st.session_state.llm_model_temp,
    help="Minimum value is 0.0, maximum value is 2.00.",
)

st.session_state.top_p = st.number_input(
    "Enter top_p for the model:",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.top_p),
    step=0.01,
    help="Minimum value is 0.0, maximum value is 1.00.",
)

teachers_options = [OptionConstants.SELECT_TEACHER] + teachers_data["name"].tolist()

st.session_state.created_by = st.selectbox(
    "Who is running the experiment?",
    teachers_options,
    index=teachers_options.index(st.session_state.created_by),
)

teacher_id = None
if st.session_state.created_by != OptionConstants.SELECT_TEACHER:
    teacher_id = teachers_data[teachers_data["name"] == st.session_state.created_by][
        "id"
    ].iloc[0]

# Generate placeholders dynamically
placeholder_name, placeholder_description = generate_experiment_placeholders(
    st.session_state.llm_model,
    st.session_state.llm_model_temp,
    st.session_state.limit,
    len(prompt_ids),
    len(sample_ids),
    st.session_state.created_by,
)

tracked = True

with st.form(key="experiment_form"):
    st.subheader("Experiment information")
    experiment_name = st.text_input(
        "Enter experiment name:", value=placeholder_name, placeholder=placeholder_name
    )
    exp_description = st.text_input(
        "Enter experiment description:",
        value=placeholder_description,
        placeholder=placeholder_description,
    )
    batch_description = st.text_input(
        "Enter a description for your batch submission to identify it later:"
    )

    if st.form_submit_button("Submit batch"):
        experiment_id = add_to_batch(   
            experiment_name,
            exp_description,
            sample_ids,
            teacher_id,
            prompt_ids,
            st.session_state.limit,
            st.session_state.llm_model,
            tracked,
            st.session_state.llm_model_temp,
            st.session_state.top_p
        )

        # Verify data before submission
        if st.session_state.evaluations_list:
            st.write("Sample evaluation entry:")
            st.json(st.session_state.evaluations_list[0])
            st.json(st.session_state.evaluations_list[1])
            st.write(f"Total evaluations: {len(st.session_state.evaluations_list)}")
        else:
            st.error("No evaluations to submit. Please check your selections.")
            st.stop()

        # Convert the list of dictionaries to JSONL format in-memory
        jsonl_data = io.BytesIO()
        for entry in st.session_state.evaluations_list:
            jsonl_data.write((json.dumps(entry) + "\n").encode('utf-8'))
        jsonl_data.seek(0)  # Reset the pointer to the beginning of the BytesIO object

        # Upload the in-memory JSONL data to OpenAI
        batch_input_file = client.files.create(
            file=jsonl_data,
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        st.write("File uploaded with ID:", batch_input_file_id)

        # Create batch and capture the response
        try:
            batch_object = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": batch_description}
            )

            # Print the batch object for inspection
            st.write("Batch created successfully:")
            st.write(batch_object)

            # Access attributes directly
            batch_id = batch_object.id
            initial_status = batch_object.status

            st.write(f"Batch ID: {batch_id}")
            st.write(f"Initial Status: {initial_status}")

            # Retrieve and print full batch details using the ID
            batch_details = client.batches.retrieve(batch_id)
            st.write("Batch details:", batch_details)

        except OpenAIError as e:
            # Print detailed error message for troubleshooting
            st.write("Failed to create batch with error:", e.http_status, e.user_message)
            st.write("Error details:", e.json_body if hasattr(e, 'json_body') else "No details available")
        
        
        
        batch_id = batch_object.id
        batch_num_id = add_batch(batch_id, experiment_id, batch_description, st.session_state.created_by)
        st.success(
            f"Batch created with {len(st.session_state.evaluations_list)} experiments.\n\n"
            f"Batch submitted with ID: {batch_id}"
        )
        