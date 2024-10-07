""" 
Streamlit page for running evaluations in the AutoEval app.
    
Functionality:
- Allows running evaluations on a dataset using selected prompts.
- Results are stored in the database and can be viewed in the
    Visualise Results page.
"""

import pandas as pd
import streamlit as st
import json


from utils.common_utils import (
    clear_all_caches
)
from utils.formatting import (
    generate_experiment_placeholders,
    lesson_plan_parts_at_end,
    display_at_end_score_criteria,
    display_at_end_boolean_criteria
    )
from utils.db_scripts import (
    get_prompts,
    get_samples,
    get_teachers,
    start_experiment)

from utils.constants import (
    OptionConstants,
    ColumnLabels,
    LessonPlanParameters,
)


# Set page configuration
st.set_page_config(page_title="Run Auto Evaluations", page_icon="🤖")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

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
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o"
if "llm_model_temp" not in st.session_state:
    st.session_state.llm_model_temp = 0.5
if "limit" not in st.session_state:
    st.session_state.limit = 5
if "created_by" not in st.session_state:
    st.session_state.created_by = OptionConstants.SELECT_TEACHER
if "experiment_run" not in st.session_state:
    st.session_state.experiment_run = False

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
sample_options = st.multiselect(
    "Select datasets to run evaluation on:",
    samples_options,
    help="(Number of Lesson Plans in the Sample)",
)
samples_data = samples_data[(samples_data["samples_options"].isin(sample_options))]

# Get sample IDs
sample_ids = [
    samples_data[samples_data["samples_options"] == sample]["id"].iloc[0]
    for sample in sample_options
]

# Create samples table
samples_table = pd.DataFrame(
    {
        "Sample": sample_options,
        ColumnLabels.NUM_LESSONS: [
            samples_data[samples_data["samples_options"] == sample][
                "number_of_lessons"
            ].iloc[0]
            for sample in sample_options
        ],
    }
)

st.dataframe(samples_table, hide_index=True, use_container_width=True)

# Calculate time estimates and set limits
max_lessons = (
    samples_table[ColumnLabels.NUM_LESSONS].max() if not samples_table.empty else 5
)

total_sample_count = (
    samples_table[ColumnLabels.NUM_LESSONS].sum() if not samples_table.empty else 0
)
total_prompt_count = prompt_table.shape[0] if not prompt_table.empty else 0

AVG_LATENCY = 7.78  # seconds
total_time = total_sample_count * total_prompt_count * AVG_LATENCY
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

st.warning("A limit is advised to avoid long run times.")
st.warning(
    f"""
    Estimated time to run evaluations without Limit: {int(hours)} hours,
    {int(minutes)} minutes, {int(seconds)} seconds
    """
)

# Set limit on lesson plans
st.session_state.limit = st.number_input(
    "Set a limit on the number of lesson plans per sample to evaluate:",
    min_value=1,
    max_value=9000,
    value=max_lessons,
    help="Minimum value is 1.",
)

llm_model_options = [
    'o1-preview-2024-09-12','o1-mini-2024-09-12',
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "llama",
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

if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0


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

tracked = st.selectbox("Should experiment be tracked?", options=["True", "False"])

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

    if st.form_submit_button("Run evaluation"):
        st.warning("Please do not close the page until the evaluation is complete.")
        experiment_complete = start_experiment(
            experiment_name,
            exp_description,
            sample_ids,
            teacher_id,
            prompt_ids,
            st.session_state.limit,
            st.session_state.llm_model,
            tracked,
            st.session_state.llm_model_temp,
            st.session_state.top_p,
        )

        if experiment_complete:
            st.session_state.experiment_run = True
        else:
            st.error(
                "Experiment failed to complete. " "Please check the logs for details."
            )

if st.session_state.experiment_run:
    st.write("**Click the button to view insights.**")
    if st.button("View Insights"):
        st.switch_page("pages/4_🔍_Visualise_Results.py")
