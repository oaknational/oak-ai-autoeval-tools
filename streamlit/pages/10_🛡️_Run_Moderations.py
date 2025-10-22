""" 
Streamlit page for running moderations in the AutoEval app.
    
Functionality:
- Allows running moderations on a dataset using the moderation system.
- Results are stored in the database and can be viewed in the
    Visualise Results page.
"""

import pandas as pd
import streamlit as st
import json
import time
from typing import List, Dict, Any

from utils.common_utils import (
    clear_all_caches,
    log_message
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
    start_experiment,
    add_experiment,
    get_lesson_plans_by_id,
    add_results,
    insert_prompt,
    get_prompt,
    get_db_connection
)
from utils.moderation_utils import (
    moderate_lesson_plan,
    load_moderation_categories
)
from utils.constants import (
    OptionConstants,
    ColumnLabels,
    LessonPlanParameters,
)


# Set page configuration
st.set_page_config(page_title="Run Moderations", page_icon="🛡️")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

# Page and sidebar headers
st.markdown("# 🛡️ Run Moderations")
st.write(
    """
    This page allows you to run content moderation on a dataset using the 
    AI moderation system. Results will be stored in the database and can be 
    viewed in the Visualise Results page.
    """
)

# Initialize session state
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o-mini"
if "llm_model_temp" not in st.session_state:
    st.session_state.llm_model_temp = 0.7
if "limit" not in st.session_state:
    st.session_state.limit = 5
if "created_by" not in st.session_state:
    st.session_state.created_by = OptionConstants.SELECT_TEACHER
if "moderation_run" not in st.session_state:
    st.session_state.moderation_run = False

# Fetching data
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

# Load moderation categories
try:
    moderation_categories = load_moderation_categories()
    st.success(f"✅ Loaded {len(moderation_categories)} moderation category groups")
except Exception as e:
    st.error(f"❌ Failed to load moderation categories: {e}")
    st.stop()

def ensure_moderation_prompt_exists():
    """Ensure that a moderation prompt exists in the database.
    
    Returns:
        str: The ID of the moderation prompt
    """
    # Check if moderation prompt already exists
    prompts_data = get_prompts()
    moderation_prompts = prompts_data[prompts_data['objective_title'] == 'AILA Moderation']
    
    if not moderation_prompts.empty:
        # Return the ID of the first (preferred) moderation prompt
        preferred_prompt = moderation_prompts[moderation_prompts['preferred'] == True]
        if not preferred_prompt.empty:
            return preferred_prompt.iloc[0]['id']
        else:
            return moderation_prompts.iloc[0]['id']
    
    # Create a new moderation prompt if none exists
    try:
        insert_prompt(
            prompt_objective="Content moderation for lesson plans using AI-powered assessment across multiple safety and appropriateness categories.",
            lesson_plan_params=json.dumps({"all": "All lesson plan content"}),
            output_format="Score",
            rating_criteria=json.dumps({
                "5": "Not involved at all",
                "4": "Minimally involved", 
                "3": "Moderately involved",
                "2": "Significantly involved",
                "1": "Heavily involved"
            }),
            general_criteria_note="This prompt assesses lesson plans across multiple moderation categories including language, sensitive content, safety, and compliance issues.",
            rating_instruction="Provide a score from 1-5 for each moderation category, where 5 means the content is not involved at all and 1 means it's heavily involved.",
            prompt_title="AILA Content Moderation",
            experiment_description="AI-powered content moderation system for educational lesson plans",
            objective_title="AILA Moderation",
            objective_desc="Comprehensive content moderation across safety, appropriateness, and compliance categories",
            prompt_created_by="System",
            version="1.0",
            preferred=True
        )
        
        # Get the newly created prompt ID
        prompts_data = get_prompts()
        moderation_prompts = prompts_data[prompts_data['objective_title'] == 'AILA Moderation']
        if not moderation_prompts.empty:
            return moderation_prompts.iloc[0]['id']
        else:
            raise RuntimeError("Failed to create moderation prompt")
            
    except Exception as e:
        log_message("error", f"Failed to create moderation prompt: {e}")
        raise RuntimeError(f"Could not ensure moderation prompt exists: {e}")

# Ensure moderation prompt exists
try:
    moderation_prompt_id = ensure_moderation_prompt_exists()
    st.success(f"✅ Moderation prompt ready (ID: {moderation_prompt_id})")
except Exception as e:
    st.error(f"❌ Failed to setup moderation prompt: {e}")
    st.stop()

# Section: Moderation Configuration
st.subheader("Moderation Configuration")

# Display moderation categories info
with st.expander("View Moderation Categories"):
    st.markdown(f"**Total Categories:** {len(moderation_categories)}")
    st.markdown("**Available Categories:**")
    
    for category in moderation_categories:
        st.markdown(f"### {category['title']}")
        st.markdown(f"**Code:** {category['code']}")
        st.markdown(f"**Abbreviation:** {category['abbreviation']}")
        st.markdown(f"**Description:** {category['llmDescription']}")
        
        # Show criteria levels if available
        criteria_text = []
        for i in [5, 4, 3, 2, 1]:
            criteria_key = f"criteria{i}"
            if criteria_key in category and category[criteria_key]:
                criteria_text.append(f"**{i}:** {category[criteria_key]}")
        
        if criteria_text:
            st.markdown("**Rating Criteria:**")
            for criteria in criteria_text:
                st.markdown(f"- {criteria}")
        
        st.markdown("---")

# Dataset selection section
st.subheader("Dataset Selection")
sample_options = st.multiselect(
    "Select datasets to run moderation on:",
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

AVG_LATENCY = 3.5  # seconds for moderation (faster than evaluation)
total_time = total_sample_count * AVG_LATENCY
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

st.warning("A limit is advised to avoid long run times.")
st.warning(
    f"""
    Estimated time to run moderations without Limit: {int(hours)} hours,
    {int(minutes)} minutes, {int(seconds)} seconds
    """
)

# Set limit on lesson plans
st.session_state.limit = st.number_input(
    "Set a limit on the number of lesson plans per sample to moderate:",
    min_value=1,
    max_value=9000,
    value=max_lessons,
    help="Minimum value is 1.",
)

# Model configuration
llm_model_options = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gemini-2.5-pro-preview-05-06",
    "azure-openai",
]

st.session_state.llm_model = st.selectbox(
    'Select a model for moderation:',
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

teachers_options = [OptionConstants.SELECT_TEACHER] + teachers_data["name"].tolist()

st.session_state.created_by = st.selectbox(
    "Who is running the moderation?",
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
    1,  # Only one moderation prompt
    len(sample_ids),
    st.session_state.created_by,
)

# Modify placeholders for moderation - ensure "moderation" is in the name
placeholder_name = f"Moderation-{placeholder_name}"
placeholder_description = placeholder_description.replace("Evaluating", "Moderating")

# Category Selection Section
st.subheader("🛡️ Moderation Categories")
st.write("Select which moderation categories to include in the analysis. All categories are selected by default.")

try:
    # Load moderation categories
    from utils.moderation_utils import load_moderation_categories, process_moderation_categories
    
    categories_data = load_moderation_categories()
    processed_categories, _, _ = process_moderation_categories(categories_data)
    
    # Create a dictionary to store selected categories
    if 'selected_categories' not in st.session_state:
        # Initialize with all categories selected
        st.session_state.selected_categories = {cat['abbreviation']: True for cat in processed_categories}
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Available Categories:**")
        
        # Display all categories in a single expandable section
        with st.expander(f"📁 All Categories ({len(processed_categories)} total)"):
            for cat in processed_categories:
                abbreviation = cat['abbreviation']
                title = cat['title']
                user_desc = cat.get('userDescription', title)
                
                # Checkbox for each category
                selected = st.checkbox(
                    f"**{title}** ({abbreviation})",
                    value=st.session_state.selected_categories.get(abbreviation, True),
                    help=user_desc,
                    key=f"category_{abbreviation}"
                )
                st.session_state.selected_categories[abbreviation] = selected
    
    with col2:
        st.write("**Selection Summary:**")
        total_categories = len(processed_categories)
        selected_count = sum(1 for selected in st.session_state.selected_categories.values() if selected)
        
        st.metric("Total Categories", total_categories)
        st.metric("Selected Categories", selected_count)
        
        if selected_count == 0:
            st.error("⚠️ Please select at least one category to run moderation.")
        elif selected_count < total_categories:
            st.warning(f"⚠️ Only {selected_count} out of {total_categories} categories selected.")
        else:
            st.success("✅ All categories selected.")
        
        # Quick selection buttons
        st.write("**Quick Actions:**")
        if st.button("Select All", key="select_all_categories"):
            st.session_state.selected_categories = {cat['abbreviation']: True for cat in processed_categories}
            st.rerun()
        
        if st.button("Deselect All", key="deselect_all_categories"):
            st.session_state.selected_categories = {cat['abbreviation']: False for cat in processed_categories}
            st.rerun()

    # Prompt Preview Section
    st.subheader("🔍 Prompt Preview")
    if st.session_state.selected_categories:
        selected_list = [k for k, v in st.session_state.selected_categories.items() if v]
        if selected_list:
            try:
                # Get only the selected categories
                selected_categories_data = [
                    cat for cat in processed_categories 
                    if cat['abbreviation'] in selected_list
                ]
                
                # Generate the prompt that will be sent to LLM
                from utils.moderation_utils import generate_new_moderation_prompt_with_abbr
                preview_prompt = generate_new_moderation_prompt_with_abbr(selected_categories_data)
                
                st.write("**Prompt that will be sent to LLM:**")
                st.write(f"*Length: {len(preview_prompt)} characters*")
                st.write(f"*Categories included: {len(selected_categories_data)}*")
                
                with st.expander("📄 View Full Prompt", expanded=False):
                    st.code(preview_prompt, language="text")
                
                # Show a summary of what categories are included
                st.write("**Categories included in prompt:**")
                for cat in selected_categories_data:
                    st.write(f"- **{cat['title']}** ({cat['abbreviation']})")
                    
            except Exception as e:
                st.error(f"Failed to generate prompt preview: {e}")
        else:
            st.warning("Please select at least one category to see the prompt preview.")
    else:
        st.info("Select categories above to see the prompt preview.")

except Exception as e:
    st.error(f"Failed to load moderation categories: {e}")
    st.session_state.selected_categories = {}

tracked = st.selectbox("Should experiment be tracked?", options=["True", "False"])

def add_moderation_results(experiment_id: int, prompt_id: str, lesson_plan_id: str, scores_json: str, justification: str, status: str) -> None:
    """Add moderation results to the database with JSON scores.
    
    Args:
        experiment_id: ID of the experiment
        prompt_id: ID of the prompt used
        lesson_plan_id: ID of the lesson plan
        scores_json: JSON string containing individual category scores
        justification: Justification for the result
        status: Status of the result
    """
    try:
        conn = get_db_connection()
        if not conn:
            log_message("error", "Failed to connect to database")
            return
            
        cursor = conn.cursor()
        
        # Escape the JSON string for SQL
        if justification:
            justification = justification.replace("\\", "\\\\")  # Escape backslashes
            justification = justification.replace("'", "''")    # Escape single quotes for SQL
        
        insert_query = """
            INSERT INTO m_results (
                created_at, updated_at, experiment_id, prompt_id, 
                lesson_plan_id, result, justification, status)
            VALUES (now(), now(), %s, %s, %s, %s, %s, %s);
        """
        
        cursor.execute(insert_query, (
            experiment_id, prompt_id, lesson_plan_id, scores_json, justification, status
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        log_message("error", f"Failed to add moderation results: {e}")
        if conn:
            conn.rollback()
            conn.close()

def run_moderation_on_dataset(sample_id: str, limit: int, llm_model: str, llm_model_temp: float, experiment_id: int, prompt_id: str) -> Dict[str, Any]:
    """Run moderation on a dataset.
    
    Args:
        sample_id: ID of the sample to moderate
        limit: Maximum number of lesson plans to process
        llm_model: Model to use for moderation
        llm_model_temp: Temperature setting
        experiment_id: ID of the experiment
        prompt_id: ID of the moderation prompt
        
    Returns:
        Dictionary with moderation results summary
    """
    try:
        lesson_plans = get_lesson_plans_by_id(sample_id, limit)
        total_lessons = len(lesson_plans)
        
        if total_lessons == 0:
            return {"success": False, "error": "No lesson plans found"}
        
        successful_moderations = 0
        failed_moderations = 0
        flagged_lessons = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, lesson in enumerate(lesson_plans):
            try:
                lesson_plan_id = lesson[0]
                lesson_json_str = lesson[2]
                
                # Update progress
                progress = (i + 1) / total_lessons
                progress_bar.progress(progress)
                status_text.text(f"Moderating lesson {i+1}/{total_lessons} (ID: {lesson_plan_id})")
                
                # Run moderation with selected categories
                moderation_result = moderate_lesson_plan(
                    lesson_plan=lesson_json_str,
                    llm=llm_model,
                    temp=llm_model_temp,
                    selected_categories=st.session_state.selected_categories
                )
                
                # Check if any categories were flagged (score < 5)
                if moderation_result.flagged_categories:
                    flagged_lessons += 1
                
                # Store results in database
                # For moderation, we store the individual category scores as JSON
                scores_dict = moderation_result.scores.model_dump()
                if scores_dict:
                    # Store the individual category scores as JSON string
                    scores_json = json.dumps(scores_dict)
                    
                    # Create structured justification data
                    justification_data = {
                        "justifications": moderation_result.justifications,
                        "flagged_categories": moderation_result.flagged_categories,
                        "summary": "No categories flagged - all scores were 5 (not involved)" if not moderation_result.justifications else f"{len(moderation_result.justifications)} categories flagged"
                    }
                    
                    # Convert to JSON string for storage
                    justification_json = json.dumps(justification_data)
                    
                    # Store the result with the JSON scores using specialized function
                    add_moderation_results(
                        experiment_id=experiment_id,
                        prompt_id=prompt_id,
                        lesson_plan_id=lesson_plan_id,
                        scores_json=scores_json,  # Store JSON string with individual category scores
                        justification=justification_json,  # Store structured justification data
                        status="SUCCESS"
                    )
                    
                    # Log the individual scores for debugging
                    flagged_count = len([score for score in scores_dict.values() if score < 5])
                    log_message("info", f"Moderation completed for lesson {lesson_plan_id} - {flagged_count} categories flagged")
                else:
                    log_message("warning", f"No scores returned for lesson {lesson_plan_id}")
                    
                    # Create failure justification data
                    failure_justification_data = {
                        "justifications": {},
                        "flagged_categories": [],
                        "summary": "No moderation scores returned",
                        "error": "Failed to get scores from LLM"
                    }
                    
                    add_moderation_results(
                        experiment_id=experiment_id,
                        prompt_id=prompt_id,
                        lesson_plan_id=lesson_plan_id,
                        scores_json="{}",  # Empty JSON object for no scores
                        justification=json.dumps(failure_justification_data),
                        status="FAILURE"
                    )
                
                successful_moderations += 1
                
            except Exception as e:
                log_message("error", f"Failed to moderate lesson {lesson_plan_id}: {e}")
                
                # Store error result
                error_justification_data = {
                    "justifications": {},
                    "flagged_categories": [],
                    "summary": f"Moderation failed: {str(e)}",
                    "error": str(e)
                }
                
                add_moderation_results(
                    experiment_id=experiment_id,
                    prompt_id=prompt_id,
                    lesson_plan_id=lesson_plan_id,
                    scores_json="{}",
                    justification=json.dumps(error_justification_data),
                    status="FAILURE"
                )
                
                failed_moderations += 1
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return {
            "success": True,
            "total_lessons": total_lessons,
            "successful_moderations": successful_moderations,
            "failed_moderations": failed_moderations,
            "flagged_lessons": flagged_lessons
        }
        
    except Exception as e:
        log_message("error", f"Error in moderation process: {e}")
        return {"success": False, "error": str(e)}

with st.form(key="moderation_form"):
    st.subheader("Moderation Information")
    experiment_name = st.text_input(
        "Enter experiment name:", 
        value=placeholder_name, 
        placeholder=placeholder_name,
        help="⚠️ The experiment name must contain 'moderation' to be found by the visualization page."
    )
    exp_description = st.text_input(
        "Enter experiment description:",
        value=placeholder_description,
        placeholder=placeholder_description,
    )

    if st.form_submit_button("Run Moderation"):
        if not sample_options:
            st.error("Please select at least one dataset to moderate.")
        elif st.session_state.created_by == OptionConstants.SELECT_TEACHER:
            st.error("Please select who is running the moderation.")
        elif not st.session_state.selected_categories or not any(st.session_state.selected_categories.values()):
            st.error("Please select at least one moderation category to run.")
        elif "moderation" not in experiment_name.lower():
            st.error("Experiment name must contain 'moderation' to be found by the visualization page.")
        else:
            st.warning("Please do not close the page until the moderation is complete.")
            
            # Create experiment record without running tests (we'll handle moderation separately)
            from utils.db_scripts import add_experiment
            experiment_id = add_experiment(
                experiment_name,
                sample_ids,
                teacher_id,
                tracked,
                st.session_state.llm_model,
                st.session_state.llm_model_temp,
                exp_description,
                "PENDING"
            )
            
            if experiment_id:
                st.success(f"Experiment created with ID: {experiment_id}")
                
                # Run moderation on each selected dataset
                total_results = {
                    "total_lessons": 0,
                    "successful_moderations": 0,
                    "failed_moderations": 0,
                    "flagged_lessons": 0
                }
                
                for sample_id in sample_ids:
                    st.markdown(f"### Moderating dataset: {sample_id}")
                    result = run_moderation_on_dataset(
                        sample_id,
                        st.session_state.limit,
                        st.session_state.llm_model,
                        st.session_state.llm_model_temp,
                        experiment_id,
                        moderation_prompt_id
                    )
                    
                    if result["success"]:
                        total_results["total_lessons"] += result["total_lessons"]
                        total_results["successful_moderations"] += result["successful_moderations"]
                        total_results["failed_moderations"] += result["failed_moderations"]
                        total_results["flagged_lessons"] += result["flagged_lessons"]
                        
                        st.success(f"✅ Completed moderation for dataset {sample_id}")
                    else:
                        st.error(f"❌ Failed to moderate dataset {sample_id}: {result.get('error', 'Unknown error')}")
                
                # Display final results
                st.markdown("### Moderation Results Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Lessons", total_results["total_lessons"])
                with col2:
                    st.metric("Successful Moderations", total_results["successful_moderations"])
                with col3:
                    st.metric("Failed Moderations", total_results["failed_moderations"])
                with col4:
                    st.metric("Flagged Lessons", total_results["flagged_lessons"])
                
                if total_results["flagged_lessons"] > 0:
                    st.warning(f"⚠️ {total_results['flagged_lessons']} lessons were flagged for content issues.")
                
                st.session_state.moderation_run = True
            else:
                st.error("Failed to create experiment. Please check the logs for details.")

if st.session_state.moderation_run:
    st.write("**Click the button to view insights.**")
    if st.button("View Insights"):
        st.switch_page("pages/4_🔍_Visualise_Results.py")
