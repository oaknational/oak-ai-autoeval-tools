""" 
Streamlit page for checking batches of evaluations have completed 
processing by OpenAI.
    
Functionality:
- xxxx.

- xxxx.
"""
import io
import json

import pandas as pd
import streamlit as st
from openai import OpenAI

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
    get_batches,
    get_prompts,
    get_samples,
    get_teachers,
    add_experiment,
    get_lesson_plans_by_id,
    get_prompt
)
from utils.constants import (
    OptionConstants,
    ColumnLabels,
    LessonPlanParameters
)

# Initialize the OpenAI client
client = OpenAI()

# Set page configuration
st.set_page_config(page_title="Batch Results", page_icon="🤖")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

# Page and sidebar headers
st.markdown("# 🤖 Batch Results Checker")
st.write(
    """
    This page allows you to check whether batches of evaluations have completed 
    processing by OpenAI.
    """
)

# Fetching data
batches_data = get_batches()

# Order batches_data by created_at
batches_data = batches_data.sort_values(by="created_at", ascending=False)

batches_data["batches_options"] = (
    batches_data["batch_ref"]
    + " -- "
    + batches_data["batch_description"]
    + " -- "
    + batches_data["created_by"]
)
batches_options = batches_data["batches_options"].tolist()
batches_options.insert(0, " ")


# Batch selection section
st.subheader("Batch selection")
batches_options = st.selectbox(
    "Select pending batch to check status:",
    batches_options
)

#batches_data = batches_data[(batches_data["batches_options"].isin(batches_options))]
#st.write(batches_data)


#batch_id = "batch_6707df91cd608190ba0a9a7ff3283dd5"

# Check status of batch job
#st.write(client.batches.retrieve(batch_id))
        
        