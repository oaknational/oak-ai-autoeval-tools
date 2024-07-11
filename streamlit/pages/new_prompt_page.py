"""
This script is for creating and managing prompt tests using Streamlit.
It allows users to either create new prompts from scratch with guidance or modify existing prompts.

Modules and libraries used:
- streamlit: For creating the web interface
- psycopg2: For connecting to the PostgreSQL database
- pandas: For data manipulation and analysis
- os and dotenv: For loading environment variables
- json: For handling JSON data
- datetime: For working with dates and times
- numpy: For numerical operations
- jinja_funcs: Custom module with utility functions (get_teachers, to_prompt_metadata_db)
- re: For regular expression operations

The script sets up the Streamlit page configuration, loads environment variables,
and initialises database connection parameters.
"""

import streamlit as st
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np
from jinja_funcs import get_teachers, to_prompt_metadata_db
import re

st.set_page_config(page_title="Create Prompt Tests", page_icon="📝")
st.markdown("# 📝 Create Prompt Tests")

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Duplicate strings
description_1 = 'Description for 1'
description_5 = 'Description for 5'
description_true = 'Description for TRUE'
description_false = 'Description for FALSE'
general_criteria_note_text = (
    "Either leave this section empty or add things you'd like the LLM to focus on"
)
rating_instruction_text = "Tell the LLM to actually do the evaluation"
teacher_option_text = 'Select a teacher'
created_by_text = 'Who is creating the prompt?'
prompt_objective_text = 'State what you want the LLM to check for'
output_format_text = (
    "Choose 'Score' for a Likert scale rating (1-5) or 'Boolean' for a TRUE/FALSE"
    " evaluation"
)

# Headers
prompt_title_header = "#### Prompt Title"
prompt_objective_header = "#### Prompt Objective"
lesson_plan_params_header = "#### Relevant Lesson Plan Parts"
output_format_header = "#### Output Format"
rating_criteria_header = "#### Rating Criteria"
evaluation_criteria_header = "#### Evaluation Criteria"
general_criteria_note_header = "#### General Criteria Note"
rating_instruction_header = "#### Rating Instruction"
experiment_description_header = "#### Experiment Description"
objective_title_header = "#### Prompt Group"

# Sidebar Headers
objective_sb_header = "### Objective:"
rating_criteria_sb_header = "### Rating Criteria:"
evaluation_criteria_sb_header = "### Evaluation Criteria:"
rating_instruction_sb_header = "### Rating Instruction:"
evaluation_instruction_sb_header = "### Evaluation Instruction:"

# Function to clear cache
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

def get_all_prompts():
    """
    Connects to the PostgreSQL database and retrieves all prompts from the 'm_prompts' table.
    The function returns the data as a pandas DataFrame with appropriate column names and parses 
    the 'rating_criteria' column from JSON strings to Python dictionaries.

    Returns:
        pd.DataFrame: A DataFrame containing all the prompts from the 'm_prompts' table.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    query = """
    SELECT id, prompt_objective, lesson_plan_params, output_format, rating_criteria, general_criteria_note, rating_instruction, encode(prompt_hash, 'hex'), prompt_title, experiment_description, objective_title, objective_desc, created_at, created_by, version
    FROM public.m_prompts;
    """
    cur = conn.cursor()
    cur.execute(query)
    data = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()
    conn.close()
    
    # Parse JSON fields safely
    data['rating_criteria'] = data['rating_criteria'].apply(lambda x: json.loads(x) if x else {})
    
    return data

def check_prompt_title_exists(prompt_title):
    """
    Checks if a prompt title exists in the 'm_prompts' table of the PostgreSQL database.

    Args:
        prompt_title (str): The prompt title to check for existence in the database.

    Returns:
        bool: True if the prompt title exists, False otherwise.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    query = """
    SELECT COUNT(*)
    FROM public.m_prompts
    WHERE prompt_title = %s;
    """
    cur = conn.cursor()
    cur.execute(query, (prompt_title,))
    exists = cur.fetchone()[0] > 0
    cur.close()
    conn.close()
    return exists

def show_rating_criteria_input(output_format, new=False, current_prompt=None):
    """
    Displays input fields for rating criteria based on the given output format. The function 
    initialises the criteria either as new or based on an existing prompt and allows the user 
    to update the criteria through input fields.

    Args:
        output_format (str): The format of the output, either 'Score' or 'Boolean'.
        new (bool): Indicates whether the criteria are new or based on an existing prompt.
        current_prompt (dict): The existing prompt data, used when new is False.

    Returns:
        dict: The updated rating criteria.
    """
    if output_format == 'Score':
        st.markdown(rating_criteria_header)
        st.markdown("Please make 5 the ideal score and 1 the worst score.")
        # Initialise placeholders for the rating criteria
        rating_criteria_placeholder = st.empty()
        
        # Determine the initial values based on whether it's new or existing
        if new:
            rating_criteria = {
                '5 ()': "",
                '1 ()': ""
            }
            label_5 = ""
            desc_5 = ""
            label_1 = ""
            desc_1 = ""
        
        else:
            # Parse the current rating criteria
            current_rating_criteria = current_prompt['rating_criteria']
            label_5 = list(current_rating_criteria.keys())[0].split('(')[-1].strip(')')
            desc_5 = list(current_rating_criteria.values())[0]
            label_1 = list(current_rating_criteria.keys())[1].split('(')[-1].strip(')')
            desc_1 = list(current_rating_criteria.values())[1]
            rating_criteria = current_rating_criteria
        
        # Display the initial rating criteria
        rating_criteria_placeholder.json(rating_criteria)
        
        # Input fields for labels and descriptions
        label_5 = st.text_input('Label for 5', value=label_5, key="label_5")
        desc_5 = st.text_area(description_5, value=desc_5, key="desc_5", height=50)

        label_1 = st.text_input('Label for 1', value=label_1, key="label_1")
        desc_1 = st.text_area(description_1, value=desc_1, key="desc_1", height=50)
        
        
        # Update the rating criteria based on user input
        rating_criteria = {
            f'5 ({label_5})': desc_5,
            f'1 ({label_1})': desc_1
        }

        # Update the rating criteria placeholder with the new values
        rating_criteria_placeholder.json(rating_criteria)
    elif output_format == 'Boolean':
        st.markdown(evaluation_criteria_header)
        st.markdown("Please make TRUE the ideal output")
        
        # Initialise placeholders for the rating criteria
        rating_criteria_placeholder = st.empty()
        
        # Determine the initial values based on whether it's new or existing
        if new:
            rating_criteria = {
                'TRUE': "",
                'FALSE': ""
            }
            desc_t = ""
            desc_f = ""
        else:
            # Parse the current rating criteria
            current_rating_criteria = current_prompt['rating_criteria']
            desc_t = current_rating_criteria.get('TRUE', "")
            desc_f = current_rating_criteria.get('FALSE', "")
            rating_criteria = current_rating_criteria
        
        # Display the initial rating criteria
        rating_criteria_placeholder.json(rating_criteria)

        # Input fields for labels and descriptions
        desc_t = st.text_area(description_true, value=desc_t, key="desc_t", height=50)
        desc_f = st.text_area(description_false, value=desc_f, key="desc_f", height=50)
        
        # Update the rating criteria based on user input
        rating_criteria = {
            'TRUE': desc_t,
            'FALSE': desc_f
        }

        # Update the rating criteria placeholder with the new values
        rating_criteria_placeholder.json(rating_criteria)
        
    return rating_criteria

def objective_title_select(new = False, current_prompt=None):
    """
    Displays input fields for selecting or entering the objective title and description based on whether
    the prompt is new or existing. 

    Args:
        new (bool): Indicates whether the prompt is new or existing.
        current_prompt (dict): The existing prompt data, used when new is False.

    Returns:
        tuple: A tuple containing the objective title and description.
    """
    st.markdown(objective_title_header)

    if new:
        objective = st.selectbox("Select the group that the prompt belongs to", ["Sanity Checks - Check if the lesson is up to oak standards", "Low-quality Content - Check for low-quality content in the lesson plans", "Moderation Eval - Check for moderation flags in the lesson plans", "New Group"])
        
        if objective == "New Group":
            objective_title = st.text_input("Enter the new group name", value="")
            objective_desc = st.text_area("Enter the description for the new group e.g. Check if the lesson is up to oak standards", value="", height=100)
        
        if objective == "Sanity Checks - Check if the lesson is up to oak standards":
            objective_title = "Sanity Checks"
            objective_desc = "Check if the lesson is up to oak standards."
        
        if objective == "Low-quality Content - Check for low-quality content in the lesson plans":
            objective_title = "Low-quality Content"
            objective_desc = "Check for low-quality content in the lesson plans."

        if objective == "Moderation Eval - Check for moderation flags in the lesson plans":
            objective_title = "Moderation Eval"
            objective_desc = "Check for moderation flags in the lesson plans"

        return objective_title, objective_desc
    else:
        objective_title = current_prompt['objective_title']
        objective_desc = current_prompt['objective_desc']

        st.markdown(f"{objective_title} - {objective_desc}")

        return objective_title, objective_desc

def display_sidebar_score_criteria():
    """
    Display rating criteria for 'Score' output format in the sidebar.
    """
    st.sidebar.markdown(rating_criteria_sb_header)

    label_5 = list(rating_criteria.keys())[0].split('(')[-1].strip(')')
    desc_5 = list(rating_criteria.values())[0]
    desc_5_short = get_first_ten_words(desc_5)
    st.sidebar.markdown(f"**5 ({label_5}):** {desc_5_short}")
    
    # Extract and display label and description for 1
    label_1 = list(rating_criteria.keys())[1].split('(')[-1].strip(')')
    desc_1 = list(rating_criteria.values())[1]
    desc_1_short = get_first_ten_words(desc_1)
    st.sidebar.markdown(f"**1 ({label_1}):** {desc_1_short}")

def display_sidebar_boolean_criteria():
    """
    Display rating criteria for 'Boolean' output format in the sidebar.
    """
    st.sidebar.markdown(evaluation_criteria_sb_header)

    desc_true_short = get_first_ten_words(rating_criteria['TRUE'])
    desc_false_short = get_first_ten_words(rating_criteria['FALSE'])
    st.sidebar.markdown(f"**TRUE:** {desc_true_short}")
    st.sidebar.markdown(f"**FALSE:** {desc_false_short}")

def example_score_rating_criteria():
    """
    Display example rating criteria for 'Score' output format in an expander.
    """
    with st.expander("Example"):
        example_rating_criteria = example_prompt_score['rating_criteria']
        label_5 = list(example_rating_criteria.keys())[0].split('(')[-1].strip(')')
        desc_5 = list(example_rating_criteria.values())[0]
        label_1 = list(example_rating_criteria.keys())[1].split('(')[-1].strip(')')
        desc_1 = list(example_rating_criteria.values())[1]
        
        st.write(f"**Label for 1**: {label_1}")
        st.write(f"**Description for 1**: {desc_1}")
        st.write(f"**Label for 5**: {label_5}")
        st.write(f"**Description for 5**: {desc_5}")

def example_boolean_rating_criteria():
    """
    Display example rating criteria for 'Boolean' output format in an expander.
    """
    with st.expander("Example"):
        example_rating_criteria = example_prompt_boolean['rating_criteria']
        desc_t = example_rating_criteria.get('TRUE', "")
        desc_f = example_rating_criteria.get('FALSE', "")
        
        st.write(f"**Description for TRUE**: {desc_t}")
        st.write(f"**Description for FALSE**: {desc_f}")

def save_prompt():
    return
       
def get_lesson_plan_params(plain_eng_list):
    """
    Maps a list of plain English lesson plan parameter names to their corresponding keys used in the system.

    Args:
        plain_eng_list (list of str): A list of lesson plan parameter names in plain English.

    Returns:
        list of str: A list of corresponding keys for the lesson plan parameters.
    """
    # Define the mapping dictionary within the function
    lesson_params_to_titles = dict(zip(lesson_params_plain_eng, lesson_params))
    
    # Generate the list of corresponding keys
    acc_params = [lesson_params_to_titles[item] for item in plain_eng_list if item in lesson_params_to_titles]
    
    return acc_params

def lesson_plan_parts_sidebar(lesson_plan_params):
    """
    Generates a formatted string for displaying lesson plan parts in the sidebar.
    The function maps lesson plan parameters to their titles and formats them for display.

    Args:
        lesson_plan_params (list or str): A list of lesson plan parameters or a JSON string representing the list.

    Returns:
        str: A formatted string with lesson plan parts for sidebar display.
    """
    # Create the mapping dictionary using zip
    lesson_params_to_titles = dict(zip(lesson_params, lesson_params_titles))
    
    # Parse string input to list if necessary
    if isinstance(lesson_plan_params, str):
        lesson_plan_params = json.loads(lesson_plan_params)
    
    # Generate the output
    return "\n".join(
        f"### {lesson_params_to_titles.get(param, param)}:\n*insert {param} here*\n### _(End of {lesson_params_to_titles.get(param, param)})_\n"
        for param in lesson_plan_params
    )

def get_first_ten_words(text):
    """
    Extracts the first ten words from a given text and appends an ellipsis ('...') 
    if there are more than ten words.

    Args:
        text (str): The input text from which to extract the first ten words.

    Returns:
        str: A string containing the first ten words followed by an ellipsis if the 
             original text has more than ten words, otherwise returns the original text.
    """
    words = text.split()
    first_ten_words = ' '.join(words[:10]) + '...' if len(words) > 10 else text
    return first_ten_words

def fetch_prompt_details_by_id(prompt_id):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    query = """
    SELECT prompt_title, prompt_objective, lesson_plan_params, output_format,
           rating_criteria, general_criteria_note, rating_instruction, experiment_description,
           objective_title, objective_desc, created_by
    FROM public.m_prompts
    WHERE id = %s;
    """
    cur = conn.cursor()
    cur.execute(query, (prompt_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        columns = ["prompt_title", "prompt_objective", "lesson_plan_params", "output_format",
                   "rating_criteria", "general_criteria_note", "rating_instruction", "experiment_description",
                   "objective_title", "objective_desc", "created_by"]
        prompt_data = pd.Series(result, index=columns)

        # Parse JSON fields safely
        prompt_data['rating_criteria'] = json.loads(prompt_data['rating_criteria']) if prompt_data['rating_criteria'] else {}
        prompt_data['lesson_plan_params'] = json.loads(prompt_data['lesson_plan_params']) if prompt_data['lesson_plan_params'] else []

        return prompt_data
    else:
        return None

# Fetch example prompt data
# Quiz Qs Require Explicit Knowledge
example_prompt_id_score = "fa57a7ca-604c-462d-b4e0-0d43b17b691d"
example_prompt_score = fetch_prompt_details_by_id(example_prompt_id_score)

# Learning Cycles Increase in Challenge
example_prompt_id_boolean = "872592e3-ba7a-408d-9995-a66f056b1ed3"
example_prompt_boolean = fetch_prompt_details_by_id(example_prompt_id_boolean)

# Lesson parameters and their corresponding titles (for display sidebar
# purposes) and plain English descriptions
lesson_params = ['lesson', 'title', 'topic', 'subject', 'cycles', 
                 'cycle_titles', 'cycle_feedback', 'cycle_practice',
                 'cycle_explanations', 'cycle_spokenexplanations',
                 'cycle_accompanyingslidedetails', 'cycle_imageprompts',
                 'cycle_slidetext', 'cycle_durationinmins',
                 'cycle_checkforunderstandings', 'cycle_scripts', 'exitQuiz',
                 'keyStage','starterQuiz', 'learningCycles', 'misconceptions',
                 'priorKnowledge', 'learningOutcome', 'keyLearningPoints',
                 'additionalMaterials']

lesson_params_titles = ['Lesson', 'Title', 'Topic', 'Subject', 'Cycles',
                        'Titles', 'Feedback', 'Practice Tasks', 'Explanations',
                        'Spoken Explanations', 'Accompanying Slide Details',
                        'Image Prompts', 'Slide Text', 'Duration in Minutes',
                        'Check for Understandings', 'Scripts', 'Exit Quiz',
                        'Key Stage', 'Starter Quiz', 'Learning Cycles',
                        'Misconceptions', 'Prior Knowledge',
                        'Learning Outcome', 'Key Learning Points',
                        'Additional Materials']

lesson_params_plain_eng = ['Whole lesson', 'Title', 'Topic', 'Subject',
                           'All content from all cycles', 'All cycle titles',
                           'All cycle feedback', 'All cycle practice',
                           'Entire explanations from all cycles',
                           'All spoken explanations from all cycles',
                           'All accompanying slide details from all cycles',
                           'All image prompts from all cycles',
                           'All slide text from all cycles',
                           'All durations in minutes from all cycles',
                           'All check for understandings from all cycles',
                           'All scripts from all cycles', 'Exit Quiz',
                           'Key Stage', 'Starter Quiz', 'Learning cycles',
                           'Misconceptions', 'Prior knowledge',
                           'Learning outcomes', 'Key learning points',
                           'Additional materials']

#The following function isn't in use yet, wasn't working as expected
def get_lesson_plan_plain_eng(proper_lesson_params):
    """
    Maps a list of proper lesson plan parameters to their plain English counterparts.

    Args:
        proper_lesson_params (list of str): A list of proper lesson plan parameters.

    Returns:
        list of str: A list of plain English names corresponding to the given proper lesson plan parameters.
    """
    # Create a dictionary from lesson_params_plain_eng to lesson_params
    dict_lesson_params = dict(zip(lesson_params_plain_eng, lesson_params))
    acc_params = []

    # Loop through the lesson_params and map them to their plain English counterparts
    for i in proper_lesson_params:
        for key, value in dict_lesson_params.items():
            if i == value:
                acc_params.append(key) 
    
    return acc_params

# Retrieve all prompt data from the database
data = get_all_prompts()

filtered_data = data

# Display a header in the sidebar
st.sidebar.markdown("# Your Prompt")

# Display a dropdown menu for the user to select an action
action = st.selectbox("What would you like to do?", [" ", "Create a new prompt", "Modify an existing prompt"])

if action == "Create a new prompt":
    # Display the header for prompt title input
    st.markdown(prompt_title_header)

    # Input field for the prompt title
    prompt_title = st.text_input('Choose a unique title for your prompt', value='')
    
    # Display the header for prompt objective input
    st.markdown(prompt_objective_header)

    # Text area for the prompt objective
    prompt_objective = st.text_area(prompt_objective_text, value="", height=200)
    
    # Display the objective in the sidebar
    st.sidebar.markdown(objective_sb_header)
    truncated_prompt_objective = get_first_ten_words(prompt_objective)
    st.sidebar.markdown(f"{truncated_prompt_objective}")

    # Display an example prompt objective
    with st.expander("Example"):
        st.write(f"{example_prompt_score['prompt_objective']}")

    # Display the header for lesson plan parameters input
    st.markdown(lesson_plan_params_header)

    # Multiselect for lesson plan parameters
    lesson_plan_params_st = st.multiselect("Choose the parts of the lesson plan that you're evaluating", options=lesson_params_plain_eng)

    # Get the lesson plan parameters in the required format
    lesson_plan_params = get_lesson_plan_params(lesson_plan_params_st)

    # Generate and display the lesson plan parameters in the sidebar
    output = lesson_plan_parts_sidebar(lesson_plan_params)
    st.sidebar.markdown(output)

    # Display the header for output format selection
    st.markdown(output_format_header)

    # Select box for output format
    output_format = st.selectbox(output_format_text, options=[' ', 'Score', 'Boolean'])

    if output_format != ' ':
        # Select the appropriate example prompt based on output_format
        if output_format == 'Score':
            example_prompt = example_prompt_score
        elif output_format == 'Boolean':
            example_prompt = example_prompt_boolean

        # Display input fields for rating criteria based on output format
        rating_criteria = show_rating_criteria_input(output_format, new=True)
        
        # Display the rating criteria in the sidebar and example rating criteria as an expander
        if output_format == 'Score':
            display_sidebar_score_criteria()
            example_score_rating_criteria()

        elif output_format == 'Boolean':
            display_sidebar_boolean_criteria()
            example_boolean_rating_criteria()

        # Display and input for general criteria note
        st.markdown(general_criteria_note_header)
        general_criteria_note = st.text_area(general_criteria_note_text, value="", height=100)

        # Display example general criteria note in an expander
        with st.expander("Example"):
            st.write(f"{example_prompt['general_criteria_note']}")

        # Display truncated general criteria note in the sidebar
        truncated_general_criteria_note = get_first_ten_words(general_criteria_note)
        st.sidebar.markdown(f"{truncated_general_criteria_note}")

        # Display and input for rating instruction
        st.markdown(rating_instruction_header)
        rating_instruction = st.text_area(rating_instruction_text, value="", height=100)

        # Display example rating instruction
        with st.expander("Example"):
            st.write(f"{example_prompt['rating_instruction']}")
        
        # Display truncated rating instruction in the sidebar
        truncated_rating_instruction = get_first_ten_words(rating_instruction)
        if output_format == 'Score':
            st.sidebar.markdown(rating_instruction_sb_header)
            st.sidebar.markdown(f"{truncated_rating_instruction}")
        elif output_format == 'Boolean':
            st.sidebar.markdown(evaluation_instruction_sb_header)
            st.sidebar.markdown(f"{truncated_rating_instruction}")

        # Leave experiment description empty (redundant field)
        experiment_description = ' '

        # Select or input objective title and description
        objective_title, objective_desc = objective_title_select(new=True)

        # Get and display list of teachers for selection
        teachers = get_teachers()
        teachers_options = teachers['name'].tolist()
        teacher_option = [teacher_option_text] + teachers_options
        created_by = st.selectbox(created_by_text, teacher_option)

        # Save the new prompt to the database when the button is clicked
        if st.button("Save New Prompt", help="Save the new prompt to the database."):
            if check_prompt_title_exists(prompt_title):
                st.error("This name already exists. Choose another one.")
            else:
                
                returned_id = to_prompt_metadata_db(
                    prompt_objective, 
                    json.dumps(lesson_plan_params),
                    output_format, 
                    rating_criteria,
                    general_criteria_note, 
                    rating_instruction, 
                    prompt_title, 
                    experiment_description, 
                    objective_title, 
                    objective_desc, 
                    created_by, 
                    '1' 
                )

                st.success(f"New prompt created successfully! With ID: {returned_id}")

if action == "Modify an existing prompt":
    # Get a list of unique prompt titles and insert a blank option at the beginning
    prompt_title_options = filtered_data['prompt_title'].unique().tolist()
    prompt_title_options.insert(0, '') 
    prompt_title = st.selectbox('Select an existing prompt to modify:', prompt_title_options)

    # Track the selected prompt in the session state
    if 'selected_prompt' not in st.session_state:
        st.session_state['selected_prompt'] = prompt_title

    # Check if the selected prompt has changed
    if st.session_state['selected_prompt'] != prompt_title:
        st.session_state['selected_prompt'] = prompt_title
        st.session_state['refresh'] = True

    if prompt_title != '':
        # Filter the data for the selected prompt title
        filtered_data = filtered_data[filtered_data['prompt_title'] == prompt_title]

        if not filtered_data.empty:
            # Get the latest version of the selected prompt
            latest_prompt = filtered_data.loc[filtered_data['created_at'].idxmax()]

            current_prompt = filtered_data.loc[filtered_data['created_at'].idxmax()]
            prompt_id = current_prompt['id']

            # Display the key details of the current prompt in a table
            st.table(current_prompt[['created_at', 'prompt_title', 'prompt_objective', 'output_format', 'created_by', 'version']])

            # Initialise or refresh the draft prompt in session state
            st.session_state['draft_prompt'] = current_prompt.copy(deep=True)
            st.session_state['refresh'] = False

            # Display the non-editable prompt title
            st.markdown(prompt_title_header)
            st.markdown(f"{current_prompt['prompt_title']}")

            # Display the header for prompt objective
            st.markdown(prompt_objective_header)

            # Text area for the prompt objective, initialised with the current prompt's objective
            prompt_objective = st.text_area(prompt_objective_text, value=current_prompt['prompt_objective'], height=100)

            # Update the prompt objective in the session state
            st.session_state['draft_prompt']['prompt_objective'] = prompt_objective

            # Display the objective in the sidebar
            truncated_prompt_objective = get_first_ten_words(prompt_objective)
            st.sidebar.markdown(objective_sb_header)
            st.sidebar.markdown(f"{truncated_prompt_objective}")

            # Display the non-editable lesson plan parameters
            st.markdown(lesson_plan_params_header)
            lesson_plan_params = current_prompt['lesson_plan_params']
            st.markdown(f"{lesson_plan_params}")

            # Generate and display the lesson plan parameters in the sidebar
            output = lesson_plan_parts_sidebar(lesson_plan_params)
            st.sidebar.markdown(output)
            
            # Display the header for output format selection
            st.markdown(output_format_header)

            # Select box for output format, defaults to the chosen prompt's output format
            output_format = st.selectbox(output_format_text, options=['Score', 'Boolean'], index = ['Score', 'Boolean'].index(current_prompt['output_format']))
            
            if output_format == current_prompt['output_format']:
                
                # Display input fields for rating criteria, initialised with the current prompt's rating criteria
                rating_criteria = show_rating_criteria_input(output_format, current_prompt=current_prompt)

                # Update the rating criteria in the session state
                st.session_state['draft_prompt']['rating_criteria'] = rating_criteria

                # Display the rating criteria in the sidebar
                if output_format == 'Score':
                    display_sidebar_score_criteria()

                elif output_format == 'Boolean':
                    display_sidebar_boolean_criteria() 

                # Display and input for general criteria note, initialised with the current prompt's general criteria note
                st.markdown(general_criteria_note_header)
                general_criteria_note = st.text_area(general_criteria_note_text, value=current_prompt['general_criteria_note'], height=100)

                # Update the general criteria note in the session state
                st.session_state['draft_prompt']['general_criteria_note'] = general_criteria_note

                # Display truncated general criteria note in the sidebar
                truncated_general_criteria_note = get_first_ten_words(st.session_state['draft_prompt']['general_criteria_note'])
                st.sidebar.markdown(f"{truncated_general_criteria_note}")
                
                # Display and input for rating instruction, initialised with the current prompt's rating instruction
                st.markdown(rating_instruction_header)
                rating_instruction = st.text_area(rating_instruction_text, value=current_prompt['rating_instruction'], height=100)

                # Update the rating instruction in the session state
                st.session_state['draft_prompt']['rating_instruction'] = rating_instruction

                # Display truncated rating instruction in the sidebar
                truncated_rating_instruction = get_first_ten_words(st.session_state['draft_prompt']['rating_instruction'])
                if output_format == 'Score':
                    st.sidebar.markdown(rating_instruction_sb_header)
                    st.sidebar.markdown(f"{truncated_rating_instruction}")
                elif output_format == 'Boolean':
                    st.sidebar.markdown(evaluation_instruction_sb_header)
                    st.sidebar.markdown(f"{truncated_rating_instruction}")

                # Leave experiment description empty (redundant field)
                experiment_description = ' '

                # Update the experiment description in the session state
                st.session_state['draft_prompt']['experiment_description'] = experiment_description

                # Select or input objective title and description
                objective_title, objective_desc = objective_title_select(current_prompt=current_prompt)

                # Get and display list of teachers for selection
                teachers = get_teachers()
                teachers_options = teachers['name'].tolist()
                teacher_option = [teacher_option_text] + teachers_options
                created_by = st.selectbox(created_by_text, teacher_option, index=teacher_option.index(current_prompt['created_by']) if current_prompt['created_by'] in teacher_option else 0)

                # Update the created by in the session state
                st.session_state['draft_prompt']['created_by'] = created_by

            else:
                # Handle changes in output format by resetting the rating criteria, general criteria note and rating instruction
                # Select the appropriate example prompt based on output_format
                if output_format == 'Score':
                    example_prompt = example_prompt_score
                elif output_format == 'Boolean':
                    example_prompt = example_prompt_boolean
                
                # Update the output format in the session state
                st.session_state['draft_prompt']['output_format'] = output_format

                # Display input fields for rating criteria based on output format
                rating_criteria = show_rating_criteria_input(output_format, new=True)

                # Update the rating criteria in the session state
                st.session_state['draft_prompt']['rating_criteria'] = rating_criteria

                # Display the rating criteria in the sidebar
                if output_format == 'Score':

                    # Extract and display label and description for likert ratings
                    display_sidebar_score_criteria()

                    # Display example rating criteria in an expander
                    example_score_rating_criteria()

                elif output_format == 'Boolean':

                    # Extract and display description for TRUE and FALSE ratings
                    display_sidebar_boolean_criteria()

                    # Display example rating criteria in an expander
                    example_boolean_rating_criteria()

                # Display and input for general criteria note
                st.markdown(general_criteria_note_header)
                general_criteria_note = st.text_area(general_criteria_note_text, value="", height=100)

                # Update the general criteria note in the session state
                st.session_state['draft_prompt']['general_criteria_note'] = general_criteria_note
                
                # Display example general criteria note in an expander
                with st.expander("Example"):
                    st.write(f"{example_prompt['general_criteria_note']}")

                # Display truncated general criteria note in the sidebar
                truncated_general_criteria_note = get_first_ten_words(general_criteria_note)
                st.sidebar.markdown(f"{truncated_general_criteria_note}")

                # Display and input for rating instruction
                st.markdown(rating_instruction_header)
                rating_instruction = st.text_area(rating_instruction_text, value="", height=100)

                # Update the rating instruction in the session state
                st.session_state['draft_prompt']['rating_instruction'] = rating_instruction

                # Display example rating instruction in an expander
                with st.expander("Example"):
                    st.write(f"{example_prompt['rating_instruction']}")

                # Display truncated rating instruction in the sidebar
                truncated_rating_instruction = get_first_ten_words(rating_instruction)
                if output_format == 'Score':
                    st.sidebar.markdown(rating_instruction_sb_header)
                    st.sidebar.markdown(f"{truncated_rating_instruction}")
                elif output_format == 'Boolean':
                    st.sidebar.markdown(evaluation_instruction_sb_header)
                    st.sidebar.markdown(f"{truncated_rating_instruction}")
                
                # Leave experiment description empty (redundant field)
                experiment_description = ' '

                # Update the experiment description in the session state
                st.session_state['draft_prompt']['experiment_description'] = experiment_description

                # Select or input objective title and description
                objective_title, objective_desc = objective_title_select(current_prompt=current_prompt)

                # Get and display list of teachers for selection
                teachers = get_teachers()
                teachers_options = teachers['name'].tolist()
                teacher_option = [teacher_option_text] + teachers_options
                created_by = st.selectbox(created_by_text, teacher_option)
                st.session_state['draft_prompt']['created_by'] = created_by
            
            
            if st.button("Save Prompt", help="Save the prompt to the database."):
                # Retrieve updated prompt details from session state
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
                version = str(int(st.session_state['draft_prompt']['version']) + 1)
                
                # Save the updated prompt to the database
                returned_id = to_prompt_metadata_db(
                    prompt_objective, 
                    lesson_plan_params, 
                    output_format, 
                    rating_criteria,
                    general_criteria_note, 
                    rating_instruction, 
                    prompt_title, 
                    experiment_description, 
                    objective_title, 
                    objective_desc, 
                    prompt_created_by, 
                    version
                )

                # Display success message with the returned ID
                st.success(f"Prompt saved successfully! With ID: {returned_id}")

        else:
            # Display message if no prompts are available for the selected title
            st.write("No prompts available for the selected title.")
            


