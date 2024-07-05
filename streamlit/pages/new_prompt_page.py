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

# st.markdown("#### Please select a prompt to modify. As you make modifications on the selected prompt, the changes will be visible on the sidebar.:arrow_left:")
st.markdown("##### Don't forget to click Refresh Prompt if you decide to use another existing prompt.")
st.markdown("##### Once you are satisfied with the each part, click Save Changes to save your prompt.")
# st.markdown("##### You can also view the rendered version of the prompt with a sample lesson plan down below.")

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

#duplicate strings
description_1 = 'Description for 1'
description_5 = 'Description for 5'
description_true = 'Description for TRUE'
description_false = 'Description for FALSE'
# Function to clear cache
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

def get_all_prompts():
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
    # json_string = re.sub(r"'", r'"', json_string)
    json_string = json_string.replace("'", '"')
    return json_string

def check_prompt_title_exists(prompt_title):
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
    if output_format == 'Score':
        st.markdown("##### Rating Criteria")
        st.text("Please make 5 the ideal score and 1 the worst score.")
        # Initialize placeholders for the rating criteria
        rating_criteria_placeholder = st.empty()
        
        # Determine the initial values based on whether it's new or existing
        if new:
            rating_criteria = {
                '5 ()': "",
                '1 ()': ""
            }
            label_1 = ""
            desc_1 = ""
            label_5 = ""
            desc_5 = ""
        else:
            # Parse the current rating criteria
            current_rating_criteria = current_prompt['rating_criteria']
            label_1 = list(current_rating_criteria.keys())[1].split('(')[-1].strip(')')
            desc_1 = list(current_rating_criteria.values())[1]
            label_5 = list(current_rating_criteria.keys())[0].split('(')[-1].strip(')')
            desc_5 = list(current_rating_criteria.values())[0]
            rating_criteria = current_rating_criteria
        
        # Display the initial rating criteria
        rating_criteria_placeholder.json(rating_criteria)
        
        # Input fields for labels and descriptions
        label_1 = st.text_input('Label for 1 e.g. Predominantly American ', value=label_1, key="label_1")
        desc_1 = st.text_area(description_1, value=desc_1, key="desc_1", height=50)
        
        label_5 = st.text_input('Label for 5 e.g. No Americanisms Detected', value=label_5, key="label_5")
        desc_5 = st.text_area(description_5, value=desc_5, key="desc_5", height=50)
        
        # Update the rating criteria based on user input
        rating_criteria = {
            f'5 ({label_5})': desc_5,
            f'1 ({label_1})': desc_1
        }

        # Update the rating criteria placeholder with the new values
        rating_criteria_placeholder.json(rating_criteria)
    elif output_format == 'Boolean':
        st.markdown("##### Evaluation Criteria")
        st.text("Please make TRUE the ideal output")
        
        # Initialize placeholders for the rating criteria
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


def parse_experiment_desc(current_experiment_desc, output_format):
    
    if output_format == 'Score':
        parts = current_experiment_desc.split(', ')
        for part in parts:
            if part.startswith('5'):
                desc_5 = part.split('= ')[1]
            elif part.startswith('1'):
                desc_1 = part.split('= ')[1]
        return desc_5, desc_1
    elif output_format == 'Boolean':
        parts = current_experiment_desc.split(', ')
        for part in parts:
            if part.startswith('TRUE'):
                desc_true = part.split('= ')[1]
            elif part.startswith('FALSE'):
                desc_false = part.split('= ')[1]
        return desc_true, desc_false

def experiment_desc_input(output_format, new=False, current_prompt=None):
    if output_format == 'Score':
        st.markdown("#### Experiment Description")
        st.text("This is just for database purposes. State briefly what a 1 means and what a 5 means.")
        experiment_desc_placeholder = st.empty()

        if new:

            experiment_desc = "5 (ideal) =, 1 = "
            experiment_desc_placeholder.text(experiment_desc)

            exp_desc_5 = st.text_input(description_5, value="", key="exp_desc_5")
            exp_desc_1 = st.text_input(description_1, value="", key="exp_desc_1")

            experiment_desc = f"5 (ideal) = {exp_desc_5}, 1 = {exp_desc_1}"

            experiment_desc_placeholder.text(experiment_desc)
        else:
            current_experiment_desc = current_prompt['experiment_description']
            desc_5, desc_1 = parse_experiment_desc(current_experiment_desc, 'Score')
            
            exp_desc_5 = st.text_input(description_5, value=desc_5, key='exp_desc_5_mod')
            exp_desc_1 = st.text_input(description_1, value=desc_1, key='exp_desc_1_mod')
            
            experiment_desc = f"5 (ideal) = {exp_desc_5}, 1 = {exp_desc_1}"
            experiment_desc_placeholder.text(experiment_desc)


    if output_format == 'Boolean':
        st.markdown("#### Experiment Description")
        st.text("This is just for database purposes. State briefly what a TRUE means and what FALSE means.")
        experiment_desc_placeholder = st.empty()
        if new:

            experiment_desc = "TRUE (ideal) = , FALSE = "
            experiment_desc_placeholder.text(experiment_desc)

            exp_desc_t = st.text_input(description_true, value="", key="exp_desc_t")
            exp_desc_f = st.text_input(description_false, value="", key="exp_desc_f")

            experiment_desc = f"TRUE (ideal) = {exp_desc_t}, FALSE = {exp_desc_f}"

            experiment_desc_placeholder.text(experiment_desc)
        else:
            current_experiment_desc = current_prompt['experiment_description']
            desc_true, desc_false = parse_experiment_desc(current_experiment_desc, output_format)
            
            exp_desc_t = st.text_input(description_true, value=desc_true, key='exp_desc_t_mod')
            exp_desc_f = st.text_input(description_false, value=desc_false, key='exp_desc_f_mod')
            
            experiment_desc = f"TRUE (ideal) = {exp_desc_t}, FALSE = {exp_desc_f}"
            experiment_desc_placeholder.text(experiment_desc)
    
    return experiment_desc

def objective_title_select(new = False, current_prompt=None):
    st.markdown("#### Prompt Group")

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
        
       

def get_lesson_plan_params(plain_eng_list):

    dict_lesson_params = dict(zip(lesson_params, lesson_params_plain_eng))
    acc_params = []
    for i in plain_eng_list:
        for key, value in dict_lesson_params.items():
            if i == value:
                plain_eng_list[plain_eng_list.index(i)] = key
                acc_params.append(key) 
    
    return acc_params

lesson_params = ['lesson', 'title', 'topic', 'subject', 'cycles', 'cycle_titles', 'cycle_feedback',
                     'cycle_practice', 'cycle_explanations', 'cycle_spokenexplanations', 'cycle_accompanyingslidedetails',
                     'cycle_imageprompts', 'cycle_slidetext', 'cycle_durationinmins', 'cycle_checkforunderstandings',
                     'cycle_scripts', 'exitQuiz', 'keyStage', 'starterQuiz', 'learningCycles', 'misconceptions',
                     'priorKnowledge', 'learningOutcome, keyLearningPoints', 'additionalMaterials']

lesson_params_plain_eng = ['Whole lesson', 'Title', 'Topic', 'Subject', 'All content from all cycles', 'All cycle titles',
                           'All cycle feedback', 'All cycle practice', 'Entire explanations from all cycles',
                           'All spoken explanations from all cycles', 'All accompanying slide details from all cycles',
                           'All image prompts from all cycles', 'All slide text from all cycles', 'All durations in minutes from all cycles',
                           'All check for understandings from all cycles', 'All scripts from all cycles', 'Exit Quiz', 'Key Stage',
                           'Starter Quiz', 'Learning cycles', 'Misconceptions', 'Prior knowledge', 'Learning outcomes',
                           'Key learning points', 'Additional materials']
def get_lesson_plan_plain_eng(proper_lesson_params):
    # Create a dictionary from lesson_params_plain_eng to lesson_params
    dict_lesson_params = dict(zip(lesson_params_plain_eng, lesson_params))
    acc_params = []

    # Loop through the lesson_params and map them to their plain English counterparts
    for i in proper_lesson_params:
        for key, value in dict_lesson_params.items():
            if i == value:
                acc_params.append(key) 
    
    return acc_params
        
data = get_all_prompts()

filtered_data = data




action = st.selectbox("What would you like to do?", ["Select an action", "Create a new prompt", "Modify an existing prompt"])

if action == "Create a new prompt":

    st.markdown("## Creating a New Prompt")

    st.markdown('#### Prompt Title')
    prompt_title = st.text_input('E.g. Americanisms', value='')
    
    st.markdown('#### Prompt Objective')
    prompt_objective = st.text_area("State what you want the LLM to check for", value="", height=200)

    st.markdown('#### Relevant Lesson Plan Parts')
    lesson_plan_params_st = st.multiselect("Choose the parts of the lesson plan that you're evaluating", options=lesson_params_plain_eng)
    lesson_plan_params = get_lesson_plan_params(lesson_plan_params_st)
    st.markdown('#### Output Format')
    output_format = st.selectbox("Choose 'Score' for a Likert scale rating between 1 and 5, where 5 is ideal. Choose 'Boolean' for a simple TRUE/FALSE evaluation, where TRUE is ideal", options=['Score', 'Boolean'])

    rating_criteria = show_rating_criteria_input(output_format, new=True)
    st.markdown("#### General Criteria Note")
    general_criteria_note = st.text_area("Either leave this section empty or add things you'd like the LLM to focus on", value="", height=100)

    st.markdown("#### Rating Instruction")
    rating_instruction = st.text_area("Tell the LLM to actually do the evaluation", value="", height=100)

    experiment_description = experiment_desc_input(output_format, new=True)
        
    objective_title, objective_desc = objective_title_select(new=True)

    teachers = get_teachers()
    teachers_options = teachers['name'].tolist()
    teacher_option = ['Select a teacher'] + teachers_options
    created_by = st.selectbox('Who is creating the prompt?', teacher_option)


    if st.button("Save New Prompt", help="Save the new prompt to the database."):
        if check_prompt_title_exists(prompt_title):
            st.error("This name already exists. Choose another one.")
        else:
            rating_criteria_fixed = fix_json_format(rating_criteria)
            if is_valid_json(rating_criteria_fixed):
                rating_criteria = json.loads(rating_criteria_fixed)
            else:
                st.error("Unable to fix JSON format in Rating Criteria")
            
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
                created_by, 
                '1'  # Starting version for new prompt
            )

            st.success(f"New prompt created successfully! With ID: {returned_id}")



if action == "Modify an existing prompt":
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
    
        
        # Display the non-editable prompt title
        st.markdown('#### Prompt Title')
        st.markdown(f"{current_prompt['prompt_title']}")

        st.markdown('#### Prompt Objective')
        prompt_objective = st.text_area("State what you want the LLM to check for", value=current_prompt['prompt_objective'], height=100)

        
        st.markdown('#### Relevant Lesson Plan Parts')
        lesson_plan_params = current_prompt['lesson_plan_params']
        st.markdown(f"{lesson_plan_params}")
        
        st.markdown('#### Output Format')
        output_format = st.selectbox("Choose 'Score' for a Likert scale rating between 1 and 5, where 5 is ideal. Choose 'Boolean' for a simple TRUE/FALSE evaluation, where TRUE is ideal", options=['Score', 'Boolean'], index = ['Score', 'Boolean'].index(current_prompt['output_format']))
        
        if output_format == current_prompt['output_format']:
            

            rating_criteria = show_rating_criteria_input(output_format, current_prompt=current_prompt)
            st.markdown("#### General Criteria Note")
            general_criteria_note = st.text_area("Either leave this section empty or add things you'd like the LLM to focus on", value=current_prompt['general_criteria_note'], height=100)
            st.markdown("#### Rating Instruction")
            rating_instruction = st.text_area("Tell the LLM to actually do the evaluation", value=current_prompt['rating_instruction'], height=100)
            experiment_description = experiment_desc_input(output_format, current_prompt=current_prompt)
            objective_title, objective_desc = objective_title_select(current_prompt=current_prompt)

            teachers = get_teachers()
            teachers_options = teachers['name'].tolist()
            teacher_option = ['Select a teacher'] + teachers_options
            created_by = st.selectbox('Who is creating the prompt?', teacher_option, index=teacher_option.index(current_prompt['created_by']) if current_prompt['created_by'] in teacher_option else 0)

            

        else:
            
            rating_criteria = show_rating_criteria_input(output_format, new=True)
            # Add markdown headers and input fields
            st.markdown("#### General Criteria Note")
            general_criteria_note = st.text_area("Either leave this section empty or add things you'd like the LLM to focus on", value=current_prompt['general_criteria_note'], height=100)
            st.markdown("#### Rating Instruction")
            rating_instruction = st.text_area("Tell the LLM to actually do the evaluation", value=current_prompt['rating_instruction'], height=100)
            
            experiment_description = experiment_desc_input(output_format, new=True)
            
            objective_title, objective_desc = objective_title_select(current_prompt=current_prompt)

            teachers = get_teachers()
            teachers_options = teachers['name'].tolist()
            teacher_option = ['Select a teacher'] + teachers_options
            created_by = st.selectbox('Who is creating the prompt?', teacher_option)
        


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

    else:
        st.write("No prompts available for the selected title.")

