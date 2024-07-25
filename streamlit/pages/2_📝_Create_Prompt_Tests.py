"""
This script is for creating and managing prompt tests using Streamlit.
It allows users to either create new prompts from scratch with guidance or modify existing prompts.

"""
import json

import pandas as pd
import streamlit as st

from utils import (
    clear_all_caches, execute_single_query, get_teachers, to_prompt_metadata_db
)

# Lesson parameters and their corresponding titles (for 'View Your Prompt'
# display purposes) and plain English descriptions
lesson_params = [
    "lesson",
    "title",
    "topic",
    "subject",
    "cycles",
    "cycle_titles",
    "cycle_feedback",
    "cycle_practice",
    "cycle_explanations",
    "cycle_spokenexplanations",
    "cycle_accompanyingslidedetails",
    "cycle_imageprompts",
    "cycle_slidetext",
    "cycle_durationinmins",
    "cycle_checkforunderstandings",
    "cycle_scripts",
    "exitQuiz",
    "keyStage",
    "starterQuiz",
    "learningCycles",
    "misconceptions",
    "priorKnowledge",
    "learningOutcome",
    "keyLearningPoints",
    "additionalMaterials",
]
lesson_params_titles = [
    "Lesson",
    "Title",
    "Topic",
    "Subject",
    "Cycles",
    "Titles",
    "Feedback",
    "Practice Tasks",
    "Explanations",
    "Spoken Explanations",
    "Accompanying Slide Details",
    "Image Prompts",
    "Slide Text",
    "Duration in Minutes",
    "Check for Understandings",
    "Scripts",
    "Exit Quiz",
    "Key Stage",
    "Starter Quiz",
    "Learning Cycles",
    "Misconceptions",
    "Prior Knowledge",
    "Learning Outcome",
    "Key Learning Points",
    "Additional Materials",
]
lesson_params_plain_eng = [
    "Whole lesson",
    "Title",
    "Topic",
    "Subject",
    "All content from all cycles",
    "All cycle titles",
    "All cycle feedback",
    "All cycle practice",
    "Entire explanations from all cycles",
    "All spoken explanations from all cycles",
    "All accompanying slide details from all cycles",
    "All image prompts from all cycles",
    "All slide text from all cycles",
    "All durations in minutes from all cycles",
    "All check for understandings from all cycles",
    "All scripts from all cycles",
    "Exit Quiz",
    "Key Stage",
    "Starter Quiz",
    "Learning cycles",
    "Misconceptions",
    "Prior knowledge",
    "Learning outcomes",
    "Key learning points",
    "Additional materials",
]


def initialize_session_state():
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = ""
    if "refresh" not in st.session_state:
        st.session_state.refresh = False
    if "draft_prompt" not in st.session_state:
        st.session_state.draft_prompt = {}


def get_all_prompts():
    """ Retrieves all prompts from the 'm_prompts' table in the database.
    The function returns the data as a pandas DataFrame and parses the 
    'rating_criteria' column from JSON strings to Python dictionaries.

    Returns:
        pd.DataFrame: A DataFrame containing all the prompts from the 
            'm_prompts' table.
    """
    query = """
    SELECT id, prompt_objective, lesson_plan_params, output_format, 
        rating_criteria, general_criteria_note, rating_instruction, 
        encode(prompt_hash, 'hex'), prompt_title, experiment_description, 
        objective_title, objective_desc, created_at, created_by, version
    FROM public.m_prompts;
    """
    data = execute_single_query(query, return_dataframe=True)
    if not data.empty:
        data["rating_criteria"] = data["rating_criteria"].apply(
            lambda x: json.loads(x) if x else {}
        )
    return data


def check_prompt_title_exists(prompt_title):
    """
    Checks if a prompt title exists in the 'm_prompts' table.

    Args:
        prompt_title (str): Prompt title to check for in the database.

    Returns:
        bool: True if the prompt title exists, False otherwise.
    """
    query = """
    SELECT COUNT(*)
    FROM public.m_prompts
    WHERE prompt_title = %s;
    """
    result = execute_single_query(query, params=(prompt_title,))
    return result[0][0] > 0 if result else False


def show_rating_criteria_input(output_format, new=False, current_prompt=None):
    """ Displays input fields for rating criteria based on the given 
    output format. The function initialises the criteria either as new 
    or based on an existing prompt and allows the user to update the 
    criteria through input fields.

    Args:
        output_format (str): The format of the output, either 'Score'
            or 'Boolean'.
        new (bool): Indicates whether the criteria are new or based on 
            an existing prompt.
        current_prompt (dict): The existing prompt data, used when new 
            is False.

    Returns:
        dict: The updated rating criteria.
    """
    st.markdown(
        "#### Rating Criteria"
        if output_format == "Score"
        else "#### Evaluation Criteria"
    )
    st.markdown(
        "Please make 5 the ideal score and 1 the worst score."
        if output_format == "Score" 
        else "Please make TRUE the ideal output"
    )
    rating_criteria_placeholder = st.empty()

    if new:
        if output_format == "Score":
            label_5, desc_5, label_1, desc_1 = "", "", "", ""
            rating_criteria = {"5 ()": "", "1 ()": ""}
        else:
            desc_t, desc_f = "", ""
            rating_criteria = {"TRUE": "", "FALSE": ""}
    else:
        current_rating_criteria = current_prompt["rating_criteria"]
        if output_format == "Score":
            label_5 = list(
                current_rating_criteria.keys()
            )[0].split("(")[-1].strip(")")
            desc_5 = list(current_rating_criteria.values())[0]
            label_1 = list(
                current_rating_criteria.keys()
            )[1].split("(")[-1].strip(")")
            desc_1 = list(current_rating_criteria.values())[1]
        else:
            desc_t = current_rating_criteria.get("TRUE", "")
            desc_f = current_rating_criteria.get("FALSE", "")
        rating_criteria = current_rating_criteria
    
    rating_criteria_placeholder.json(rating_criteria)

    # Input fields for labels and descriptions
    if output_format == "Score":
        label_5 = st.text_input("Label for 5", value=label_5, key="label_5")
        desc_5 = st.text_area(
            description_5, value=desc_5, key="desc_5", height=50
        )
        label_1 = st.text_input("Label for 1", value=label_1, key="label_1")
        desc_1 = st.text_area(
            description_1, value=desc_1, key="desc_1", height=50
        )
        rating_criteria = {f"5 ({label_5})": desc_5, f"1 ({label_1})": desc_1}
    else:
        desc_t = st.text_area(
            description_true, value=desc_t, key="desc_t", height=50
        )
        desc_f = st.text_area(
            description_false, value=desc_f, key="desc_f", height=50
        )
        rating_criteria = {"TRUE": desc_t, "FALSE": desc_f}

    rating_criteria_placeholder.json(rating_criteria)
    return rating_criteria


def objective_title_select(new=False, current_prompt=None):
    """
    Displays input fields for selecting or entering the objective title 
    and description based on whether the prompt is new or existing.

    Args:
        new (bool): Indicates whether the prompt is new or existing.
        current_prompt (dict): The existing prompt data, used when new
        is False.

    Returns:
        tuple: A tuple containing the objective title and description.
    """
    OBJECTIVES = {
        "Sanity Checks - Check if the lesson is up to oak standards": 
            ("Sanity Checks",
            "Check if the lesson is up to oak standards."),
        "Low-quality Content - Check for low-quality content in the lesson plans": 
            ("Low-quality Content",
            "Check for low-quality content in the lesson plans."),
        "Moderation Eval - Check for moderation flags in the lesson plans": 
            ("Moderation Eval",
            "Check for moderation flags in the lesson plans"),
        "New Group": (None, None)
    }

    st.markdown("#### Prompt Group")

    if new:
        objective = st.selectbox(
            "Select the group that the prompt belongs to",
            list(OBJECTIVES.keys())
        )

        if objective == "New Group":
            objective_title = st.text_input(
                "Enter the new group name", value=""
            )
            objective_desc = st.text_area(
                "Enter the description for the new group e.g. Check if the "
                "lesson is up to oak standards",
                value="",
                height=100,
            )
        else:
            objective_title, objective_desc = OBJECTIVES[objective]

        return objective_title, objective_desc
    else:
        objective_title = current_prompt["objective_title"]
        objective_desc = current_prompt["objective_desc"]
        st.markdown(f"{objective_title} - {objective_desc}")
        return objective_title, objective_desc


def display_at_end_score_criteria(truncated=True):
    """ This function presents the rating criteria for scores 5 and 1.
    It extracts labels and descriptions from a global 'rating_criteria' 
    dictionary and formats them for display.
    
    Args:
        truncated (bool, optional): If True, only the first ten words of the
            descriptions are displayed. Defaults to True.
    """
    st.markdown(rating_criteria_sb_header)

    label_5 = list(rating_criteria.keys())[0].split("(")[-1].strip(")")
    desc_5 = list(rating_criteria.values())[0]
    desc_5_short = get_first_ten_words(desc_5)

    label_1 = list(rating_criteria.keys())[1].split("(")[-1].strip(")")
    desc_1 = list(rating_criteria.values())[1]
    desc_1_short = get_first_ten_words(desc_1)

    if truncated:
        st.markdown(f"**5 ({label_5}):** {desc_5_short}")
        st.markdown(f"**1 ({label_1}):** {desc_1_short}")
    else:
        st.markdown(f"**5 ({label_5}):** {desc_5}")
        st.markdown(f"**1 ({label_1}):** {desc_1}")


def display_at_end_boolean_criteria(truncated=True):
    """ Displays the rating criteria for TRUE and FALSE outcomes. It 
    extracts descriptions from a global 'rating_criteria' dictionary and
    formats them for display.

    Args:
        truncated (bool, optional): If True, only the first ten words of the
            descriptions are displayed. Defaults to True.
    """
    st.markdown(evaluation_criteria_sb_header)

    desc_true_short = get_first_ten_words(rating_criteria["TRUE"])
    desc_false_short = get_first_ten_words(rating_criteria["FALSE"])

    if truncated:
        st.markdown(f"**TRUE:** {desc_true_short}")
        st.markdown(f"**FALSE:** {desc_false_short}")
    else:
        st.markdown(f"**TRUE:** {rating_criteria['TRUE']}")
        st.markdown(f"**FALSE:** {rating_criteria['FALSE']}")


def example_score_rating_criteria():
    """ Display example rating criteria for the 'Score' output format
        in an expander.
    """
    with st.expander("Example"):
        example_rating_criteria = example_prompt_score["rating_criteria"]
        label_5 = list(example_rating_criteria.keys())[0].split("(")[-1].strip(")")
        desc_5 = list(example_rating_criteria.values())[0]
        label_1 = list(example_rating_criteria.keys())[1].split("(")[-1].strip(")")
        desc_1 = list(example_rating_criteria.values())[1]

        st.write(f"**Label for 1**: {label_1}")
        st.write(f"**Description for 1**: {desc_1}")
        st.write(f"**Label for 5**: {label_5}")
        st.write(f"**Description for 5**: {desc_5}")


def example_boolean_rating_criteria():
    """ Display example rating criteria for 'Boolean' output format in
        an expander.
    """
    with st.expander("Example"):
        example_rating_criteria = example_prompt_boolean["rating_criteria"]
        desc_t = example_rating_criteria.get("TRUE", "")
        desc_f = example_rating_criteria.get("FALSE", "")

        st.write(f"**Description for TRUE**: {desc_t}")
        st.write(f"**Description for FALSE**: {desc_f}")


def get_lesson_plan_params(plain_eng_list):
    """ Maps a list of plain English lesson plan parameter names to
        their corresponding keys used in the system.

    Args:
        plain_eng_list (list of str): A list of lesson plan parameter 
            names in plain English.

    Returns:
        list of str: A list of corresponding keys for the lesson plan 
            parameters.
    """
    lesson_params_to_titles = dict(zip(lesson_params_plain_eng, lesson_params))
    return [
        lesson_params_to_titles[item]
        for item in plain_eng_list
        if item in lesson_params_to_titles
    ]

def lesson_plan_parts_at_end(lesson_plan_params):
    """ Generates a formatted string for displaying lesson plan parts 
        after users click 'View Your Prompt'. The function maps lesson 
        plan parameters to their titles and formats them for display.

    Args:
        lesson_plan_params (list or str): A list of lesson plan 
            parameters or a JSON string representing the list.

    Returns:
        str: A formatted string with lesson plan parts for display.
    """
    lesson_params_to_titles = dict(zip(lesson_params, lesson_params_titles))

    if isinstance(lesson_plan_params, str):
        lesson_plan_params = json.loads(lesson_plan_params)

    return "\n".join(
        f"""
            ### {lesson_params_to_titles.get(param, param)}:\n*insert 
            # {param} here*\n### _(End of 
            # {lesson_params_to_titles.get(param, param)})_\n
        """
        for param in lesson_plan_params
    )


def get_first_ten_words(text):
    """ Extracts the first ten words from a given text and appends an 
        ellipsis ('...') if there are more than ten words.

    Args:
        text (str): The input text from which to extract the first ten 
            words.

    Returns:
        str: A string containing the first ten words followed by an 
            ellipsis if the original text has more than ten words, 
            otherwise returns the original text.
    """
    words = text.split()
    first_ten_words = " ".join(words[:10]) + "..." if len(words) > 10 else text
    return first_ten_words


def fetch_prompt_details_by_id(prompt_id):
    """ Fetch prompt details by prompt ID.

    Args:
        prompt_id (int): The ID of the prompt to fetch.

    Returns:
        pd.Series or None: A pandas Series containing the prompt details
            if found, None if no prompt is found.
    """
    query = """
    SELECT prompt_title, prompt_objective, lesson_plan_params, output_format,
        rating_criteria, general_criteria_note, rating_instruction,
        experiment_description, objective_title, objective_desc, created_by
    FROM public.m_prompts
    WHERE id = %s;
    """
    result = execute_single_query(query, (prompt_id,), return_dataframe=True)

    if not result.empty:
        prompt_data = result.iloc[0]

        # Parse JSON fields safely
        prompt_data["rating_criteria"] = (
            json.loads(prompt_data["rating_criteria"])
            if prompt_data["rating_criteria"]
            else {}
        )
        prompt_data["lesson_plan_params"] = (
            json.loads(prompt_data["lesson_plan_params"])
            if prompt_data["lesson_plan_params"]
            else []
        )

        return prompt_data
    else:
        return None


def create_new_prompt(example_prompts):
    st.markdown("#### Prompt Title")
    prompt_title = st.text_input("Choose a unique title for your prompt", value="")
    (
        prompt_objective, lesson_plan_params, output_format, rating_criteria,
        general_criteria_note, rating_instruction, objective_title,
        objective_desc
    ) = display_prompt_fields()

    # View all the details of the prompt, in a simplified version of the rendered jinja template
    if st.button("View Your Prompt"):
        st.markdown(f"# *{prompt_title}* #")
        st.markdown("### Objective:")
        truncated_prompt_objective = get_first_ten_words(prompt_objective)
        st.markdown(f"{truncated_prompt_objective}")
        output = lesson_plan_parts_at_end(lesson_plan_params)
        st.markdown(output)

        if output_format == "Score":
            display_at_end_score_criteria()
        elif output_format == "Boolean":
            display_at_end_boolean_criteria()
            
        truncated_general_criteria_note = get_first_ten_words(general_criteria_note)
        st.markdown(f"{truncated_general_criteria_note}")
        
        truncated_rating_instruction = get_first_ten_words(rating_instruction)
        st.markdown("### Provide Your Rating:")
        st.markdown(f"{truncated_rating_instruction}")

    teachers = get_teachers()
    teachers_options = ["Select a teacher"] + teachers["name"].tolist()
    created_by = st.selectbox("Who is creating the prompt?", teachers_options)

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
                " ",
                objective_title,
                objective_desc,
                created_by,
                "1",
            )
            st.success(f"New prompt created successfully! With ID: {returned_id}")


def modify_existing_prompt():
    data = get_all_prompts()
    prompt_title_options = [""] + data["prompt_title"].unique().tolist()
    prompt_title = st.selectbox("Select an existing prompt to modify:", prompt_title_options)

    if "selected_prompt" not in st.session_state:
        st.session_state["selected_prompt"] = prompt_title

    if st.session_state["selected_prompt"] != prompt_title:
        st.session_state["selected_prompt"] = prompt_title
        st.session_state["refresh"] = True

    if prompt_title:
        filtered_data = data[data["prompt_title"] == prompt_title]
        latest_prompt = filtered_data.loc[filtered_data["created_at"].idxmax()]

        # Display the key details of the current prompt in a table
        st.table(latest_prompt[["created_at", "prompt_title", "prompt_objective", "output_format", "created_by", "version"]])

        (prompt_objective, lesson_plan_params, output_format, rating_criteria,
            general_criteria_note, rating_instruction, objective_title,
            objective_desc
        ) = display_prompt_fields(latest_prompt)

        # In an expander, view the full details of the prompt in a simplified version of the rendered jinja template
        with st.expander("View Full Prompt"):
            st.markdown(f'# *{latest_prompt["prompt_title"]}* #')
            st.markdown("### Objective:")
            st.markdown(f"{latest_prompt['prompt_objective']}")
            output = lesson_plan_parts_at_end(latest_prompt["lesson_plan_params"])
            st.markdown(output)

            if latest_prompt["output_format"] == "Score":
                display_at_end_score_criteria(truncated=False)
            elif latest_prompt["output_format"] == "Boolean":
                display_at_end_boolean_criteria(truncated=False)
                
            st.markdown(f"{latest_prompt['general_criteria_note']}")
            st.markdown("### Provide Your Rating:")
            st.markdown(f"{latest_prompt['rating_instruction']}")

        st.session_state["draft_prompt"] = latest_prompt.copy(deep=True)
        st.session_state["refresh"] = False

        st.markdown("#### Prompt Title")
        st.markdown(f"{latest_prompt['prompt_title']}")

        if st.button("View Your Prompt"):
            st.markdown(f"# *{prompt_title}* #")
            st.markdown("### Objective:")
            truncated_prompt_objective = get_first_ten_words(prompt_objective)
            st.markdown(f"{truncated_prompt_objective}")
            output = lesson_plan_parts_at_end(lesson_plan_params)
            st.markdown(output)
            
            if output_format == "Score":
                display_at_end_score_criteria()
            elif output_format == "Boolean":
                display_at_end_boolean_criteria()
                
            truncated_general_criteria_note = get_first_ten_words(general_criteria_note)
            st.markdown(f"{truncated_general_criteria_note}")
            
            truncated_rating_instruction = get_first_ten_words(rating_instruction)
            st.markdown("### Provide Your Rating:")
            st.markdown(f"{truncated_rating_instruction}")
        
        teachers = get_teachers()
        teachers_options = ["Select a teacher"] + teachers["name"].tolist()
        created_by = st.selectbox("Who is creating the prompt?", teachers_options)
        
        if st.button("Save Prompt", help="Save the prompt to the database."):
            # Save the updated prompt to the database
            returned_id = to_prompt_metadata_db(
                prompt_objective,
                json.dumps(lesson_plan_params),
                output_format,
                rating_criteria,
                general_criteria_note,
                rating_instruction,
                prompt_title,
                " ",
                objective_title,
                objective_desc,
                created_by,
                str(int(latest_prompt["version"]) + 1),
            )
            st.success(f"Prompt saved successfully! With ID: {returned_id}")


def display_prompt_fields(prompt_data=None):
    """
    Display and handle input fields for prompt creation or modification.

    Args:
        prompt_data (dict, optional): Existing prompt data for
            modification. Defaults to None for new prompts.

    Returns:
        tuple: Contains the following fields:
            - prompt_objective (str): The objective of the prompt
            - lesson_plan_params (list): Selected lesson plan parameters
            - output_format (str): Selected output format
                ('Score' or 'Boolean')
            - rating_criteria (dict): Rating criteria based on output
                format
            - general_criteria_note (str): General criteria notes
            - rating_instruction (str): Instructions for rating
            - objective_title (str): Title of the prompt objective
            - objective_desc (str): Description of the prompt objective
    """
    st.markdown("#### Prompt Objective")
    prompt_objective = st.text_area(
        "State what you want the LLM to check for",
        value=prompt_data["prompt_objective"] if prompt_data else "",
        height=200
    )
    st.session_state["draft_prompt"]["prompt_objective"] = prompt_objective
    
    if not prompt_data:
        with st.expander("Example"):
            st.write(f"{example_prompts['score']['prompt_objective']}")

    st.markdown("#### Relevant Lesson Plan Parts")
    if prompt_data:
        st.markdown(f"{prompt_data['lesson_plan_params']}")
        lesson_plan_params = prompt_data['lesson_plan_params']
    else:
        lesson_plan_params_st = st.multiselect(
            "Choose the parts of the lesson plan that you're evaluating",
            options=lesson_params_plain_eng
        )
        lesson_plan_params = get_lesson_plan_params(lesson_plan_params_st)

    st.markdown("#### Output Format")
    output_format = st.selectbox(
        "Choose 'Score' for a Likert scale rating (1-5) or 'Boolean' for a TRUE/FALSE evaluation",
        options=["Score", "Boolean"],
        index=["Score", "Boolean"].index(prompt_data["output_format"]) if prompt_data else 0
    )

    if not prompt_data:
        example_prompt = example_prompts["score"] if output_format == "Score" else example_prompts["boolean"]
        rating_criteria = show_rating_criteria_input(
            output_format,
            new=not prompt_data,
            current_prompt=prompt_data
        )
        st.session_state["draft_prompt"]["rating_criteria"] = rating_criteria
        
        if output_format == "Score":
            example_score_rating_criteria()
        elif output_format == "Boolean":
            example_boolean_rating_criteria()

        st.markdown("#### General Criteria Note")
        general_criteria_note = st.text_area(
            "Either leave this section empty or add things you'd like the LLM to focus on",
            value=prompt_data["general_criteria_note"] if prompt_data else "",
            height=100
        )
        with st.expander("Example"):
            st.write(f"{example_prompt['general_criteria_note']}")

        st.markdown("#### Rating Instruction")
        rating_instruction = st.text_area(
            "Tell the LLM to actually do the evaluation",
            value=prompt_data["rating_instruction"] if prompt_data else "",
            height=100
        )
        if not prompt_data:
            with st.expander("Example"):
                st.write(f"{example_prompt['rating_instruction']}")

    objective_title, objective_desc = objective_title_select(
        new=not prompt_data,
        current_prompt=prompt_data
    )

    return (
        prompt_objective,
        lesson_plan_params,
        output_format,
        rating_criteria,
        general_criteria_note,
        rating_instruction,
        objective_title,
        objective_desc
    )


# Set page configuration
st.set_page_config(page_title="Create Prompt Tests", page_icon="📝")

# Initialize session state variables
initialize_session_state()

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

# Page header
st.title("📝 Create Prompt Tests")

# Fetch example prompt data
# Score: Quiz Qs Require Explicit Knowledge
# Learning Cycles Increase in Challenge
example_prompts = {
    "score": fetch_prompt_details_by_id("fa57a7ca-604c-462d-b4e0-0d43b17b691d"),
    "boolean": fetch_prompt_details_by_id("872592e3-ba7a-408d-9995-a66f056b1ed3")
}

# Retrieve all prompt data from the database
data = get_all_prompts()

# Display a dropdown menu for the user to select an action
action = st.selectbox(
    "What would you like to do?",
    [" ", "Create a new prompt", "Modify an existing prompt"],
)

if action == "Create a new prompt":
    create_new_prompt(example_prompts)
elif action == "Modify an existing prompt":
    modify_existing_prompt()




'''
The code appears to be a part of a Streamlit application for creating and modifying prompts.
The functions create_new_prompt() and modify_existing_prompt() seem to be the main functions 
for creating new prompts and modifying existing ones, respectively.

The code uses various helper functions like display_prompt_fields(), show_rating_criteria_input(), and objective_title_select().

The structure and syntax of the code look correct overall.

There are some potential issues or areas for improvement:

a. In the modify_existing_prompt() function, there's a commented-out block of code that 
retrieves updated prompt details from st.session_state. This might be necessary, but it's currently unused.

b. In the same function, the experiment_description variable is used in the to_prompt_metadata_db() 
call, but it's not defined earlier in the function.

c. The display_prompt_fields() function returns 8 values, but when it's called in 
create_new_prompt(), it's unpacked into 8 variables. However, in modify_existing_prompt(), it's only 
unpacked into 7 variables (missing experiment_description). This could lead to issues.

d. There are some inconsistencies in how the st.session_state["draft_prompt"] is used. It's updated for some fields but not for others.

e. The lesson_plan_params variable in display_prompt_fields() is set differently depending on 
whether prompt_data is provided, but it's not clear how this is used later.

f. Some variables like example_prompts, lesson_params_plain_eng, and functions like get_lesson_plan_params(), 
get_first_ten_words(), etc., are not defined in this code snippet. They're likely defined elsewhere in the application.

g. The code uses a mix of f-strings and regular string concatenation. It might be more consistent to use f-strings throughout.

While the overall structure looks correct, there are a few potential issues that could cause problems depending on how the 
rest of the application is structured. To ensure 100% correctness, you would need to test the code thoroughly in the 
context of the entire application, ensuring all dependencies are correctly imported and all variables are properly defined and used.

'''