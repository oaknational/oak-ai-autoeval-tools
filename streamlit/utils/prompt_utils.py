

import json
import streamlit as st
import pandas as pd
from utils.constants import ExamplePrompts, LessonPlanParameters

from utils.db_scripts import (
execute_single_query, get_teachers, insert_prompt,

)
from utils.formatting import (
lesson_plan_parts_at_end, get_first_ten_words, display_at_end_score_criteria,
    display_at_end_boolean_criteria
)

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
            label_5 = desc_5 = label_1 = desc_1 = ""
            rating_criteria = {"5 ()": "", "1 ()": ""}
        else:
            desc_t = desc_f = ""
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
            "Description for 5", value=desc_5, key="desc_5", height=50
        )
        label_1 = st.text_input("Label for 1", value=label_1, key="label_1")
        desc_1 = st.text_area(
            "Description for 1", value=desc_1, key="desc_1", height=50
        )
        rating_criteria = {f"5 ({label_5})": desc_5, f"1 ({label_1})": desc_1}
    else:
        desc_t = st.text_area(
            "Description for TRUE", value=desc_t, key="desc_t", height=50
        )
        desc_f = st.text_area(
            "Description for FALSE", value=desc_f, key="desc_f", height=50
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
    objectives = {
        "Sanity Checks - Check if the lesson is up to oak standards.": 
            ("Sanity Checks",
            "Check if the lesson is up to oak standards."),
        "Low-quality Content - Check for low-quality content in the lesson plans.": 
            ("Low-quality Content",
            "Check for low-quality content in the lesson plans."),
        "Moderation Eval - Check for moderation flags in the lesson plans.": 
            ("Moderation Eval",
            "Check for moderation flags in the lesson plans."),
        "New Group": (None, None)
    }

    st.markdown("#### Prompt Group")

    if new:
        objective = st.selectbox(
            "Select the group that the prompt belongs to",
            list(objectives.keys())
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
            objective_title, objective_desc = objectives[objective]

        return objective_title, objective_desc
    else:
        objective_title = current_prompt["objective_title"]
        objective_desc = current_prompt["objective_desc"]
        st.markdown(f"{objective_title} - {objective_desc}")
        return objective_title, objective_desc


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
    lesson_params_to_titles = dict(zip(LessonPlanParameters.LESSON_PARAMS_PLAIN_ENG, LessonPlanParameters.LESSON_PARAMS))
    return [
        lesson_params_to_titles[item]
        for item in plain_eng_list
        if item in lesson_params_to_titles
    ]


def get_prompt_title(prompt_title, create):
    """ Helper function to get the prompt title input field.
        Called by prompt_details_inputs.
    """
    st.markdown("#### Prompt Title")
    if create:
        return st.text_input(
            "Choose a unique title for your prompt",
            value=prompt_title
        )
    else:
        st.markdown(prompt_title)
        return prompt_title


def get_prompt_objective(prompt_objective, create):
    """ Helper function to get the prompt objective input field.
        Called by prompt_details_inputs.
    """
    st.markdown("#### Prompt Objective")
    prompt_objective = st.text_area(
        "State what you want the LLM to check for",
        value=prompt_objective,
        height=200
    )
    if create:
        with st.expander("Example"):
            st.write(ExamplePrompts.PROMPT_OBJECTIVE)
    return prompt_objective


def get_lesson_plan_params_input(lesson_plan_params, create):
    """ Helper function to get the lesson plan parameters input field.
        Called by prompt_details_inputs.
    """
    st.markdown("#### Relevant Lesson Plan Parts")
    
    if create:
        lesson_plan_params_st = st.multiselect(
            "Choose the parts of the lesson plan that you're evaluating",
            options=LessonPlanParameters.LESSON_PARAMS_PLAIN_ENG
        )
        return get_lesson_plan_params(lesson_plan_params_st)
    else:
        st.markdown(lesson_plan_params)
        return lesson_plan_params


def get_output_format_details(
        output_format, rating_criteria, general_criteria_note,
        rating_instruction, create):
    """ Helper function to get the output format details.
        Called by prompt_details_inputs.
    """
    st.markdown("#### Output Format")
    
    previous_output_format = output_format
    
    output_format = st.selectbox(
        "Choose 'Score' for a Likert scale rating (1-5) or 'Boolean' for "
        "a TRUE/FALSE evaluation",
        options=[" ", "Score", "Boolean"],
        index=[" ", "Score", "Boolean"].index(output_format)
            if output_format in [" ", "Score", "Boolean"] else 0
    )
    
    if previous_output_format != output_format:
        rating_criteria = None
        general_criteria_note = rating_instruction = ""
        create = True
    
    if output_format != " ":
        rating_criteria = show_rating_criteria_input(
            output_format,
            new=not rating_criteria,
            current_prompt={"rating_criteria": rating_criteria}
                if rating_criteria else None
        )
        if create:
            show_output_format_example(output_format)
        general_criteria_note = get_general_criteria_note_input(
            general_criteria_note, create
        )
        rating_instruction = get_rating_instruction_input(
            rating_instruction, create, output_format
        )
    return (
        output_format, rating_criteria, general_criteria_note,
        rating_instruction
    )


def show_output_format_example(output_format):
    """ Helper function to show an example of the output format.
        Called by get_output_format_details.
    """
    if output_format == "Score":
        with st.expander("Example"):
            st.write(ExamplePrompts.SCORE)
    elif output_format == "Boolean":
        with st.expander("Example"):
            st.write(ExamplePrompts.BOOL)


def get_general_criteria_note_input(general_criteria_note, create):
    """ Helper function to get the general criteria note input field.
        Called by get_output_format_details.
    """
    st.markdown("#### General Criteria Note")
    general_criteria_note = st.text_area(
        "Either leave this section empty or add things you'd like the "
        "LLM to focus on",
        value=general_criteria_note,
        height=100
    )
    if create:
        with st.expander("Example"):
            st.write(ExamplePrompts.GENERAL_CRITERIA_SCORE)
    return general_criteria_note


def get_rating_instruction_input(
        rating_instruction, create, output_format):
    """ Helper function to get the rating instruction input field.
        Called by get_output_format_details.
    """
    st.markdown("#### Evaluation Instruction")
    rating_instruction = st.text_area(
        "Tell the LLM to actually do the evaluation",
        value=rating_instruction,
        height=100
    )
    if create:
        if output_format == "Score":
            with st.expander("Example"):
                st.write(ExamplePrompts.RATING_INSTRUCTION_SCORE)
        elif output_format == "Boolean":
            with st.expander("Example"):
                st.write(ExamplePrompts.RATING_INSTRUCTION_BOOL)
    return rating_instruction


def prompt_details_inputs(prompt_title="", prompt_objective="",
        lesson_plan_params=[], output_format=" ", rating_criteria=None,
        general_criteria_note="", rating_instruction="", create=True):
    """
    Displays input fields for prompt creation or modification.

    Args:
        prompt_title (str, optional): The title of the prompt.
        prompt_objective (str, optional): The objective of the prompt.
        lesson_plan_params (list, optional): The lesson plan parameters.
        output_format (str, optional): The output format, either 'Score'
            or 'Boolean'.
        rating_criteria (dict, optional): The rating criteria for the
            prompt.
        general_criteria_note (str, optional): General criteria note for
            the prompt.
        rating_instruction (str, optional): Instructions for rating the
            prompt.
        create (bool, optional): Indicates whether the prompt is being
            created or modified. Defaults to True.

    Returns:
        tuple: A tuple containing the prompt details.
    """
    prompt_title = get_prompt_title(prompt_title, create)
    prompt_objective = get_prompt_objective(prompt_objective, create)
    lesson_plan_params = (
        get_lesson_plan_params_input(lesson_plan_params, create)
    )
    (
        output_format, rating_criteria, general_criteria_note,
        rating_instruction
    ) = get_output_format_details(
        output_format, rating_criteria, general_criteria_note,
        rating_instruction, create
    )
    return (
        prompt_title, prompt_objective, lesson_plan_params, output_format,
        rating_criteria, general_criteria_note, rating_instruction
    )


def view_prompt_details(prompt_title, prompt_objective, lesson_plan_params,
        output_format, rating_criteria, general_criteria_note,
        rating_instruction):
    """
    Displays the details of a prompt.

    Args:
        prompt_title (str): The title of the prompt.
        prompt_objective (str): The objective of the prompt.
        lesson_plan_params (list): The lesson plan parameters.
        output_format (str): The output format, either 'Score' or 'Boolean'.
        rating_criteria (dict): The rating criteria for the prompt.
        general_criteria_note (str): General criteria note for the prompt.
        rating_instruction (str): Instructions for rating the prompt.
    """
    st.markdown(f"# *{prompt_title}* #")
    st.markdown("### Objective:")
    truncated_prompt_objective = get_first_ten_words(prompt_objective)
    st.markdown(f"{truncated_prompt_objective}")
    output = lesson_plan_parts_at_end(lesson_plan_params, LessonPlanParameters.LESSON_PARAMS, LessonPlanParameters.LESSON_PARAMS_TITLES)
    st.markdown(output)

    if output_format == "Score":
        display_at_end_score_criteria(rating_criteria)
    elif output_format == "Boolean":
        display_at_end_boolean_criteria(rating_criteria)

    truncated_general_criteria_note = get_first_ten_words(
        general_criteria_note
    )
    st.markdown(f"{truncated_general_criteria_note}")

    truncated_rating_instruction = get_first_ten_words(rating_instruction)
    st.markdown("### Evaluation Instruction:")
    st.markdown(f"{truncated_rating_instruction}")
    st.markdown("---")


def save_prompt(prompt_objective, lesson_plan_params, output_format,
        rating_criteria, general_criteria_note, rating_instruction,
        prompt_title, objective_title, objective_desc, created_by,
        version="1", preferred=True):
    """
    Saves a prompt to the database.

    Args:
        prompt_objective (str): The objective of the prompt.
        lesson_plan_params (list): The lesson plan parameters.
        output_format (str): The output format, either 'Score' or 'Boolean'.
        rating_criteria (dict): The rating criteria for the prompt.
        general_criteria_note (str): General criteria note for the prompt.
        rating_instruction (str): Instructions for rating the prompt.
        prompt_title (str): The title of the prompt.
        objective_title (str): The objective title.
        objective_desc (str): The objective description.
        created_by (str): The name of the creator.
        version (str, optional): The version of the prompt. Defaults to "1".
    """
    returned_id = insert_prompt(
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
        version,
        preferred,
    )
    st.success(f"Prompt saved successfully! With ID: {returned_id}")

    if preferred: 
        return version
    


def create_new_prompt():
    """
    Handles the creation of a new prompt.
    Displays input fields for the new prompt and saves it to the database.
    """
    (
        prompt_title, prompt_objective, lesson_plan_params, output_format,
        rating_criteria, general_criteria_note, rating_instruction
    ) = prompt_details_inputs()

    if output_format != " ":
        if st.button("View Your Prompt"):
            view_prompt_details(
                prompt_title, prompt_objective, lesson_plan_params,
                output_format, rating_criteria, general_criteria_note,
                rating_instruction
            )

        objective_title, objective_desc = objective_title_select(new=True)
        teachers = get_teachers()
        teachers_options = teachers["name"].tolist()
        teacher_option = ["Select a teacher"] + teachers_options
        created_by = st.selectbox(
            "Who is creating the prompt?",
            teacher_option
        )

        if st.button(
            "Save New Prompt", 
            help="Save the new prompt to the database."
        ):
            if check_prompt_title_exists(prompt_title):
                st.error("This name already exists. Choose another one.")
            else:
                save_prompt(
                    prompt_objective, lesson_plan_params, output_format,
                    rating_criteria, general_criteria_note, rating_instruction,
                    prompt_title, objective_title, objective_desc, created_by
                )

def update_previous_versions(prompt_title, saved_version):
    """
    Updates the preferred status of previous versions of a prompt.

    Args:
        prompt_title (str): The title of the prompt.
        saved_version (str): The version of the prompt that was saved.
    """
    
    query = """
                    UPDATE public.m_prompts
                    SET preferred = %s
                    WHERE prompt_title = %s
                    AND version != %s;
                    """
    execute_single_query(query, params=(False, prompt_title, saved_version))

def modify_existing_prompt():
    """
    Handles the modification of an existing prompt.
    Allows the user to select an existing prompt, modify its details,
    and save the changes to the database.
    """
    data = get_all_prompts()
    prompt_title_options = [""] + data["prompt_title"].unique().tolist()
    prompt_title = st.selectbox(
        "Select an existing prompt to modify:", prompt_title_options
    )

    if "selected_prompt" not in st.session_state:
        st.session_state["selected_prompt"] = prompt_title

    if st.session_state["selected_prompt"] != prompt_title:
        st.session_state["selected_prompt"] = prompt_title
        st.session_state["refresh"] = True

    if prompt_title:
        filtered_data = data[data["prompt_title"] == prompt_title]

        if not filtered_data.empty:
            current_prompt = filtered_data.loc[
                filtered_data["created_at"].idxmax()
            ]

            # Display the key details of the current prompt in a table
            display_data = current_prompt.copy()
            display_data['created_at'] = (
                display_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            )
            st.table(display_data[[
                "created_at",
                "prompt_title",
                "prompt_objective",
                "output_format",
                "created_by",
                "version"
            ]])

            with st.expander("View Full Prompt"):
                st.markdown(f'# *{current_prompt["prompt_title"]}* #')
                st.markdown("### Objective:")
                st.markdown(f"{current_prompt['prompt_objective']}")
                output = lesson_plan_parts_at_end(
                    current_prompt["lesson_plan_params"], LessonPlanParameters.LESSON_PARAMS, LessonPlanParameters.LESSON_PARAMS_TITLES
                )
                st.markdown(output)

                rating_criteria = current_prompt["rating_criteria"]
                if current_prompt["output_format"] == "Score":
                    display_at_end_score_criteria(
                        rating_criteria,
                        truncated=False
                    )
                elif current_prompt["output_format"] == "Boolean":
                    display_at_end_boolean_criteria(
                        rating_criteria,
                        truncated=False
                    )

                st.markdown(f"{current_prompt['general_criteria_note']}")

                st.markdown("### Evaluation Instruction:")
                st.markdown(f"{current_prompt['rating_instruction']}")

            # Initialise or refresh the draft prompt in session state
            st.session_state["draft_prompt"] = current_prompt.copy(deep=True)
            st.session_state["refresh"] = False

            (
                prompt_title, prompt_objective, lesson_plan_params,
                output_format, rating_criteria, general_criteria_note,
                rating_instruction
            ) = prompt_details_inputs(
                prompt_title=current_prompt["prompt_title"],
                prompt_objective=current_prompt["prompt_objective"],
                lesson_plan_params=json.loads(
                    current_prompt["lesson_plan_params"]
                ),
                output_format=current_prompt["output_format"],
                rating_criteria=current_prompt["rating_criteria"],
                general_criteria_note=current_prompt["general_criteria_note"],
                rating_instruction=current_prompt["rating_instruction"],
                create=False
            )
            st.session_state["draft_prompt"]["prompt_objective"] = (
                prompt_objective)
            st.session_state["draft_prompt"]["rating_criteria"] = (
                rating_criteria)
            st.session_state["draft_prompt"]["general_criteria_note"] = (
                general_criteria_note)
            st.session_state["draft_prompt"]["rating_instruction"] = (
                rating_instruction)

            if st.button("View Your Prompt"):
                view_prompt_details(
                    prompt_title, prompt_objective, lesson_plan_params,
                    output_format, rating_criteria, general_criteria_note,
                    rating_instruction
                )
            objective_title, objective_desc = objective_title_select(
                current_prompt=current_prompt)

            teachers = get_teachers()
            teachers_options = teachers["name"].tolist()
            teacher_option = ["Select a teacher"] + teachers_options
            created_by = st.selectbox(
                "Who is creating the prompt?",
                teacher_option
            )
            st.session_state["draft_prompt"]["created_by"] = created_by

            #add a checkbox to make the prompt preferred
            make_preferred = st.checkbox("Make this prompt preferred", value=True)
            saved_version = None
            if st.button(
                "Save Prompt", help="Save the prompt to the database."
            ):
                version = str(int(current_prompt["version"]) + 1)
                saved_version = save_prompt(
                    prompt_objective, lesson_plan_params, output_format,
                    rating_criteria, general_criteria_note, rating_instruction,
                    prompt_title, objective_title, objective_desc, created_by,
                    version, make_preferred
                )

                if saved_version is not None:
                    update_previous_versions(prompt_title, saved_version)
                    

        else:
            st.write("No prompts available for the selected title.")