"""
This script is for creating and managing prompt tests using Streamlit.
It allows users to either create new prompts from scratch with guidance or 
modify existing prompts.

"""
import json

import pandas as pd
import streamlit as st

from utils import (
    clear_all_caches, execute_single_query, get_teachers, insert_prompt
)
from constants import ExamplePrompts

# Lesson parameters and their corresponding titles (for 'View Your Prompt' display
# purposes) and plain English descriptions
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


def display_at_end_score_criteria(rating_criteria, truncated=True):
    """ This function presents the rating criteria for scores 5 and 1.
    It extracts labels and descriptions from a global 'rating_criteria' 
    dictionary and formats them for display.
    
    Args:
        rating_criteria (dict): A dictionary containing the rating criteria
        truncated (bool, optional): If True, only the first ten words of the
            descriptions are displayed. Defaults to True.
    """
    st.markdown("### Rating Criteria:")

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


def display_at_end_boolean_criteria(rating_criteria, truncated=True):
    """ Displays the rating criteria for TRUE and FALSE outcomes. It 
    extracts descriptions from a global 'rating_criteria' dictionary and
    formats them for display.

    Args:
        rating_criteria (dict): A dictionary containing the rating criteria
        truncated (bool, optional): If True, only the first ten words of the
            descriptions are displayed. Defaults to True.
    """
    st.markdown("**Evaluation Criteria:**")

    desc_true_short = get_first_ten_words(rating_criteria["TRUE"])
    desc_false_short = get_first_ten_words(rating_criteria["FALSE"])

    if truncated:
        st.markdown(f"TRUE: {desc_true_short}")
        st.markdown(f"FALSE: {desc_false_short}")
    else:
        st.markdown(f"TRUE: {rating_criteria['TRUE']}")
        st.markdown(f"FALSE: {rating_criteria['FALSE']}")


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
            ### {lesson_params_to_titles.get(param, param)}:\n
            *insert {param} here*\n
            ### *(End of {lesson_params_to_titles.get(param, param)})*\n
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


def create_new_prompt():
    st.markdown("#### Prompt Title")
    prompt_title = st.text_input(
        "Choose a unique title for your prompt", value=""
    )

    # Display the header for prompt objective input
    st.markdown("#### Prompt Objective")

    # Text area for the prompt objective
    prompt_objective = st.text_area("State what you want the LLM to check for", value="", height=200)

    # Display an example prompt objective
    with st.expander("Example"):
        st.write(ExamplePrompts.PROMPT_OBJECTIVE)

    # Display the header for lesson plan parameters input
    st.markdown("#### Relevant Lesson Plan Parts")

    # Multiselect for lesson plan parameters
    lesson_plan_params_st = st.multiselect(
        "Choose the parts of the lesson plan that you're evaluating",
        options=lesson_params_plain_eng,
    )

    # Get the lesson plan parameters in the required format
    lesson_plan_params = get_lesson_plan_params(lesson_plan_params_st)

    # Display the header for output format selection
    st.markdown("#### Output Format")

    # Select box for output format
    output_format = st.selectbox("Choose 'Score' for a Likert scale rating (1-5) or 'Boolean' for a TRUE/FALSE evaluation", options=[" ", "Score", "Boolean"])

    if output_format != " ":
        # Display input fields for rating criteria based on output format
        rating_criteria = show_rating_criteria_input(output_format, new=True)

        # Display the example rating criteria as an expander
        if output_format == "Score":
            with st.expander("Example"):
                st.write(ExamplePrompts.SCORE)
        elif output_format == "Boolean":
            with st.expander("Example"):
                st.write(ExamplePrompts.BOOL)

        # Display and input for general criteria note
        st.markdown("#### General Criteria Note")
        general_criteria_note = st.text_area(
            "Either leave this section empty or add things you'd like the LLM to focus on", value="", height=100
        )

        # Display example general criteria note in an expander
        with st.expander("Example"):
            st.write(ExamplePrompts.GENERAL_CRITERIA_SCORE)

        # Display and input for rating instruction
        st.markdown("#### Rating Instruction")
        rating_instruction = st.text_area("Tell the LLM to actually do the evaluation", value="", height=100)

        # Display the example rating criteria as an expander
        if output_format == "Score":
            with st.expander("Example"):
                st.write(ExamplePrompts.RATING_INSTRUCTION_SCORE)
        elif output_format == "Boolean":
            with st.expander("Example"):
                st.write(ExamplePrompts.RATING_INSTRUCTION_BOOL)

        # Leave experiment description empty (redundant field)
        experiment_description = " "

        # View all the details of the prompt, in a simplified version of the rendered jinja template
        if st.button("View Your Prompt"):
            st.markdown(f"# *{prompt_title}* #")
            st.markdown("### Objective:")
            truncated_prompt_objective = get_first_ten_words(prompt_objective)
            st.markdown(f"{truncated_prompt_objective}")
            output = lesson_plan_parts_at_end(lesson_plan_params)
            st.markdown(output)

            if output_format == "Score":
                display_at_end_score_criteria(rating_criteria)
            elif output_format == "Boolean":
                display_at_end_boolean_criteria(rating_criteria)

            truncated_general_criteria_note = get_first_ten_words(general_criteria_note)
            st.markdown(f"{truncated_general_criteria_note}")

            truncated_rating_instruction = get_first_ten_words(rating_instruction)
            st.markdown("### Rating Instruction:")
            st.markdown(f"{truncated_rating_instruction}")
            st.markdown("---")

        # Select or input objective title and description
        objective_title, objective_desc = objective_title_select(new=True)

        # Get and display list of teachers for selection
        teachers = get_teachers()
        teachers_options = teachers["name"].tolist()
        teacher_option = ["Select a teacher"] + teachers_options
        created_by = st.selectbox("Who is creating the prompt?", teacher_option)

        # Save the new prompt to the database when the button is clicked
        if st.button("Save New Prompt", help="Save the new prompt to the database."):
            if check_prompt_title_exists(prompt_title):
                st.error("This name already exists. Choose another one.")
            else:
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
                    "1",
                )
                st.success(f"New prompt created successfully! With ID: {returned_id}")


def modify_existing_prompt():
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
            current_prompt = filtered_data.loc[filtered_data["created_at"].idxmax()]

            # Display the key details of the current prompt in a table
            st.table(current_prompt[[
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
                output = lesson_plan_parts_at_end(current_prompt["lesson_plan_params"])
                st.markdown(output)

                rating_criteria = current_prompt["rating_criteria"]
                if current_prompt["output_format"] == "Score":
                    display_at_end_score_criteria(rating_criteria, truncated=False)
                elif current_prompt["output_format"] == "Boolean":
                    display_at_end_boolean_criteria(rating_criteria, truncated=False)

                st.markdown(f"{current_prompt['general_criteria_note']}")

                st.markdown("### Rating Instruction:")
                st.markdown(f"{current_prompt['rating_instruction']}")

            # Initialise or refresh the draft prompt in session state
            st.session_state["draft_prompt"] = current_prompt.copy(deep=True)
            st.session_state["refresh"] = False

            # Display the non-editable prompt title
            st.markdown("#### Prompt Title")
            st.markdown(f"{current_prompt['prompt_title']}")

            # Display the header for prompt objective
            st.markdown("#### Prompt Objective")

            # Text area for the prompt objective, initialised with the current prompt's objective
            prompt_objective = st.text_area(
                "State what you want the LLM to check for",
                value=current_prompt["prompt_objective"],
                height=100,
            )

            # Update the prompt objective in the session state
            st.session_state["draft_prompt"]["prompt_objective"] = prompt_objective

            # Display the non-editable lesson plan parameters
            st.markdown("#### Relevant Lesson Plan Parts")
            lesson_plan_params = current_prompt["lesson_plan_params"]
            st.markdown(f"{lesson_plan_params}")

            # Display the header for output format selection
            st.markdown("#### Output Format")

            # Select box for output format, defaults to the chosen prompt's output format
            output_format = st.selectbox(
                "Choose 'Score' for a Likert scale rating (1-5) or 'Boolean' for a TRUE/FALSE evaluation",
                options=["Score", "Boolean"],
                index=["Score", "Boolean"].index(current_prompt["output_format"]),
            )

            if output_format == current_prompt["output_format"]:
                # Display input fields for rating criteria, initialised with the current prompt's rating criteria
                rating_criteria = show_rating_criteria_input(
                    output_format, current_prompt=current_prompt
                )

                # Update the rating criteria in the session state
                st.session_state["draft_prompt"]["rating_criteria"] = rating_criteria

                # Display and input for general criteria note, initialised with the current prompt's general criteria note
                st.markdown("#### General Criteria Note")
                general_criteria_note = st.text_area(
                    "Either leave this section empty or add things you'd like the LLM to focus on",
                    value=current_prompt["general_criteria_note"],
                    height=100,
                )

                # Update the general criteria note in the session state
                st.session_state["draft_prompt"][
                    "general_criteria_note"
                ] = general_criteria_note

                # Display and input for rating instruction, initialised with the current prompt's rating instruction
                st.markdown("#### Rating Instruction")
                rating_instruction = st.text_area(
                    "Tell the LLM to actually do the evaluation",
                    value=current_prompt["rating_instruction"],
                    height=100,
                )

                # Update the rating instruction in the session state
                st.session_state["draft_prompt"][
                    "rating_instruction"
                ] = rating_instruction

                # Leave experiment description empty (redundant field)
                experiment_description = " "

                # Update the experiment description in the session state
                st.session_state["draft_prompt"][
                    "experiment_description"
                ] = experiment_description

                # Select or input objective title and description
                objective_title, objective_desc = objective_title_select(
                    current_prompt=current_prompt
                )

            else:
                # Update the output format in the session state
                st.session_state["draft_prompt"]["output_format"] = output_format

                # Display input fields for rating criteria based on output format
                rating_criteria = show_rating_criteria_input(output_format, new=True)

                # Update the rating criteria in the session state
                st.session_state["draft_prompt"]["rating_criteria"] = rating_criteria

                # Display example rating criteria in an expander
                if output_format == "Score":
                    with st.expander("Example"):
                        st.write(ExamplePrompts.SCORE)
                elif output_format == "Boolean":
                    with st.expander("Example"):
                        st.write(ExamplePrompts.BOOL)

                # Display and input for general criteria note
                st.markdown("#### General Criteria Note")
                general_criteria_note = st.text_area(
                    "Either leave this section empty or add things you'd like the LLM to focus on", value="", height=100
                )

                # Update the general criteria note in the session state
                st.session_state["draft_prompt"][
                    "general_criteria_note"
                ] = general_criteria_note

                # Display example general criteria note in an expander
                with st.expander("Example"):
                    st.write(ExamplePrompts.GENERAL_CRITERIA_SCORE)

                # Display and input for rating instruction
                st.markdown("#### Rating Instruction")
                rating_instruction = st.text_area(
                    "Tell the LLM to actually do the evaluation", value="", height=100
                )

                # Update the rating instruction in the session state
                st.session_state["draft_prompt"][
                    "rating_instruction"
                ] = rating_instruction

                # Display example rating instruction in an expander
                if output_format == "Score":
                    with st.expander("Example"):
                        st.write(ExamplePrompts.RATING_INSTRUCTION_SCORE)

                elif output_format == "Boolean":
                    with st.expander("Example"):
                        st.write(ExamplePrompts.RATING_INSTRUCTION_BOOL)

                # Leave experiment description empty (redundant field)
                experiment_description = " "

                # Update the experiment description in the session state
                st.session_state["draft_prompt"][
                    "experiment_description"
                ] = experiment_description

                # Select or input objective title and description
                objective_title, objective_desc = objective_title_select(
                    current_prompt=current_prompt
                )

            # View all the details of the prompt, in a simplified version of the rendered jinja template
            if st.button("View Your Prompt"):
                st.markdown(f"# *{prompt_title}* #")
                st.markdown("### Objective:")
                truncated_prompt_objective = get_first_ten_words(prompt_objective)
                st.markdown(f"{truncated_prompt_objective}")
                output = lesson_plan_parts_at_end(lesson_plan_params)
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
                st.markdown("### Rating Instruction:")
                st.markdown(f"{truncated_rating_instruction}")

            # Get and display list of teachers for selection
            teachers = get_teachers()
            teachers_options = teachers["name"].tolist()
            teacher_option = ["Select a teacher"] + teachers_options
            created_by = st.selectbox("Who is creating the prompt?", teacher_option)
            st.session_state["draft_prompt"]["created_by"] = created_by

            if st.button("Save Prompt", help="Save the prompt to the database."):
                # Retrieve updated prompt details from session state
                prompt_objective = st.session_state["draft_prompt"]["prompt_objective"]
                lesson_plan_params = st.session_state["draft_prompt"][
                    "lesson_plan_params"
                ]
                output_format = st.session_state["draft_prompt"]["output_format"]
                rating_criteria = st.session_state["draft_prompt"]["rating_criteria"]
                general_criteria_note = st.session_state["draft_prompt"][
                    "general_criteria_note"
                ]
                rating_instruction = st.session_state["draft_prompt"][
                    "rating_instruction"
                ]
                prompt_title = st.session_state["draft_prompt"]["prompt_title"]
                experiment_description = st.session_state["draft_prompt"][
                    "experiment_description"
                ]
                objective_title = st.session_state["draft_prompt"]["objective_title"]
                objective_desc = st.session_state["draft_prompt"]["objective_desc"]
                prompt_created_by = st.session_state["draft_prompt"]["created_by"]
                version = str(int(st.session_state["draft_prompt"]["version"]) + 1)

                # Save the updated prompt to the database
                returned_id = insert_prompt(
                    prompt_objective,
                    lesson_plan_params,
                    output_format,
                    rating_criteria,
                    general_criteria_note,
                    rating_instruction,
                    prompt_title,
                    " ",
                    objective_title,
                    objective_desc,
                    prompt_created_by,
                    version,
                )
                st.success(f"Prompt saved successfully! With ID: {returned_id}")

        else:
            # Display message if no prompts are available for the selected title
            st.write("No prompts available for the selected title.")


# Set page configuration
st.set_page_config(page_title="Create Prompt Tests", page_icon="📝")

# Add a button to the sidebar to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")


st.title("📝 Create Prompt Tests")
action = st.selectbox(
    "What would you like to do?",
    [" ", "Create a new prompt", "Modify an existing prompt"],
)
if action == "Create a new prompt":
    create_new_prompt()
elif action == "Modify an existing prompt":
    modify_existing_prompt()
