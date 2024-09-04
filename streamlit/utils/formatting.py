""" Functions used to standardize or format data for use.

This module provides the following functions:

- standardize_key_stage: 
    Standardizes Key Stage labels.
- standardize_subject: 
    Standardizes subject labels.
- convert_to_json:
    Converts text to JSON format.
- json_to_html: 
    Converts a JSON object to an HTML-formatted string.
- fix_json_format: 
    Fixes JSON formatting issues in a given JSON string.
- process_prompt: 
    Processes prompt details, ensuring correct formatting.
- clean_response:
    Cleans JSON response by removing extraneous characters and decoding 
    the JSON content.
- decode_lesson_json:
    Decodes JSON string and logs errors if any.
- generate_experiment_placeholders: 
    Generates placeholders for an experiment based on specified parameters.
- lesson_plan_parts_at_end:
    Generates a formatted string for displaying lesson plan parts after
- get_first_ten_words:
    Extracts the first ten words from a given text and appends an ellipsis.
- display_at_end_score_criteria:
    Presents the rating criteria for scores 5 and 1.
- display_at_end_boolean_criteria:
    Displays the rating criteria for TRUE and FALSE outcomes.
    """

import json
import re
import pandas as pd
import streamlit as st
import re
import json

from utils.common_utils import log_message
from utils.constants import ErrorMessages


#TODO: do we move those to constants.py?

# Mappings for standardization
KS_MAPPINGS = {
    "year 6": "key-stage-2",
    "ks1": "key-stage-1",
    "1": "key-stage-1",
    "2": "key-stage-2",
    "3": "key-stage-3",
    "4": "key-stage-4",
    "ks3": "key-stage-3",
    "ks4": "key-stage-4",
    "ks2": "key-stage-2",
    "key stage 1": "key-stage-1",
    "key stage 2": "key-stage-2",
    "key stage 3": "key-stage-3",
    "key stage 4": "key-stage-4",
    "key stage 5": "key-stage-5",
}

SUBJECT_MAPPINGS = {
    "maths": "mathematics",
    "english": "english",
    "science": "science",
    "history": "history",
    "geography": "geography",
    "psed": "personal, social and emotional development",
    "rshe-pshe": "personal, social, health and economic education",
}

def standardize_key_stage(ks):
    """Standardizes Key Stage labels."""
    if isinstance(ks, str):
        ks = ks.strip().lower()
        return KS_MAPPINGS.get(ks, ks)
    return ks  # Return as is if not a string

def standardize_subject(subj):
    """Standardizes subject labels."""
    if isinstance(subj, str):
        subj = subj.strip().lower()
        return SUBJECT_MAPPINGS.get(subj, subj)
    return subj  # Return as is if not a string

def convert_to_json(text):
    """
    Convert text to JSON format.

    If the text is already in JSON format, it is returned as a dictionary. 
    If the text is not in JSON format or an error occurs during parsing, 
    the text is converted to a JSON object with the text stored under the 
    key 'text'.

    Args:
        text (str): The input text to be converted to JSON.

    Returns:
        dict: A dictionary representing the JSON object. If the input 
            text is valid JSON, it returns the parsed JSON. If the input 
            is not valid JSON, it returns a dictionary with the original 
            text under the key 'text'. If the input is NaN, it returns 
            None.
    """
    if pd.isna(text):
        return None
    try:
        json_data = json.loads(text)
    except json.JSONDecodeError:
        json_data = {"text": text}
    except TypeError as e:
        st.error(f"TypeError: {e} - Value: {text}")
        json_data = {"text": str(text)}
    return json_data

def json_to_html(json_obj, indent=0):
    """ Convert a JSON object to an HTML-formatted string recursively.

    Args:
        json_obj (dict or list): JSON object to convert.
        indent (int): Current level of indentation for formatting.

    Returns:
        str: HTML-formatted string representing the JSON object.
    """
    def dict_to_html(d, indent):
        """Convert a dictionary to an HTML-formatted string."""
        if not d:
            return f"{get_indent(indent)}{{}}"
        html = f"{get_indent(indent)}{{<br>"
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            html += f"{get_indent(indent + 1)}<strong>{key}</strong>: "
            html += convert_to_html(value, indent + 1)
            if i < len(items) - 1:
                html += ","
            html += "<br>" if i < len(items) - 1 else ""
        html += f"{get_indent(indent)}}}"
        return html

    def list_to_html(lst, indent):
        """Convert a list to an HTML-formatted string."""
        if not lst:
            return f"{get_indent(indent)}[]"
        html = f"{get_indent(indent)}[<br>"
        for i, item in enumerate(lst):
            html += convert_to_html(item, indent + 1)
            if i < len(lst) - 1:
                html += ","
            html += "<br>" if i < len(lst) - 1 else ""
        html += f"{get_indent(indent)}]"
        return html

    def get_indent(indent):
        """Return a string of HTML spaces for indentation."""
        return "&nbsp;&nbsp;" * indent

    def convert_to_html(obj, indent):
        """Convert a JSON object to an HTML-formatted string."""
        if isinstance(obj, dict):
            return dict_to_html(obj, indent)
        elif isinstance(obj, list):
            return list_to_html(obj, indent)
        else:
            return f"{get_indent(indent)}{obj}"

    return convert_to_html(json_obj, indent)

def fix_json_format(json_string):
    """ Fix JSON formatting issues in a given JSON string.

    Args:
        json_string (str): JSON string to fix.

    Returns:
        str: Fixed JSON string or an empty JSON object if fixing fails.
    """
    try:
        json.loads(json_string)
        return json_string
    except ValueError:
        pass

    json_string = json_string.replace('\\\\\\"', '"')
    json_string = json_string.replace("'", '"')
    json_string = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', json_string)
    try:
        json.loads(json_string)
        return json_string
    except ValueError:
        return "{}"

def process_prompt(prompt_details):
    """ Process prompt details, ensuring correct formatting.

    Args:
        prompt_details (dict): Dictionary containing prompt details.

    Returns:
        dict: Processed prompt details.
    """
    if isinstance(prompt_details.get("rating_criteria"), str):
        try:
            cleaned_criteria = (
                prompt_details["rating_criteria"].replace('\\"', '"')
            )
            prompt_details["rating_criteria"] = json.loads(cleaned_criteria)
        except json.JSONDecodeError as e:
            log_message("error", f"Error decoding JSON: {e}")
            prompt_details["rating_criteria"] = {}

    if isinstance(prompt_details.get("lesson_plan_params"), str):
        try:
            prompt_details["lesson_plan_params"] = json.loads(
                prompt_details["lesson_plan_params"]
            )
        except json.JSONDecodeError:
            prompt_details["lesson_plan_params"] = []

    prompt_details.setdefault("prompt_objective", "")
    prompt_details.setdefault("output_format", "Boolean")
    prompt_details.setdefault("general_criteria_note", "")
    prompt_details.setdefault("rating_instruction", "")
    prompt_details.setdefault("prompt_title", "")
    prompt_details.setdefault("experiment_description", "")
    prompt_details.setdefault("objective_title", "")
    prompt_details.setdefault("objective_desc", "")
    return prompt_details

def clean_response(response_text):
    """ Clean and process a JSON response text by removing extraneous 
    characters and decoding the JSON content.

    The function strips the input text, removes enclosing triple 
    backticks if they exist, and then removes any newline, 
    carriage return, tab, and backslash characters. It attempts to 
    decode the cleaned text into a JSON object. If successful, it 
    returns the decoded JSON object and a success status. If a JSON 
    decoding error occurs, it identifies the error position, extracts a 
    snippet around the problematic area, and returns an error message 
    and failure status.

    Args:
        response_text (str): The raw response text to be cleaned and 
            decoded.

    Returns:
        Tuple[Union[Dict, Any], str]: A tuple containing the cleaned and 
            decoded JSON object or an error message in case of failure, 
            and a status string ("SUCCESS" or "FAILURE").
    """
    try:
        raw_content = response_text.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:].strip()
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3].strip()
        cleaned_content = re.sub(r"[\n\r\t\\]", "", raw_content)
        return json.loads(cleaned_content), "SUCCESS"
    except json.JSONDecodeError as e:
        error_position = e.pos
        start_snippet = max(0, error_position - 40)
        end_snippet = min(len(response_text), error_position + 40)
        snippet = response_text[start_snippet:end_snippet]
        return {
            "result": None,
            "justification": (
                f"{ErrorMessages.UNEXPECTED_ERROR}: {e}. "
                f"Problematic snippet: {repr(snippet)}"
            )
        }, "FAILURE"

def decode_lesson_json(lesson_json_str, lesson_plan_id, lesson_id, index):
    """Decode JSON string and log errors if any.

    Args:
        lesson_json_str (str): JSON string of the lesson.
        lesson_plan_id (str): ID of the lesson plan.
        lesson_id (str): ID of the lesson.
        index (int): Index of the lesson in the list.

    Returns:
        dict: Decoded JSON content or None if decoding fails.
    """
    if not lesson_json_str:
        log_message("error", f"Lesson JSON is None for lesson index {index}")
        return None
    
    try:
        return json.loads(lesson_json_str)
    except json.JSONDecodeError as e:
        error_position = e.pos
        start_snippet = max(0, error_position - 40)
        end_snippet = min(len(lesson_json_str), error_position + 40)
        snippet = lesson_json_str[start_snippet:end_snippet]
        log_message("error", f"Error decoding JSON for lesson index {index}:")
        log_message("error", f"Lesson Plan ID: {lesson_plan_id}")
        log_message("error", f"Lesson ID: {lesson_id}")
        log_message("error", f"Error Message: {e}")
        log_message("error", f"Problematic snippet: {repr(snippet)}")
        return None

def generate_experiment_placeholders(model_name, temperature, limit,
        prompt_count, sample_count, teacher_name):
    """ Generate placeholders for an experiment based on specified parameters.

    Args:
        model_name (str): Name of the LLM model.
        temperature (float): Temperature parameter for the LLM.
        limit (int): Limit of lesson plans per sample.
        prompt_count (int): Number of prompts used in the experiment.
        sample_count (int): Number of samples in the experiment.
        teacher_name (str): Name of the teacher who initiated the experiment.

    Returns:
        tuple: placeholder name and description formatted as strings.
    """
    placeholder_name = (
        f"{model_name}-temp:{temperature}-prompts:{prompt_count}-samples:"
        f"{sample_count}-limit:{limit}-created:{teacher_name}"
    )
    placeholder_description = (
        f"{model_name} Evaluating with temperature {temperature}, using "
        f"{prompt_count} prompts on {sample_count} samples, with a limit of "
        f"{limit} lesson plans per sample. Run by {teacher_name}."
    )
    return placeholder_name, placeholder_description

def lesson_plan_parts_at_end(selected_lesson_plan_params, all_lesson_params, all_lesson_params_titles):
    """ Generates a formatted string for displaying lesson plan parts 
        after users click 'View Your Prompt'. The function maps lesson 
        plan parameters to their titles and formats them for display.

    Args:
        selected_lesson_plan_params (list or str): A list of lesson plan 
            parameters or a JSON string representing the list.

    Returns:
        str: A formatted string with lesson plan parts for display.
    """
    lesson_params_to_titles = dict(zip(all_lesson_params, all_lesson_params_titles))

    if isinstance(selected_lesson_plan_params, str):
        selected_lesson_plan_params = json.loads(selected_lesson_plan_params)

    return "\n".join(
        f"""
            ### {lesson_params_to_titles.get(param, param)}:\n
            *insert {param} here*\n
            ### *(End of {lesson_params_to_titles.get(param, param)})*\n
        """
        for param in selected_lesson_plan_params
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

def display_at_end_score_criteria(rating_criteria, truncated=True):
    """ This function presents the rating criteria for scores 5 and 1.
    Extracts labels and descriptions from the rating_criteria 
    dictionary and formats them for display.
    
    Args:
        rating_criteria (dict): A dictionary containing the rating
            criteria 
        truncated (bool, optional): If True, only the first ten words of
            the descriptions are displayed. Defaults to True.
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
    """ Displays the rating criteria for TRUE and FALSE outcomes.
    Extracts labels and descriptions from the rating_criteria 
    dictionary and formats them for display.

    Args:
        rating_criteria (dict): A dictionary containing the rating 
            criteria
        truncated (bool, optional): If True, only the first ten words of
            the descriptions are displayed. Defaults to True.
    """
    st.markdown("### Evaluation Criteria:")

    desc_true_short = get_first_ten_words(rating_criteria["TRUE"])
    desc_false_short = get_first_ten_words(rating_criteria["FALSE"])

    if truncated:
        st.markdown(f"TRUE: {desc_true_short}")
        st.markdown(f"FALSE: {desc_false_short}")
    else:
        st.markdown(f"TRUE: {rating_criteria['TRUE']}")
        st.markdown(f"FALSE: {rating_criteria['FALSE']}")







