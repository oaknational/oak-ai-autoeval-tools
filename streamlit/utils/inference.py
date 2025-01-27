
"""
- run_inference: 
    Runs inference using a lesson plan and a prompt ID.
- handle_inference:
    Runs inference and adds results to the database.
- run_test: 
    Runs a test for each lesson plan associated with a sample and adds
    results to the database.
"""
import streamlit as st
import os
from utils.common_utils import log_message, get_env_variable, render_prompt
import openai
from openai import OpenAI
import requests
import json
import time
import traceback

# from db_scripts import add_results, get_prompt, get_lesson_plans_by_id
from utils.formatting import clean_response, process_prompt, decode_lesson_json

def run_inference(lesson_plan, prompt_id, llm_model, llm_model_temp,
        top_p=1 ,timeout=15):
    
    
    """ Run inference using a lesson plan and a prompt ID.

    Args:
        lesson_plan (dict): Dictionary containing lesson plan details.
        prompt_id (str): ID of the prompt to use.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for the LLM.
        timeout (int, optional): Timeout duration for inference.

    Returns:
        dict: Inference result or error response.
    """
    from utils.db_scripts import get_prompt
    required_keys = ["title", "topic", "subject", "keyStage"]
    if set(lesson_plan.keys()) == set(required_keys):
        return {
            "response": {
                "result": None, 
                "justification": "Lesson data is missing for this check."
            },
            "status": "ABORTED",
        }
        
    prompt_details = get_prompt(prompt_id)
    if prompt_details['objective_title'] == "AILA Moderation":
        from utils.moderation_utils import (
                generate_moderation_prompt,
                moderate_lesson_plan,
                moderation_category_groups,
                moderation_schema,
            )
        # Generate the moderation prompt
        prompt = generate_moderation_prompt(moderation_category_groups, lesson_plan)

        try:
            result = moderate_lesson_plan(lesson_plan, moderation_category_groups, moderation_schema, llm_model,llm_model_temp)
            
            scores = result.scores.model_dump_json()  # Extract scores as a dictionary
            justification = result.justification
            categories = result.categories
            return {
                "response": {
                    "result": scores,
                    "justification": justification + " categories_detected:" + ", ".join(categories),
                },
                "status": "SUCCESS",
            }
        except Exception as e:
            log_message("error", f"Unexpected Error occurred with moderation Prompt: {e}")
            return {
                    "response": {
                        "result": None,
                        "justification": f"An error occurred: {e}",
                    },
                    "status": "FAILURE",
                }
    elif prompt_details['objective_title'] == "Merged Evals":
        from utils.multiple_output import (
                
                get_custom_category_groups,
                generate_custom_scores_model,
                merged_eval_lesson_plan,
                generate_merged_eval_prompt,
                
            )
        custom_category_groups=get_custom_category_groups()
        CustomScores = generate_custom_scores_model(custom_category_groups)
        # Generate the merged evals prompt
        prompt = generate_merged_eval_prompt(custom_category_groups, lesson_plan)

        try:

            raw_response, result = merged_eval_lesson_plan(lesson_plan, custom_category_groups, CustomScores, llm_model,llm_model_temp)

            print(result)
            scores = result.scores.model_dump_json()  # Extract scores as a dictionary
            justification = result.justification
            categories = result.categories
            return {
                "response": {
                    "result": scores,
                    "justification": justification + " categories_detected:" + ", ".join(categories),
                },
                "status": "SUCCESS",
            }
        except Exception as e:
            log_message("error", f"Unexpected Error occurred with merged evals Prompt: {e}")
            return {
                    "response": {
                        "result": None,
                        "justification": f"An error occurred: {e}",
                    },
                    "status": "FAILURE",
                }

    if not prompt_details:
        return {
            "response": {
                "result": None,
                "justification": "Prompt details not found for the given ID."
            },
            "status": "ABORTED",
        }

    cleaned_prompt_details = process_prompt(prompt_details)
    prompt = render_prompt(lesson_plan, cleaned_prompt_details)

    if "Prompt details are missing" in prompt or "Missing data" in prompt:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data missing for this check.",
            },
            "status": "ABORTED",
        }

    if llm_model != "llama":
        openai.api_key = get_env_variable("OPENAI_API_KEY")
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_model_temp,
                timeout=timeout,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
            )

            cleaned_content, status = clean_response(
                response.choices[0].message.content
            )
            return {
                "response": cleaned_content,
                "status": status,
            }

        except Exception as e:
            log_message("error", f"Unexpected error during inference: {e}")
            return {
                "response": {
                    "result": None,
                    "justification": f"An error occurred: {e}",
                },
                "status": "FAILURE",
            }
    else:
        endpoint = get_env_variable("ENDPOINT")
        username = get_env_variable("USERNAME")
        credential = get_env_variable("CREDENTIAL")
        try:
            # Define the headers for the request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credential}"
            }

            # Create the payload with the data you want to send to the model
            data = {
                "messages": [
                    {"role": "user", "content": prompt},   # Adjust this structure based on API requirements
                ],
                "temperature": llm_model_temp,
                # 'temperature': llm_model_temp,
            }

            # Make the POST request to the model endpoint
            response = requests.post(endpoint, headers=headers, data=json.dumps(data))
            

            # Check the response status and content
            if response.status_code == 200:
                response_data = response.json()
                message = response_data['choices'][0]['message']['content']
                cleaned_content, status = clean_response(message)
                return {
                    "response": cleaned_content,
                    "status": status,  
                }
            
            else:
                log_message("error", f"Failed with status code {response.status_code}: {response.text}")
                return {
                    "response": {
                        "result": None,
                        "justification": f"Failed with status code {response.status_code}: {response.text}",
                    },
                    "status": "FAILURE" 
                }

        except Exception as e:
            log_message("error", f"Unexpected error during inference: {e}")
            return {
                "response": {
                    "result": None,
                    "justification": f"An error occurred: {e}",
                },
                "status": "FAILURE" # Include elapsed time even in case of failure
            }
        
def handle_inference(content, prompt_id, llm_model, llm_model_temp, timeout,
        experiment_id, lesson_plan_id, top_p=1):
    """Run inference and add results to the database.

    Args:
        content (dict): Content to run inference on.
        prompt_id (str): ID of the prompt.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for LLM.
        timeout (int): Timeout duration for inference.
        experiment_id (int): ID of the experiment.
        lesson_plan_id (str): ID of the lesson plan.

    Returns:
        dict: Inference output.
    """
    from utils.db_scripts import add_results

    try:
        output = run_inference(
            content, prompt_id, llm_model, llm_model_temp,top_p, timeout=timeout
        )
        if not isinstance(output, dict) or "response" not in output:
            log_message("error", f"Invalid output structure: {output}")
            return None
        response = output.get("response")

        if "status" not in output:
            log_message("error", f"Key 'status' missing in output: {output}")
            return None

        if isinstance(response, dict) and all(
            isinstance(v, dict) for v in response.values()
        ):
            for _, cycle_data in response.items():
                result = cycle_data.get("result")
                justification = cycle_data.get(
                    "justification", "").replace("'", "")
                add_results(
                    experiment_id, prompt_id, lesson_plan_id, result, 
                    justification, output["status"]
                )
        else:
            result = response.get("result")
            justification = response.get("justification", "").replace("'", "")
            try:
                add_results(
                    experiment_id, prompt_id, lesson_plan_id, result, 
                    justification, output["status"]
                )
            except Exception as db_error:
                log_message("error", f"Database error: {db_error}")
                log_message("error", f"Data: {experiment_id}, {prompt_id}, {lesson_plan_id}, {result}, {justification}, {output['status']}")
                return None
        return output

    except KeyError as e:
        log_message("error", f"KeyError: Missing key in output: {e}")
        log_message("error", f"Output structure: {output}")
        return None

    except Exception as e:
        log_message("error", f"Unexpected error when adding results: {e}")
        log_message("error", traceback.format_exc())
        log_message(
            "error",
            f"""
            Lesson Plan ID: {lesson_plan_id}, 
            Prompt ID: {prompt_id}, 
            Output: {output}
            """
        )
        return None
    
def run_test(sample_id, prompt_id, experiment_id, limit, llm_model,
        llm_model_temp, top_p=1, timeout=15):
    """ Run a test for each lesson plan associated with a sample and add 
    results to the database.

    Args:
        sample_id (str): ID of the sample.
        prompt_id (str): ID of the prompt.
        experiment_id (int): ID of the experiment.
        limit (int): Maximum number of records to fetch.
        llm_model (str): Name of the LLM model.
        llm_model_temp (float): Temperature parameter for LLM.
        timeout (int, optional): Timeout duration for inference.

    Returns:
        None
    """
    from utils.db_scripts import get_lesson_plans_by_id

    lesson_plans = get_lesson_plans_by_id(sample_id, limit)
    total_lessons = len(lesson_plans)
    log_message("info", f"Total lessons: {total_lessons}")
    
    progress = st.progress(0)
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    for i, lesson in enumerate(lesson_plans):
        lesson_plan_id = lesson[0]
        lesson_id = lesson[1]
        lesson_json_str = lesson[2]

        content = decode_lesson_json(lesson_json_str, lesson_plan_id, lesson_id, i)
        if content is None:
            continue

        output = handle_inference(content, prompt_id, llm_model, llm_model_temp, timeout, experiment_id, lesson_plan_id,top_p)
        if output is None:
            continue

        response = output.get("response")
        with placeholder1.container():
            st.write(f'Inference Status: {output["status"]}')
        with placeholder2.container():
            st.write(response)
            log_message(
                "info",
                f"status = {output.get('status')},\n"
                f"lesson_plan_id = {lesson_plan_id},\n"
                f"experiment_id = {experiment_id},\n"
                f"prompt_id = {prompt_id}\n"
            )

        progress.progress((i + 1) / total_lessons)

    placeholder1.empty()
    placeholder2.empty()


def run_agent_openai_inference(prompt, llm_model, llm_model_temp, timeout=150):
        client = OpenAI( api_key= os.environ.get("OPENAI_API_KEY"), timeout=timeout)

        
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_model_temp,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            message = response.choices[0].message.content
            # print(message)
            end_time = time.time()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
            # Calculate the duration
            duration = end_time - start_time
            cleaned_content, status = clean_response(message)
            return {
                "response": cleaned_content,
                "response_time": duration,
            }

        except Exception as e:
            log_message("error", f"Unexpected error during inference: {e}")
            return {
                "response": {
                    "result": None,
                    "justification": f"An error occurred: {e}",
                },
                "status": "FAILURE",
                "response_time": duration,
            }