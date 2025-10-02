
"""
Inference module for running AI model evaluations and tests.

This module provides functions for:
- run_inference: Runs inference using a lesson plan and a prompt ID
- handle_inference: Runs inference and adds results to the database
- run_test: Runs a test for each lesson plan associated with a sample and adds results to the database
- run_agent_openai_inference: Runs OpenAI inference with timing and error handling
"""

import json
import os
import time
import traceback
from typing import Dict, Any, Optional, Union, List, Tuple

import requests
import streamlit as st
from openai import OpenAI

from utils.common_utils import log_message, get_env_variable, render_prompt
from utils.formatting import clean_response, process_prompt, decode_lesson_json


def _create_error_response(justification: str, status: str) -> Dict[str, Any]:
    """Create a standardized error response.
    
    Args:
        justification: Error message explaining what went wrong
        status: Status code ("ABORTED", "FAILURE", etc.)
        
    Returns:
        Standardized error response dictionary
    """
    return {
        "response": {
            "result": None,
            "justification": justification,
        },
        "status": status,
    }


def _handle_moderation_inference(
    lesson_plan: Union[Dict[str, Any], str], 
    llm_model: str, 
    llm_model_temp: float,
    selected_categories: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """Handle AILA Moderation inference.
    
    Args:
        lesson_plan: Lesson plan content to moderate
        llm_model: Model to use for moderation
        llm_model_temp: Temperature setting for the model
        selected_categories: Optional dict of category abbreviations to include
        
    Returns:
        Moderation result or error response
    """
    from utils.moderation_utils import moderate_lesson_plan
    
    try:
        current_temp = float(llm_model_temp)
        
        # Call the moderation function with selected categories
        moderation_result = moderate_lesson_plan(
            lesson_plan=str(lesson_plan),
            llm=llm_model,
            temp=current_temp,
            selected_categories=selected_categories
        )
        
        # Extract results from moderation response
        scores_json_str = moderation_result.scores.model_dump_json()
        individual_category_justifications = moderation_result.justifications
        flagged_pydantic_category_codes = moderation_result.flagged_categories
        
        # Create summary justification
        summary_justification_parts = []
        if individual_category_justifications:
            summary_justification_parts.append("Flagged category justifications (score < 5):")
            for pydantic_code, justification_text in individual_category_justifications.items():
                summary_justification_parts.append(f"- {pydantic_code}: {justification_text}")
        else:
            summary_justification_parts.append(
                "No categories scored less than 5, or no justifications were provided by the LLM for flagged categories."
            )
        
        if flagged_pydantic_category_codes:
            summary_justification_parts.append(
                f"All flagged category (Pydantic) codes (scored < 5): {', '.join(flagged_pydantic_category_codes)}"
            )
        else:
            summary_justification_parts.append("No categories were flagged (all scored 5).")
        
        final_summary_justification_for_db = "\n".join(summary_justification_parts)
        
        return {
            "response": {
                "result": scores_json_str,
                "justification": final_summary_justification_for_db,
                "detailed_justifications": individual_category_justifications,
                "flagged_categories_codes": flagged_pydantic_category_codes
            },
            "status": "SUCCESS",
        }
        
    except RuntimeError as e:
        log_message("error", f"Moderation Prompt processing error: {e}")
        return _create_error_response(f"An error occurred during moderation: {e}", "FAILURE")
        
    except Exception as e:
        log_message("error", f"Unexpected Error occurred with moderation Prompt logic: {e}")
        log_message("error", traceback.format_exc())
        return _create_error_response(f"An unexpected error occurred: {e}", "FAILURE")


def _run_openai_inference(
    prompt: str, 
    llm_model: str, 
    llm_model_temp: float, 
    top_p: float, 
    timeout: int
) -> Dict[str, Any]:
    """Run inference using OpenAI API.
    
    Args:
        prompt: The prompt to send to the model
        llm_model: Model name to use
        llm_model_temp: Temperature setting
        top_p: Top-p parameter
        timeout: Request timeout in seconds
        
    Returns:
        Inference result or error response
    """
    try:
        api_key = get_env_variable("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_model_temp,
            timeout=timeout,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
        
        cleaned_content, status = clean_response(response.choices[0].message.content)
        return {
            "response": cleaned_content,
            "status": status,
        }
        
    except Exception as e:
        log_message("error", f"Unexpected error during OpenAI inference: {e}")
        return _create_error_response(f"An error occurred: {e}", "FAILURE")


def _run_llama_inference(
    prompt: str, 
    llm_model_temp: float, 
    timeout: int
) -> Dict[str, Any]:
    """Run inference using Llama API.
    
    Args:
        prompt: The prompt to send to the model
        llm_model_temp: Temperature setting
        timeout: Request timeout in seconds
        
    Returns:
        Inference result or error response
    """
    try:
        endpoint = get_env_variable("ENDPOINT")
        credential = get_env_variable("CREDENTIAL")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credential}"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": llm_model_temp,
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(data))
        
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
            return _create_error_response(
                f"Failed with status code {response.status_code}: {response.text}", 
                "FAILURE"
            )
            
    except Exception as e:
        log_message("error", f"Unexpected error during Llama inference: {e}")
        return _create_error_response(f"An error occurred: {e}", "FAILURE")

def run_inference(
    lesson_plan: Union[Dict[str, Any], str], 
    prompt_id: str, 
    llm_model: str, 
    llm_model_temp: float,
    top_p: float = 1.0, 
    timeout: int = 15
) -> Dict[str, Any]:
    
    
    """Run inference using a lesson plan and a prompt ID.

    Args:
        lesson_plan: Dictionary containing lesson plan details or pre-formatted string.
                    Ensure str(lesson_plan) produces a suitable document for the LLM.
        prompt_id: ID of the prompt to use.
        llm_model: Name of the LLM model.
        llm_model_temp: Temperature parameter for the LLM.
        top_p: Top-p parameter for the LLM (default: 1.0).
        timeout: Timeout duration for inference in seconds (default: 15).

    Returns:
        Dictionary containing:
        - response: The inference result or error details
        - status: Status of the inference ("SUCCESS", "FAILURE", "ABORTED")
    """
    from utils.db_scripts import get_prompt
        
    try:
        prompt_details = get_prompt(prompt_id)
    except Exception as e:
        log_message("error", f"Failed to retrieve prompt details for ID {prompt_id}: {e}")
        return _create_error_response(
            f"Failed to retrieve prompt details: {e}", 
            "ABORTED"
        )
    
    if not prompt_details or 'objective_title' not in prompt_details:
        log_message("error", f"Prompt details not found or malformed for prompt_id: {prompt_id}")
        return _create_error_response(
            f"Prompt details not found or malformed for prompt ID: {prompt_id}.", 
            "ABORTED"
        )
    
    # Handle AILA Moderation special case
    if prompt_details['objective_title'] == "AILA Moderation":
        return _handle_moderation_inference(lesson_plan, llm_model, llm_model_temp)
                
    # This check is redundant since we already validated prompt_details above
    # Keeping for safety but it should never be reached

    try:
        cleaned_prompt_details = process_prompt(prompt_details)
        prompt = render_prompt(lesson_plan, cleaned_prompt_details)
    except Exception as e:
        log_message("error", f"Failed to process prompt or render template: {e}")
        return _create_error_response(
            f"Failed to process prompt: {e}", 
            "ABORTED"
        )

    if "Prompt details are missing" in prompt or "Missing data" in prompt:
        return _create_error_response(
            "Lesson data missing for this check.", 
            "ABORTED"
        )

    # Route to appropriate inference method based on model type
    if llm_model != "llama":
        return _run_openai_inference(prompt, llm_model, llm_model_temp, top_p, timeout)
    else:
        return _run_llama_inference(prompt, llm_model_temp, timeout)
        
def handle_inference(
    content: Union[Dict[str, Any], str], 
    prompt_id: str, 
    llm_model: str, 
    llm_model_temp: float, 
    timeout: int,
    experiment_id: int, 
    lesson_plan_id: str, 
    top_p: float = 1.0
) -> Optional[Dict[str, Any]]:
    """Run inference and add results to the database.

    Args:
        content: Content to run inference on.
        prompt_id: ID of the prompt.
        llm_model: Name of the LLM model.
        llm_model_temp: Temperature parameter for LLM.
        timeout: Timeout duration for inference.
        experiment_id: ID of the experiment.
        lesson_plan_id: ID of the lesson plan.
        top_p: Top-p parameter for the LLM (default: 1.0).

    Returns:
        Inference output dictionary or None if an error occurred.
    """
    from utils.db_scripts import add_results

    try:
        output = run_inference(
            content, prompt_id, llm_model, llm_model_temp, top_p, timeout=timeout
        )
        
        # Validate output structure
        if not isinstance(output, dict):
            log_message("error", f"Invalid output type: expected dict, got {type(output)}")
            return None
            
        if "response" not in output or "status" not in output:
            log_message("error", f"Missing required keys in output: {list(output.keys())}")
            return None
            
        response = output["response"]
        status = output["status"]
        
        # Handle different response structures
        if isinstance(response, dict) and all(
            isinstance(v, dict) for v in response.values()
        ):
            # Handle multi-cycle responses
            for cycle_name, cycle_data in response.items():
                _save_cycle_results(
                    experiment_id, prompt_id, lesson_plan_id, 
                    cycle_data, status, cycle_name
                )
        else:
            # Handle single response
            _save_cycle_results(
                experiment_id, prompt_id, lesson_plan_id, 
                response, status
            )
            
        return output

    except Exception as e:
        log_message("error", f"Unexpected error in handle_inference: {e}")
        log_message("error", traceback.format_exc())
        log_message(
            "error",
            f"Lesson Plan ID: {lesson_plan_id}, Prompt ID: {prompt_id}"
        )
        return None


def _save_cycle_results(
    experiment_id: int, 
    prompt_id: str, 
    lesson_plan_id: str, 
    cycle_data: Dict[str, Any], 
    status: str,
    cycle_name: Optional[str] = None
) -> None:
    """Save results for a single cycle to the database.
    
    Args:
        experiment_id: ID of the experiment
        prompt_id: ID of the prompt
        lesson_plan_id: ID of the lesson plan
        cycle_data: Data for this cycle
        status: Status of the inference
        cycle_name: Optional name of the cycle (for multi-cycle responses)
    """
    from utils.db_scripts import add_results
    
    try:
        result = cycle_data.get("result")
        justification = cycle_data.get("justification", "").replace("'", "")
        
        add_results(
            experiment_id, prompt_id, lesson_plan_id, result, 
            justification, status
        )
        
    except Exception as db_error:
        cycle_info = f" (cycle: {cycle_name})" if cycle_name else ""
        log_message("error", f"Database error{cycle_info}: {db_error}")
        log_message("error", f"Data: {experiment_id}, {prompt_id}, {lesson_plan_id}, {result}, {justification}, {status}")
        raise
    
def run_test(
    sample_id: str, 
    prompt_id: str, 
    experiment_id: int, 
    limit: int, 
    llm_model: str,
    llm_model_temp: float, 
    top_p: float = 1.0, 
    timeout: int = 15
) -> None:
    """Run a test for each lesson plan associated with a sample and add results to the database.

    Args:
        sample_id: ID of the sample.
        prompt_id: ID of the prompt.
        experiment_id: ID of the experiment.
        limit: Maximum number of records to fetch.
        llm_model: Name of the LLM model.
        llm_model_temp: Temperature parameter for LLM.
        top_p: Top-p parameter for the LLM (default: 1.0).
        timeout: Timeout duration for inference in seconds (default: 15).

    Returns:
        None
    """
    from utils.db_scripts import get_lesson_plans_by_id

    try:
        lesson_plans = get_lesson_plans_by_id(sample_id, limit)
        total_lessons = len(lesson_plans)
        log_message("info", f"Total lessons to process: {total_lessons}")
        
        if total_lessons == 0:
            log_message("warning", f"No lesson plans found for sample_id: {sample_id}")
            return
        
        # Initialize progress tracking
        progress = st.progress(0)
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        successful_inferences = 0
        failed_inferences = 0

        for i, lesson in enumerate(lesson_plans):
            try:
                lesson_plan_id = lesson[0]
                lesson_id = lesson[1]
                lesson_json_str = lesson[2]

                # Decode lesson content
                content = decode_lesson_json(lesson_json_str, lesson_plan_id, lesson_id, i)
                if content is None:
                    log_message("warning", f"Skipping lesson {i+1}: Failed to decode lesson JSON")
                    failed_inferences += 1
                    continue

                # Run inference
                output = handle_inference(
                    content, prompt_id, llm_model, llm_model_temp, 
                    timeout, experiment_id, lesson_plan_id, top_p
                )
                
                if output is None:
                    log_message("warning", f"Skipping lesson {i+1}: Inference failed")
                    failed_inferences += 1
                    continue

                # Update UI
                with status_placeholder.container():
                    st.write(f'Inference Status: {output["status"]} (Lesson {i+1}/{total_lessons})')
                    
                with response_placeholder.container():
                    st.write(output.get("response", {}))
                    
                # Log success
                log_message(
                    "info",
                    f"Processed lesson {i+1}/{total_lessons} - "
                    f"Status: {output.get('status')}, "
                    f"Lesson Plan ID: {lesson_plan_id}"
                )
                
                successful_inferences += 1

            except Exception as e:
                log_message("error", f"Error processing lesson {i+1}: {e}")
                failed_inferences += 1
                continue

            # Update progress
            progress.progress((i + 1) / total_lessons)

        # Final summary
        log_message("info", f"Test completed: {successful_inferences} successful, {failed_inferences} failed")
        
    except Exception as e:
        log_message("error", f"Error in run_test: {e}")
        log_message("error", traceback.format_exc())
    finally:
        # Clean up UI elements
        try:
            status_placeholder.empty()
            response_placeholder.empty()
        except Exception:
            # Ignore cleanup errors
            pass


def run_agent_openai_inference(
    prompt: str, 
    llm_model: str, 
    llm_model_temp: float, 
    timeout: int = 150
) -> Dict[str, Any]:
    """Run OpenAI inference with timing and comprehensive error handling.
    
    Args:
        prompt: The prompt to send to the model
        llm_model: Model name to use
        llm_model_temp: Temperature setting
        timeout: Request timeout in seconds (default: 150)
        
    Returns:
        Dictionary containing response, status, and response time
    """
    start_time = time.time()
    duration = 0.0
    
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        client = OpenAI(api_key=api_key, timeout=timeout)
        
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_model_temp,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
        )
        
        message = response.choices[0].message.content
        end_time = time.time()
        duration = end_time - start_time
        
        cleaned_content, status = clean_response(message)
        return {
            "response": cleaned_content,
            "status": status,
            "response_time": duration,
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        log_message("error", f"Unexpected error during agent inference: {e}")
        return {
            "response": {
                "result": None,
                "justification": f"An error occurred: {e}",
            },
            "status": "FAILURE",
            "response_time": duration,
        }