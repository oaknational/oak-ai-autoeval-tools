"""
1. Combine execute_query and get_data
Both execute_query and get_data functions are very similar. You can combine them into a single function with an optional parameter for returning a DataFrame.

Combined Function:
"""
def execute_query(query, params=None, return_dataframe=True):
    """
    Execute a SQL query and returns the results as a Pandas DataFrame or a list of tuples.

    Args:
        query (str): SQL query to execute.
        params (tuple, optional): Parameters for the SQL query.
        return_dataframe (bool): Whether to return the results as a DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or list: DataFrame containing the query results or a list of tuples.
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame() if return_dataframe else []

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if return_dataframe:
                    return pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
                else:
                    return cur.fetchall()
    except (psycopg2.DatabaseError, psycopg2.OperationalError) as db_err:
        log_message("error", f"Error executing query: {db_err}")
        conn.rollback()
        return pd.DataFrame() if return_dataframe else []
    except Exception as e:
        log_message("error", f"Error executing query: {e}")
        conn.rollback()
        return pd.DataFrame() if return_dataframe else []



"""
Usage Example
Replace calls to get_data with execute_query:
"""
def get_prompts():
    """ Retrieve prompts data from the database.

    Returns:
        pd.DataFrame: DataFrame with prompts data.
    """
    query = """
        WITH RankedPrompts AS (
            SELECT 
                id, 
                prompt_objective, 
                lesson_plan_params, 
                output_format, 
                rating_criteria, 
                prompt_title, 
                experiment_description, 
                objective_title,
                objective_desc, 
                version,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY prompt_title 
                    ORDER BY version DESC
                ) AS row_num
            FROM 
                public.m_prompts 
            WHERE
                id != '6a5acf94-f678-4dac-8460-efd58709f09f'
        )
        SELECT
            id, 
            prompt_objective, 
            lesson_plan_params, 
            output_format, 
            rating_criteria, 
            prompt_title, 
            experiment_description, 
            objective_title,
            objective_desc, 
            version
        FROM 
            RankedPrompts
        WHERE 
            row_num = 1;
    """
    return execute_query(query)


"""
2. Centralize Error Handling in Query Execution
You can create a decorator to handle database connections and errors to reduce redundancy in connection management and error handling.

Decorator for Error Handling:
"""
import functools

def db_error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        conn = get_db_connection()
        if not conn:
            return pd.DataFrame() if kwargs.get('return_dataframe', True) else []
        try:
            with conn:
                return func(conn, *args, **kwargs)
        except (psycopg2.DatabaseError, psycopg2.OperationalError) as db_err:
            log_message("error", f"Error executing query: {db_err}")
            conn.rollback()
            return pd.DataFrame() if kwargs.get('return_dataframe', True) else []
        except Exception as e:
            log_message("error", f"Error executing query: {e}")
            conn.rollback()
            return pd.DataFrame() if kwargs.get('return_dataframe', True) else []
    return wrapper

# Refactor Functions Using the Decorator
@db_error_handler
def execute_query(conn, query, params=None, return_dataframe=True):
    with conn.cursor() as cur:
        cur.execute(query, params)
        if return_dataframe:
            return pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        else:
            return cur.fetchall()
        
        
"""        
Usage Example
You can now refactor all database interaction functions to use this decorator, reducing repetitive error handling code.

3. Refactor run_inference
Extract the logic to clean and parse the response into a separate function.

Clean Response Function:
"""
def clean_response(response_text):
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
        json_str = response_text
        start_snippet = max(0, error_position - 40)
        end_snippet = min(len(json_str), error_position + 40)
        snippet = json_str[start_snippet:end_snippet]
        return {
            "result": None,
            "justification": f"An error occurred: {e}. Problematic snippet: {repr(snippet)}"
        }, "FAILURE"


"""
Refactor run_inference to Use clean_response:
"""
def run_inference(lesson_plan, prompt_id, llm_model, 
                  llm_model_temp, timeout=15):
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
    if set(lesson_plan.keys()) != {"title", "topic", "subject", "keyStage"}:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data is missing for this check.",
            },
            "status": "ABORTED",
        }

    if not lesson_plan:
        return {
            "response": {
                "result": None,
                "justification": "Lesson data is missing for this check.",
            },
            "status": "ABORTED",
        }
        
    prompt_details = get_prompt(prompt_id)
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

    openai.api_key = get_env_variable("OPENAI_API_KEY")
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_model_temp,
            timeout=timeout,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        cleaned_content, status = clean_response(response.choices[0].message.content)
        return {
            "response": cleaned_content,
            "status": status,
        }

    except Exception as e:
        return {
            "response": {
                "result": None,
                "justification": f"An error occurred: {e}",
            },
            "status": "FAILURE",
        }
        
"""
Summary
Combined similar functions: execute_query and get_data into one.
Centralized error handling: Using a decorator to reduce repetitive error handling code.
Extracted common logic: Created clean_response to handle response parsing and error checking in run_inference.
"""
