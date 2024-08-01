import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
from utils import get_db_connection, get_prompt,  add_lesson_plan_to_sample, insert_single_lesson_plan, clean_response, start_experiment
import plotly.express as px
import re
import openai
import os
from utils import  log_message, new_sample
from openai import OpenAI
import json


load_dotenv()

        
def fetch_bad_lesson_plans():
    try:
        conn = get_db_connection()
        query = """SELECT 
            r.prompt_id, 
            r.lesson_plan_id, 
            lp.generation_details,
            p.prompt_title,
            min(CAST(r.result AS numeric)) AS min_result, 
            max(CAST(r.result AS numeric)) AS max_result,
            count(r.justification) AS justification_count,
            COUNT(CASE WHEN CAST(r.result AS numeric) = 1 THEN 1 END) AS score_1_count,
            COUNT(CASE WHEN CAST(r.result AS numeric) = 2 THEN 1 END) AS score_2_count,
            COUNT(CASE WHEN CAST(r.result AS numeric) = 3 THEN 1 END) AS score_3_count,
            COUNT(CASE WHEN CAST(r.result AS numeric) = 4 THEN 1 END) AS score_4_count, 
            COUNT(CASE WHEN CAST(r.result AS numeric) = 5 THEN 1 END) AS score_5_count,
            lp.json AS lesson_plan_json
        FROM public.m_results r
        INNER JOIN m_prompts p ON p.id = r.prompt_id
        INNER JOIN lesson_plans lp ON lp.id = r.lesson_plan_id
        WHERE r.status = 'SUCCESS' AND r.result ~ '^[0-9\.]+$' AND p.output_format = 'Score' 
        AND p.prompt_title <> 'Answers Are Minimally Different'
        AND lp.generation_details LIKE '%gpt-4o%' 
        GROUP BY r.lesson_plan_id, r.prompt_id, p.prompt_title, lp.generation_details, lp.json
        HAVING min(CAST(r.result AS numeric)) < 4.0
        AND COUNT(CASE WHEN CAST(r.result AS numeric) = 5 THEN 1 END) = 0 
        AND COUNT(r.justification) > 2
        ORDER BY score_1_count DESC, justification_count DESC, max_result ASC;"""
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def fetch_result_data(lesson_plan_id, prompt_id, result):
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            r.prompt_id, r.lesson_plan_id, r.result, r.justification
        FROM public.m_results r
        WHERE r.lesson_plan_id = %s 
        AND r.prompt_id = %s 
        AND CAST(r.result AS numeric) = %s
        AND r.status = 'SUCCESS'
        ORDER BY r.result ASC
        LIMIT 1
        """
        df = pd.read_sql_query(query, conn, params=(lesson_plan_id, prompt_id, result))
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def fetch_final_data(lesson_plan_id, prompt_id, experiment_id):
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            r.result, r.justification
        FROM public.m_results r
        WHERE r.lesson_plan_id = %s 
        AND r.prompt_id = %s 
        AND r.experiment_id = %s
        """
        df = pd.read_sql_query(query, conn, params=(lesson_plan_id, prompt_id, experiment_id))
        conn.close()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def run_agent_openai_inference(prompt, llm_model, llm_model_temp, timeout=150):
        client = OpenAI( api_key= os.environ.get("OPENAI_API_KEY"), timeout=timeout)

        
        try:
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
            cleaned_content, status = clean_response(message)
            return {
                "response": cleaned_content
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
# Fetch the data
lessons_df = fetch_bad_lesson_plans()



if lessons_df is not None:
    st.write("Select a row to perform an action:")

    # Display the data frame
    st.dataframe(lessons_df[['generation_details', 'prompt_title', 'min_result', 'max_result', 'justification_count', 'score_1_count', 'score_2_count', 'score_3_count', 'score_4_count', 'score_5_count']])

    # Allow user to select a row
    selected_index = st.selectbox("Select a row", lessons_df.index, index=1)
    selected_row = lessons_df.loc[selected_index]

    st.write("You selected:")
    st.write(selected_row)

    lesson_plan = json.loads(selected_row['lesson_plan_json'])

    
    prompt_details =None
    justifications = None
    prompt_id = selected_row['prompt_id']
    prompt_details = get_prompt(prompt_id)
    # st.write("Lesson Plan:", lesson_plan)
    if prompt_details is not None:
       
        with st.expander('Prompt Details'):
            st.write("Prompt Details:", prompt_details)
    # Fetch justifications for the selected row
    justifications = fetch_result_data(selected_row['lesson_plan_id'], selected_row['prompt_id'], selected_row['min_result'])

    if justifications is not None:
        with st.expander('Justifications'):
            st.write("Justifications:", justifications['justification'].values[0])

    llm_model = 'gpt-4o'
    llm_model_temp = 0.5
    

    prompt = selected_row['lesson_plan_json'] +'test'+'\n'+ 'Above lesson plan has received the following review:\n ' + justifications['justification'].values[0] + '\n Please make necessary changes to improve the lesson plan. Adhere to the original formatting and just retrn the json of the lesson plan. Do not include your reasoining in your response. You should respond with a valid JSON document where each key of the object corresponds with the keys of the lesson plan. The value of each key should be the content for that part of the lesson plan. And avoid introducing line break characters. \n'
    sample_title = 'temp_sample'
    created_by = 'improve_lesson_plan_page'
    experiment_id = None
    limit = 1
    response =None
    lesson_plan_id = None
    output= None
    sample_id = None
    experiment_name = 'temp_experiment'
    exp_description = 'temp_experiment'
    teacher_id = 'c2358325-bf3c-44e8-80d9-37b445dad389'
    tracked= False
    # experiment_id = None

    if st.button("Perform Improvement"):
        st.write('Creating a new sample')
        sample_id = new_sample(sample_title, created_by)
        if sample_id is not None:
            st.write("Sample created successfully!")

            st.write("Running Improvement")
            response_n = run_agent_openai_inference(prompt, llm_model, llm_model_temp, timeout=150)
            if response_n['response'] is not None:
                st.write("Response")
                with st.expander('Response'):
                    st.write(response_n['response'])
                    response = response_n['response']
                    response = json.dumps(response)  # Ensure the response is correctly formatted as JSON
             
                    response_cleaned = re.sub(r'\\n|\\r', '', response)
               
                st.write('Lesson Plan Improved!')
                st.write('Inserting the improved lesson plan into the database')
                
                lesson_plan_id = insert_single_lesson_plan(response_cleaned, selected_row['lesson_plan_id'], key_stage=None, subject=None,  generation_details='improved_lesson_plan')
                if lesson_plan_id is not None:
                    st.success("Lesson plan added successfully!")
                    st.write("Lesson Plan ID:", lesson_plan_id)

                    st.write("Adding lesson plan to the sample")

                    if add_lesson_plan_to_sample(sample_id, lesson_plan_id):
                        st.success("Lesson plans added successfully!")
                        st.success("Running Evaluation")
                        sample_ids = [sample_id]
                        prompt_ids = [prompt_id]
                        experiment_id = start_experiment(
                        experiment_name, exp_description, sample_ids, teacher_id,
                        prompt_ids, st.session_state.limit, st.session_state.llm_model,
                        tracked, st.session_state.llm_model_temp)
                    
                
                        if experiment_id is not None:
                            st.success("Experiment completed successfully!")
                            result = fetch_final_data(lesson_plan_id, prompt_id, experiment_id)
                            
                            st.write('New Result:')
                            st.write(result['result'].values[0])
                            st.write('Justification:')
                            st.write(result['justification'].values[0])

                            st.write('Previous Result:')
                            st.write(selected_row['min_result'])
                            st.write('Justification:')
                            st.write(justifications['justification'].values[0])

       