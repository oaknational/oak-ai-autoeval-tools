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
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy")


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
        print(f"An error occurred while fetching lesson plans: {e}")
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
        print(f"An error occurred while fetching result data: {e}")
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
    
def delete_created_sample(sample_id):
    try:
        conn = get_db_connection()
        query = """
        DELETE FROM public.m_samples
        WHERE id = %s
        """
        cur = conn.cursor()
        cur.execute(query, (sample_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"An error occurred while deleting the sample: {e}")
        return False
def delete_lesson_plans_from_sample_lesson_plans(sample_id):
    try:
        conn = get_db_connection()
        query = """
        DELETE FROM public.m_sample_lesson_plans
        WHERE sample_id = %s
        """
        cur = conn.cursor()
        cur.execute(query, (sample_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"An error occurred while deleting the sample lesson plans: {e}")
        return False

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

if lessons_df is None or lessons_df.empty:
    lessons_df = lessons_df.astype({
    'min_result': 'float64',
    'max_result': 'float64',

    })
    st.error("Failed to fetch lesson plans. Please check the database connection or query.")
    st.stop()  # Stop further execution if DataFrame is invalid

#seesion state selected_plan none
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = None

if 'justifications' not in st.session_state:
    st.session_state.justifications = None

# Filtering options
st.sidebar.header("Filter Options")
score_threshold = st.sidebar.slider("Max Score Threshold", 1, 5, 2)
filtered_df = lessons_df[lessons_df['max_result'] <= score_threshold]
st.sidebar.write("Filtered Lesson Plans")
st.sidebar.dataframe(filtered_df)
# Expandable sections for lesson details
st.write("## Lesson Details")
for i, row in filtered_df.iterrows():
    with st.expander(f"Lesson {i + 1}: {row['prompt_title']}"):
        col1, col2 = st.columns(2)

        # Add content to the first column
        with col1:
            # st.header("Column 1")
            st.write(f"**Generation Details:** {row['generation_details']}")
            st.write(f"**Total Experiments Run:** {row['justification_count']}")
            # st.header("Column 2")
            st.write(f"**Max Result:** {row['max_result']}")
            st.write(f"**Min Result:** {row['min_result']}")
            # Score Distribution
            st.write("## Score Distribution")
            score_columns = ['score_1_count', 'score_2_count', 'score_3_count', 'score_4_count', 'score_5_count']
            score_totals = row[score_columns]

            # Plot the distribution of scores
            fig, ax = plt.subplots()
            ax.bar(score_columns, score_totals, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'])
            ax.set_title("Score Distribution")
            ax.set_ylabel("Number of Scores")
            st.pyplot(fig)
            
            
            

        # Add content to the second column
        with col2:
            
            justifications = None
            # Fetch justifications for the selected row
            justifications = fetch_result_data(row['lesson_plan_id'], row['prompt_id'], row['min_result'])
            st.write('Models Justification for the lowest score')
            st.write("Justification:", justifications['justification'].values[0])
        
        unique_key = f"select_button_{i}_{row['lesson_plan_id']}"
        if st.button("Select Lesson Plan", key=unique_key):
            st.success(f"Lesson Plan {i + 1} selected!")
            st.session_state.selected_plan= row['lesson_plan_id']
            st.session_state.justifications = justifications['justification'].values[0]
        
        

lessons_df=filtered_df


if st.session_state.selected_plan is not None:
# Fetch the selected row using session_state.selected_plan
    selected_row = lessons_df[lessons_df['lesson_plan_id'] == st.session_state.selected_plan]
    selected_row = selected_row.iloc[0]

    lesson_plan = json.loads(selected_row['lesson_plan_json'])


    prompt_details =None
    

    prompt_id = selected_row['prompt_id']
    prompt_details = get_prompt(prompt_id)
    # st.write("Lesson Plan:", lesson_plan)
    if prompt_details is not None:
        
        
        with st.expander('Prompt Details'):
            st.write("Prompt Details:", prompt_details)
   
    llm_model = 'gpt-4o'
    llm_model_temp = 0.5


    improvement_prompt = (
    "You are an expert lesson plan improvement agent." 
    +"You will be provided with a lesson plan that has received a low score due to failing to meet the required standards. "
    +"Your task is to improve the lesson plan by making necessary changes to improve the lesson plan.\n\n"
    + "Here is the lesson plan:\n\n  "
    + selected_row['lesson_plan_json']
    +"The lesson plan has received the following review:\n\n  "
    + st.session_state.justifications
    + "\n\n  Please make necessary changes to improve the lesson plan. "
    + "Adhere to the original formatting and just return the json of the lesson plan. "
    + "Do not include your reasoning in your response. "
    + "Only edit the parts of the lesson plan that need improvement based on the review. "
    + "You should respond only with a valid JSON document. "
    + "Ensure that each key in the JSON corresponds exactly to the keys in the provided lesson plan. "
    + "Do not alter the formatting in any way. "
    + "Avoid introducing line break characters.\n\n  "
    
    
)

    st.markdown(improvement_prompt)
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
    llm_model_temp = 0.5
    llm_model = 'gpt-4o'
    
    # experiment_id = None

    if st.button("Perform Improvement"):
        st.write('Creating a new sample')
        sample_id = new_sample(sample_title, created_by)
        if sample_id is not None:
            st.write("Sample created successfully!")

            st.write("Running Improvement")
            response_n = run_agent_openai_inference(improvement_prompt, llm_model, llm_model_temp, timeout=150)
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
                        experiment_id = start_experiment(experiment_name, exp_description, sample_ids, teacher_id,
                        prompt_ids, limit, llm_model,
                        tracked, llm_model_temp)
                    
                
                        if experiment_id is not None:
                            st.success("Experiment completed successfully!")
                            result = fetch_final_data(lesson_plan_id, prompt_id, experiment_id)

                            # Ensure both values are float for comparison
                            new_result = float(result['result'].values[0])
                            previous_result = selected_row['min_result']

                            st.write('New Result:')
                            st.write(new_result)
                            st.write('Justification:')
                            st.write(result['justification'].values[0])

                            st.write('Previous Result:')
                            st.write(previous_result)
                            st.write('Justification:')
                            st.write(justifications['justification'].values[0])
                            delete_created_sample(sample_id)
                            delete_lesson_plans_from_sample_lesson_plans(sample_id)
                            if new_result > previous_result:
                                st.success("Lesson plan improved successfully!")
                            elif previous_result > new_result:
                                st.error("Lesson plan has been made worse!")
                            else:
                                st.warning("No improvement detected!")

                            # if result['result'].values[0] > selected_row['min_result']:
                            #     st.success("Lesson plan improved successfully!")

    