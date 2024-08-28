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


#TODO:  -remove bad and test lesson plans query duplicaiton
#       -Test if the improvement works with multiple prompt id selections. 
#       -Add the ability to select multiple lesson plans for improvement
#       -More verbal instructions and explanations
#       -Add the ability to select multiple lesson plans for comparison



warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy")


load_dotenv()
def fetch_test_lesson_plans():
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
            COUNT(CASE WHEN CAST(r.result AS numeric) = 5 THEN 1 END) AS score_5_count
            -- , lp.json AS lesson_plan_json
        FROM public.m_results r
        INNER JOIN m_prompts p ON p.id = r.prompt_id
        INNER JOIN lesson_plans lp ON lp.id = r.lesson_plan_id
        WHERE r.status = 'SUCCESS' AND r.result ~ '^[0-9\\.]+$' AND p.output_format = 'Score' 
        AND p.prompt_title <> 'Answers Are Minimally Different'
        AND lp.generation_details LIKE '%gpt-4o%' 
        GROUP BY r.lesson_plan_id, r.prompt_id, p.prompt_title, lp.generation_details, lp.json
        
		-- min(CAST(r.result AS numeric)) < 4.0 AND
         -- COUNT(CASE WHEN CAST(r.result AS numeric) = 5 THEN 1 END) = 0 AND 
        
        ORDER BY  lesson_plan_id DESC, justification_count DESC, max_result ASC;"""
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    except Exception as e:
        print(f"An error occurred while fetching lesson plans: {e}")
        return None
        
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
        WHERE r.status = 'SUCCESS' AND r.result ~ '^[0-9\\.]+$' AND p.output_format = 'Score' 
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

test_df =fetch_test_lesson_plans()

# test_df

def calculate_success_failure_rate(test_df):
    
    test_df = test_df.groupby(['lesson_plan_id', 'generation_details', 'prompt_id']).agg({
        'min_result': 'min',
        'max_result': 'max',
        'justification_count': 'sum',
        'score_1_count': 'sum',
        'score_2_count': 'sum',
        'score_3_count': 'sum',
        'score_4_count': 'sum',
        'score_5_count': 'sum',
        'prompt_title': lambda x: ', '.join(x)  # Concatenate prompt titles as a comma-separated string
    }).reset_index()


    test_df['stellar_success_rate'] = (test_df['score_5_count'] / test_df['justification_count']) * 100
    test_df['catastrophic_fail_rate'] = (test_df['score_1_count'] / test_df['justification_count']) * 100

    test_df = test_df[['lesson_plan_id', 'generation_details','prompt_title', 'prompt_id' , 'stellar_success_rate', 'catastrophic_fail_rate']]

    #rename 'prompt_title' to 'prompt_titles'
    # display_df.rename(columns={'prompt_title': 'prompt_titles'}, inplace=True)
    test_df['overall_fail_score'] = (100 - test_df['stellar_success_rate']) + test_df['catastrophic_fail_rate']

    return test_df



# group test_df by lesson_plan_id
lessons_df =fetch_bad_lesson_plans()

lessons_df_backup = lessons_df
lessons_df_backup

# lessons_df = fetch_bad_lesson_plans()
lessons_df =fetch_test_lesson_plans()

#order the lessons_df by lesson_plan_id and prompt_id
lessons_df = lessons_df.sort_values(by=['lesson_plan_id', 'prompt_id'])
# lessons_df
# st.write("## number of rows in the lesson plan dataframe")
# st.write(lessons_df.shape[0])
processed_df = calculate_success_failure_rate(lessons_df)
# order the processed_df by lesson_plan_id and prompt_id
processed_df = processed_df.sort_values(by=['lesson_plan_id', 'prompt_id'])
# processed_df
# st.write("## number of rows in the processed dataframe")
# st.write(processed_df.shape[0])

#merge the processed_df with the lessons_df
lessons_df_merged = pd.merge(lessons_df, processed_df, on=['lesson_plan_id', 'generation_details','prompt_title', 'prompt_id'], how='inner')

#orde the lessons_df_merged by justification count desc, overall_fail_score desc
lessons_df_merged = lessons_df_merged.sort_values(by=['justification_count', 'overall_fail_score'], ascending=[False, False])

# lessons_df_merged


# group the lessons_df by prompt_title and sum justification_count, score_1_count, score_2_count, score_3_count, score_4_count, score_5_count, average min_result, average max_result, average stellar_success_rate, average catastrophic_fail_rate, average overall_fail_score
lessons_df_p_grouped = lessons_df_merged.groupby(['prompt_title']).agg({
    'justification_count': 'sum',
    'lesson_plan_id': 'count',
    'score_1_count': 'sum',
    'score_2_count': 'sum',
    'score_3_count': 'sum',
    'score_4_count': 'sum',
    'score_5_count': 'sum',
    'min_result': 'mean',
    'max_result': 'mean',
    'stellar_success_rate': 'mean',
    'catastrophic_fail_rate': 'mean',
    'overall_fail_score': 'mean',
    'prompt_id': lambda x: ', '.join(set(x))
}).reset_index()
st.dataframe(lessons_df_p_grouped)



st.write(f"##### number of rows in the merged dataframe: {lessons_df_merged.shape[0]}")



# set a threshold for the justification count
threshold = 1
# allow user to select the threshold by inputting a value
threshold = st.number_input("Enter the threshold for the justification count", min_value=1, value=1, step=1)

prompt_titles = lessons_df_merged['prompt_title'].unique()
prompt_title_selection = None

prompt_title_selection = st.selectbox("Select a Prompt Title", prompt_titles, index=None, key=None)

if prompt_title_selection is not None:
    
    
    lessons_df = lessons_df_merged
    lessons_df = lessons_df[lessons_df['prompt_title'] == prompt_title_selection]
    lessons_df_lp_grouped = lessons_df.groupby(['lesson_plan_id','prompt_title','prompt_id']).agg({
        'justification_count': 'sum',
        'score_1_count': 'sum',
        'score_2_count': 'sum',
        'score_3_count': 'sum',
        'score_4_count': 'sum',
        'score_5_count': 'sum',
        'min_result': 'mean',
        'max_result': 'mean',
        'stellar_success_rate': 'mean',
        'catastrophic_fail_rate': 'mean',
        'overall_fail_score': 'mean'
    }).reset_index()
    #drop rows where justification_count is less than 2
    
    lessons_df_lp_grouped = lessons_df_lp_grouped[lessons_df_lp_grouped['justification_count'] > threshold]
    
    # lessons_df_lp_grouped['prompt_title'] = prompt_title_selection


    if 'selected_filtered_df' not in st.session_state:
        st.session_state.selected_filtered_df = None

    # Define the column configuration for the dataframe display
    column_configuration = {
        "lesson_plan_id": st.column_config.TextColumn(
            "Lesson Plan ID", help="The unique identifier for the lesson plan", width="medium"
        ),
        "justification_count": st.column_config.NumberColumn(
            "Justification Count", help="Number of justifications provided", width="small"
        ),
        "score_1_count": st.column_config.NumberColumn(
            "Score 1 Count", help="Count of scores with value 1", width="small"
        ),
        "score_2_count": st.column_config.NumberColumn(
            "Score 2 Count", help="Count of scores with value 2", width="small"
        ),
        "score_3_count": st.column_config.NumberColumn(
            "Score 3 Count", help="Count of scores with value 3", width="small"
        ),
        "score_4_count": st.column_config.NumberColumn(
            "Score 4 Count", help="Count of scores with value 4", width="small"
        ),
        "score_5_count": st.column_config.NumberColumn(
            "Score 5 Count", help="Count of scores with value 5", width="small"
        ),
        # Add other columns as needed with appropriate configuration
    }

    # Tabs for selection and comparison
    select, compare = st.tabs(["Select Lesson Plans", "Compare Selected"])

    with select:
        st.header("All Lesson Plans")

        # Display the DataFrame with selectable rows
        event = st.dataframe(
            lessons_df_lp_grouped,
            column_config=column_configuration,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",  # Allow multiple rows to be selected
        )

        # Display selected rows
        st.header("Selected Lesson Plans")
        selected_rows = event.selection.rows
        if selected_rows:
            st.session_state.selected_filtered_df = lessons_df_lp_grouped.iloc[selected_rows]
            st.dataframe(
                st.session_state.selected_filtered_df,
                column_config=column_configuration,
                use_container_width=True,
            )
        else:
            st.markdown("No lesson plans selected.")

    with compare:
        if selected_rows and len(selected_rows) > 1:
            # Example of comparing some metrics
            st.header("Justification Count Vs Stellar Success Comparison")
            justification_df = st.session_state.selected_filtered_df.set_index('justification_count')['stellar_success_rate']
            st.bar_chart(justification_df)

            st.header("Score Distribution Comparison")
            score_df = st.session_state.selected_filtered_df[['stellar_success_rate','catastrophic_fail_rate','overall_fail_score']].set_index(st.session_state.selected_filtered_df['lesson_plan_id'])
            st.line_chart(score_df)
        else:
            st.markdown("No lesson plans selected. Select more than one lesson plan to compare.")
    

    
    if 'selected_plan' not in st.session_state:
        st.session_state.selected_plan = None

    if 'justifications' not in st.session_state:
        st.session_state.justifications = None


    st.write("## Lesson Details")

    if st.session_state.selected_filtered_df is not None:
        for i, row in st.session_state.selected_filtered_df.iterrows():
            with st.expander(f"Lesson {i + 1}: {row['prompt_title']}"):
                col1, col2 = st.columns(2)

                # Add content to the first column
                with col1:
                    # st.header("Column 1")
                    # st.write(f"**Generation Details:** {row['generation_details']}")
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
                
                

        lessons_df=st.session_state.selected_filtered_df
        lessons_df

        if st.session_state.selected_plan is not None:
            st.session_state.selected_plan
        # Fetch the selected row using session_state.selected_plan
            selected_row = lessons_df_backup[lessons_df_backup['lesson_plan_id'] == st.session_state.selected_plan]
            selected_row
            selected_row = selected_row.iloc[0]
            selected_row

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

    else:
        st.write("No lesson plan selected.")