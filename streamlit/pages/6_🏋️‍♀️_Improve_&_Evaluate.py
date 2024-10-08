import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import plotly.express as px
import re

from utils.common_utils import  log_message,clear_all_caches, calculate_success_failure_rate
from utils.db_scripts import (
    fetch_bad_lesson_plans, fetch_result_data, 
    fetch_lesson_plan_json, insert_single_lesson_plan, 
    get_prompt, fetch_prompt_objectives_desc,  
    new_sample, 
    add_lesson_plan_to_sample, 
    start_experiment, 
    fetch_final_data, 
    delete_created_sample, delete_lesson_plans_from_sample_lesson_plans
)

from utils.inference import run_agent_openai_inference

import json
import matplotlib.pyplot as plt
import warnings

    

#TODO:  
#       -Add the ability to select multiple lesson plans for improvement
#       -More verbal instructions and explanations
#       -Add the ability to select multiple lesson plans for comparison



warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy")


load_dotenv()

    
    
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')


#set page logo

st.markdown("## 🏋️‍♀️ Improve & Evaluate Lesson Plans")
st.write(
    """
    This page allows you to select a prompt and improve the lesson plan based on the evaluation results.
    You can select a prompt and view the lesson plans that have received the lowest scores.
    You can then select a lesson plan to improve and evaluate the improved lesson plan.
    """
)

prompt_objectives = fetch_prompt_objectives_desc()

prompt_objective_descriptions =prompt_objectives['objective_desc'].unique()

prompt_objective_selection = st.selectbox("Select a Prompt Objective", prompt_objective_descriptions, index=0, key=None)

selected_prompts = prompt_objectives[prompt_objectives['objective_desc'] == prompt_objective_selection]
# selected_prompts['id'] 

#assign selected_prompts['id'] to a variable
selected_prompt_ids = selected_prompts['id'].values
selected_prompt_ids = tuple(selected_prompt_ids)

#make lessons df empty
# lessons_df = None


# Fetch the data
lessons_df =fetch_bad_lesson_plans(selected_prompt_ids)

#order the lessons_df by lesson_plan_id and prompt_id
lessons_df = lessons_df.sort_values(by=['lesson_plan_id', 'prompt_id'])

processed_df = calculate_success_failure_rate(lessons_df)
# order the processed_df by lesson_plan_id and prompt_id
processed_df = processed_df.sort_values(by=['lesson_plan_id', 'prompt_id'])
# processed_df

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
    'min_result': 'min',
    'max_result': 'max',
    'stellar_success_rate': 'mean',
    'catastrophic_fail_rate': 'mean',
    'overall_fail_score': 'mean',
    'prompt_id': lambda x: ', '.join(set(x))
}).reset_index()


st.header('Prompts with the lowest scores')
# Specify the desired column order
desired_order_prompts = [
    'prompt_title',
    'overall_fail_score',
    'catastrophic_fail_rate',
    'stellar_success_rate',
    'justification_count',
    'min_result',
    'max_result',
    'score_1_count',
    'score_2_count',
    'score_3_count',
    'score_4_count',
    'score_5_count',
    'lesson_plan_id',
    'prompt_id'
]

# Reorder the columns in the DataFrame
lessons_df_p_grouped_reordered = lessons_df_p_grouped[desired_order_prompts]
lessons_df_p_grouped_reordered = lessons_df_p_grouped_reordered.sort_values(by='catastrophic_fail_rate', ascending=False)
# Display the reordered DataFrame
st.dataframe(lessons_df_p_grouped_reordered)




# st.write(f"##### number of rows in the merged dataframe: {lessons_df_merged.shape[0]}")



# # set a threshold for the justification count
threshold = 1
# # allow user to select the threshold by inputting a value
# threshold = st.number_input("Enter the threshold for the justification count", min_value=1, value=1, step=1)

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
        'min_result': 'min',
        'max_result': 'max',
        'stellar_success_rate': 'mean',
        'catastrophic_fail_rate': 'mean',
        'overall_fail_score': 'mean'
    }).reset_index()
    
    desired_order_prompts = [
    'lesson_plan_id',
    'justification_count',
    'overall_fail_score',
    'catastrophic_fail_rate',
    'stellar_success_rate',
    'min_result',
    'max_result',
    'score_1_count',
    'score_2_count',
    'score_3_count',
    'score_4_count',
    'score_5_count',
    'prompt_id',
    'prompt_title',
]

    lessons_df_lp_grouped = lessons_df_lp_grouped[desired_order_prompts]
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
    }


   

    # Tabs for selection and comparison
    select, compare = st.tabs(["Select Lesson Plans", "Compare Selected"])
    
    with select:
        st.header("Select Plans to Review Their Results")
        st.write("⬇️ You can select multiple lesson plans to review.")

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
        # st.header("Selected Lesson Plans")
        selected_rows = event.selection.rows
        if selected_rows:
            st.session_state.selected_filtered_df = lessons_df_lp_grouped.iloc[selected_rows]
            # st.dataframe(
            #     st.session_state.selected_filtered_df,
            #     column_config=column_configuration,
            #     use_container_width=True,
            # )
        # else:
        #     st.markdown("No lesson plans selected.")

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


    st.write("## Lesson Eval Result Details")

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
        # st.write('lessons_df')
        # lessons_df

        if st.session_state.selected_plan is not None:
            # st.session_state.selected_plan
        # Fetch the selected row using session_state.selected_plan
            selected_row = lessons_df[lessons_df['lesson_plan_id'] == st.session_state.selected_plan]
            # selected_row['prompt_id'].iloc[0]
            # st.session_state.selected_plan
            # selected_row


            # st.write('selected_row',selected_row)
            # st.write('selected id', selected_row['lesson_plan_id'].iloc[0])
            if not selected_row.empty:
                if selected_row['lesson_plan_id'].iloc[0] is not None:
                    lesson_plan_json_df = fetch_lesson_plan_json(selected_row['lesson_plan_id'].iloc[0])
                    if lesson_plan_json_df is not None and not lesson_plan_json_df.empty:
                        lesson_plan_json = lesson_plan_json_df['json'].iloc[0]
                    # st.write(lesson_plan_json)

                else:
                    st.error("No JSON data found for the selected lesson plan.")
                
                selected_lesson_plan_id = selected_row.iloc[0]
                # selected_lesson_plan_id
                # selected_lesson_plan_id 0c318c0e-56a2-48a0-a78a-0e68a85aa204



                lesson_plan = lesson_plan_json

                # lesson_plan

                prompt_details =None
                

                prompt_id = selected_row['prompt_id'].iloc[0]
                prompt_details = get_prompt(prompt_id)
                # st.write("Lesson Plan:", lesson_plan)
                if prompt_details is not None:
                    
                    
                    with st.expander('Full prompt that generated the Eval Results'):
                        st.write("Prompt Details:", prompt_details)
            
                llm_model = 'gpt-4o'
                llm_model_temp = 0.5


                improvement_prompt = (
                "You are an expert lesson plan improvement agent." 
                +"You will be provided with a lesson plan that has received a low score due to failing to meet the required standards. "
                +"Your task is to improve the lesson plan by making necessary changes to improve the lesson plan.\n\n"
                + "Here is the lesson plan:\n\n  "
                + lesson_plan_json
                + "\n\n"
                +"The lesson plan has received the following review:\n\n  "
                + "\n\n"
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
                with st.expander('Improvement Prompt'):
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
                llm_model_temp = 1
                llm_model = 'gpt-4o'
                judge_llm_model = 'gpt-4o'
                #make a selection for llm model selection
                llm_model = st.selectbox("Select an LLM Model for improvement", ['gpt-4o', 'gpt-4o-mini','o1-preview-2024-09-12','o1-mini-2024-09-12',], index=0, key=None)

                # experiment_id = None

                if st.button("Perform Improvement"):
                    st.write('Creating a temporary sample for the improved lesson plan')
                    sample_id = new_sample(sample_title, created_by)
                    if sample_id is not None:
                        st.write("Temporary Sample created successfully!")

                        st.write("Running Improvement on the lesson plan")
                        response_n = run_agent_openai_inference(improvement_prompt, llm_model, llm_model_temp, timeout=150)
                        if response_n['response'] is not None:
                            st.write(f"**Improvement took {response_n['response_time']:.2f} seconds using {llm_model} model**")
                            with st.expander('Improved Plan'):
                                st.write(response_n['response'])
                                response = response_n['response']
                                response = json.dumps(response)  # Ensure the response is correctly formatted as JSON
                        
                                response_cleaned = re.sub(r'\\n|\\r', '', response)
                        
                            # st.write('Lesson Plan Improved!')
                            st.write('Inserting the improved lesson plan into the database')
                            # st.write('response_cleaned',response_cleaned)
                            
                            
                            lesson_plan_id = insert_single_lesson_plan(response_cleaned, selected_lesson_plan_id.iloc[0], key_stage=None, subject=None,  generation_details='improved_lesson_plan')
                            if lesson_plan_id is not None:
                                st.success("Improved Lesson plan added to db successfully!")
                                st.write("Lesson Plan ID:", lesson_plan_id)

                                st.write("Adding lesson plan to the temporary sample")

                                if add_lesson_plan_to_sample(sample_id, lesson_plan_id):
                                    st.success("Added improved lesson plan to temporary sample successfully!")
                                    st.success("Running Evaluation on the improved lesson plan")
                                    sample_ids = [sample_id]
                                    prompt_ids = [prompt_id]
                                    experiment_id = start_experiment(experiment_name, exp_description, sample_ids, teacher_id,
                                    prompt_ids, limit, judge_llm_model,
                                    tracked, llm_model_temp)
                                
                            
                                    if experiment_id is not None:
                                        st.success("Evaluation on the Improved Lesson plan completed successfully!")
                                        result = fetch_final_data(lesson_plan_id, prompt_id, experiment_id)

                                        # Ensure both values are float for comparison
                                        new_result = float(result['result'].values[0])
                                        previous_result = selected_row['min_result'].iloc[0]

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
                                            st.success("Improved Lesson plan performed better than the previous version!")
                                        elif previous_result > new_result:
                                            st.error("Oops! It seems we couldn't improve the lesson plan, we made it worse!")
                                        else:
                                            st.warning("Improved lesson plan received the same score as old one.")

            else:
                st.write('Please select a lesson plan')
            # st.write('lessonplanjson', lesson_plan_json_df)

            
            
    else:
        st.write("No lesson plan selected.")

                                    

            