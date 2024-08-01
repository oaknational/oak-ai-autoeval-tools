import streamlit as st
import psycopg2
import pandas as pd
import os 
from dotenv import load_dotenv
import plotly.express as px
import numpy as np
import json
from dataeditor import * 
import plotly.graph_objects as go
from jinja_funcs import *

st.set_page_config(page_title="Visualise Results", page_icon="🔍")

st.markdown("# 🔍 Visualise Results")

load_dotenv()
# Function to clear cache
def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# Add a button to the sidebar to clear cache
if st.sidebar.button('Clear Cache'):
    clear_all_caches()
    st.sidebar.success('Cache cleared!')

# Fetch light data
light_data = get_light_experiment_data()
#Replacing paranthesis with brackets to allow string matching
light_data['experiment_name'] = light_data['experiment_name'].str.replace('(', '[', regex=False).str.replace(')', ']', regex=False)

light_data['run_date'] = pd.to_datetime(light_data['run_date']).dt.strftime('%Y-%m-%d')
light_data['experiment_with_date'] = light_data['experiment_name'] + " (" + light_data['run_date'].astype(str) + ")"+ " (" + light_data['teacher_name'] + ")"
experiment_description_options = light_data['experiment_with_date'].unique().tolist()
experiment_description_options = ['Select'] + experiment_description_options

experiment = st.multiselect('Select Experiment', experiment_description_options, help="Select an experiment to view more options.(Run Date is in YYYY-MM-DD format)")
# experiment

selectected_experiments = []
if experiment != 'Select':
    for experiment in experiment:

        selected_experiment_name = experiment.split(" (")[0]
        selected_experiment_id = light_data[light_data['experiment_name'] == selected_experiment_name]['experiment_id'].iloc[0]
        selectected_experiments.append(selected_experiment_id)
        result_id_input =''
else:
    selected_experiment_id = None

if selectected_experiments:
    #define empty df data
    data = pd.DataFrame()
    for selectected_experiment in selectected_experiments:
        
        #concat get_full_experiment_data(selected_experiment_id) to data
        data = pd.concat([data, get_full_experiment_data(selectected_experiment)], ignore_index=True)

    # data
    # Apply transformations on data
    
    data['key_stage_slug'] = data['key_stage_slug'].apply(standardize_key_stage)
    data['subject_slug'] = data['subject_slug'].apply(standardize_subject)
    data = data.sort_values(by='run_date', ascending=False)
    data['run_date'] = pd.to_datetime(data['run_date']).dt.strftime('%Y-%m-%d')
    
    # Filter data as needed
    if st.checkbox('Filter Experiment Data'):
        selected_teachers = st.multiselect('Select Teacher', options=data['teacher_name'].unique(), help="Select one or more teachers to filter the experiments.")
        selected_prompts = st.multiselect('Select Prompt', options=data['prompt_title'].unique(), help="Select one or more prompts to filter the experiments.")
        selected_samples = st.multiselect('Select Sample', options=data['sample_title'].unique(), help="Select one or more samples to filter the experiments.")
        
        if selected_teachers or selected_prompts or selected_samples:
            if selected_teachers:
                data = data[data['teacher_name'].isin(selected_teachers)]
            if selected_prompts:
                data = data[data['prompt_title'].isin(selected_prompts)]
            if selected_samples:
                data = data[data['sample_title'].isin(selected_samples)]
        else:
            st.write("No filters applied. Showing all data.")

    exp_data = data

    
    if experiment != 'Select':
        key_stage_options = exp_data['key_stage_slug'].unique().tolist()
        subject_options = exp_data['subject_slug'].unique().tolist()
        result_sucess_options = exp_data['result_status'].unique().tolist()
        llm_model_options = exp_data['llm_model'].unique().tolist()
        
        prompt_lp_params_options = exp_data['prompt_lp_params'].unique().tolist()
        # objective_title_options = exp_data['objective_title'].unique().tolist()
        prompt_output_format_options = exp_data['prompt_output_format'].unique().tolist()
        
        st.write(
            """Please select the filters to view the insights."""
        )
        with st.expander("Key Stage and Subject Filters"):

            key_stage = st.multiselect('Select Key Stage', key_stage_options, default=key_stage_options[:])
            subject = st.multiselect('Select Subject', subject_options, default=subject_options[:])

        st.write("""Please select the outcome option to view the insights.""")
        output_selection = st.selectbox('Select Outcome Type', prompt_output_format_options)

        if output_selection:
            
            sample_title_options = exp_data['sample_title'].unique().tolist()

            selected_samples = st.multiselect('Select Sample Title', sample_title_options, default=sample_title_options[:])

            exp_data = exp_data[
            (exp_data['sample_title'].isin(selected_samples))]

            exp_data = exp_data[
            (exp_data['prompt_output_format'] == output_selection)]
            prompt_title_options = exp_data['prompt_title'].unique().tolist()

            result_status_options = exp_data['result_status'].unique().tolist()
            result_status = st.multiselect('Select Experiment Status', result_status_options, default='SUCCESS')

            exp_data = exp_data[(exp_data['result_status'].isin(result_status))]
            
            prompt_title = st.multiselect('Select Prompt Title', prompt_title_options, default=prompt_title_options[:])
            # outcome_options
            #make outcome options integers if possible if not make them floats
            exp_data['result'] = pd.to_numeric(exp_data['result'], errors='coerce')

            # Check each row of the exp_data value of prompt_output_format and if it is Boolean, convert the exp_data['result'] 0 to False and 1 to True
            exp_data.loc[exp_data['prompt_output_format'] == 'Boolean', 'result'] = exp_data['result'].map({0: 'False', 1: 'True'})
            mask = (exp_data['prompt_output_format'] != 'Boolean') & (exp_data['result'].apply(lambda x: isinstance(x, str) or not isinstance(x, (int, float))))
            exp_data.loc[mask, 'result'] = exp_data.loc[mask, 'result'].astype(float)
            
            outcome_options = exp_data['result'].unique().tolist()

            # outcome_options
            if output_selection == 'Score':
                outcome_options = sorted(outcome_options)
        
            outcome = st.multiselect('Filter by Result Outcome', outcome_options, default=outcome_options[:])

            filtered_data = exp_data[
                (exp_data['key_stage_slug'].isin(key_stage)) &
                (exp_data['subject_slug'].isin(subject)) &
                (exp_data['result_status'].isin(result_status_options)) &
                (exp_data['prompt_title'].isin(prompt_title)) &
                (exp_data['result'].isin(outcome))
            ]
            
            # st.table(filtered_data)
            # filtered_data['result']
            filtered_data['result_numeric'] = np.where(filtered_data['result'] == "TRUE", 1,
                                            np.where(filtered_data['result'] == "FALSE", 0, filtered_data['result']))
            # filtered_data['result_numeric']
            filtered_data['result_numeric'] = pd.to_numeric(filtered_data['result_numeric'], errors='coerce')
            
     
            # success_ratio
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Lesson Plans in Selection: ", filtered_data['lesson_plan_id'].nunique())
            # Write llm model into col2
            col2.metric("Evaluator Model", filtered_data['llm_model'].iloc[0])
           
            col1, col2 = st.columns([1,1])

            if not filtered_data.empty:
                # Aggregating data for the 'Key Stage' pie chart
                lesson_plans_by_stage = filtered_data.groupby('key_stage_slug')['lesson_plan_id'].nunique().reset_index()
                lesson_plans_by_stage.columns = ['key_stage_slug', 'count_of_lesson_plans']

                if not lesson_plans_by_stage.empty:
                    # Creating the 'Key Stage' pie chart
                    ks_fig = px.pie(lesson_plans_by_stage, values='count_of_lesson_plans', names='key_stage_slug',
                                    title='Lesson Plans by Key Stage')
                    ks_fig.update_layout(width=350, height=350, showlegend=False)
                    ks_fig.update_traces(textinfo='label',
                                         hoverinfo='percent+value',  # Shows label, percent, and value on hover
                                         hovertemplate='%{label}: %{value} (<b>%{percent}</b>)'  # Custom hover template
                                         )

                    with col1:
                        st.plotly_chart(ks_fig)

                # Aggregating data for the 'Subject' pie chart
                lesson_plans_by_subject = filtered_data.groupby('subject_slug')['lesson_plan_id'].nunique().reset_index()
                lesson_plans_by_subject.columns = ['subject_slug', 'count_of_lesson_plans']

                if not lesson_plans_by_subject.empty:
                    # Creating the 'Subject' pie chart
                    s_fig = px.pie(lesson_plans_by_subject, values='count_of_lesson_plans', names='subject_slug',
                                   title='Lesson Plans by Subject')
                    s_fig.update_layout(width=350, height=350, showlegend=False)
                    s_fig.update_traces(textinfo='label',
                                        hoverinfo='percent+value',  # Shows label, percent, and value on hover
                                        hovertemplate='%{label}: %{value} (<b>%{percent}</b>)'  # Custom hover template
                                        )

                    with col2:
                        st.plotly_chart(s_fig)

            # Sidebar for justification lookup
            st.sidebar.header('Justification Lookup')
            st.sidebar.write(
                """Please copy the result_id from the Data Viewer to view the justification."""
            )
            result_id_input = st.sidebar.text_input('Enter Result ID')
            

            if result_id_input:
                result_data = filtered_data[filtered_data['result_id'] == result_id_input]
                try :
                    lesson_plan_id = result_data['lesson_plan_id'].iloc[0]
                except:
                    lesson_plan_id = ''

                # Fetch lesson plan data from the database
                conn = get_db_connection()

                cur = conn.cursor()
                cur.execute(f"SELECT json FROM lesson_plans WHERE id = '{lesson_plan_id}'")
                lesson_plan_data = cur.fetchone()  # Use fetchone to get a single result
                cur.close()
                conn.close()

                if lesson_plan_data:
                    lesson_plan_json = lesson_plan_data[0]  # Extract the JSON string from the tuple
                    lesson_plan_dict = json.loads(lesson_plan_json)  # Convert JSON string to dictionary

                    if not result_data.empty:
                        st.sidebar.header(f'Lesson Plan ID: {lesson_plan_id}')
                        st.sidebar.header('Justification for Selected Run')
                        justification_text = result_data['justification'].iloc[0]
                        st.sidebar.write(f"**Justification**: {justification_text}")

                        lesson_plan_parts = result_data['prompt_lp_params'].iloc[0]
                        
                        # Ensure lesson_plan_parts is parsed correctly, handle string or dict
                        if isinstance(lesson_plan_parts, str):
                            lesson_plan_keys = json.loads(lesson_plan_parts)  # Parse the keys if they are in string format
                        else:
                            lesson_plan_keys = lesson_plan_parts  # If it's already a dict or list

                        st.sidebar.header("**Relevant Lesson Parts**")
                        st.sidebar.write(f'{lesson_plan_keys}')
                        
                        st.sidebar.header("**Relevant Lesson Plan Parts Extracted**:")
                        
                        # Extract and format the relevant key-value pairs in HTML
                        html_text = ""
                        for key in lesson_plan_keys:
                            value = lesson_plan_dict.get(key, 'Key not found')
                            if isinstance(value, (dict, list)):
                                html_text += f"<p><strong>{key}</strong>:</p>"
                                html_text += json_to_html(value, indent=1)
                            else:
                                html_text += f"<p><strong>{key}</strong>: {value}</p>"

                        # Display the formatted HTML
                        st.sidebar.markdown(html_text, unsafe_allow_html=True)
                        st.sidebar.header('Relevant Lesson Plan')
                        lesson_plan_text = json_to_html(lesson_plan_dict, indent=0)
                        st.sidebar.markdown(lesson_plan_text, unsafe_allow_html=True)

                    else:
                        st.sidebar.error('No data found for this Run ID')
               
            color_mapping = {
                "Gen-claude-3-opus-20240229-0.3": "lightgreen",
                "Gen-gemini-1.5-pro-1": "lightblue",
                "Gen-gpt-4-turbo-0.5": "pink",
                "Gen-gpt-4-turbo-0.7": "lightcoral"
                # Add more mappings as needed
            }
            # st.write(len(filtered_data))
            common_prompt_titles = set(filtered_data[filtered_data['sample_title'] == filtered_data['sample_title'].iloc[0]]['prompt_title'])
            for sample in filtered_data['sample_title'].unique():
                sample_prompt_titles = set(filtered_data[filtered_data['sample_title'] == sample]['prompt_title'])
                common_prompt_titles = common_prompt_titles.intersection(sample_prompt_titles)

            common_prompt_titles = sorted(list(common_prompt_titles))

            # Step 2: Filter data to keep only the common prompt titles
            filtered_radar_data = filtered_data[filtered_data['prompt_title'].isin(common_prompt_titles)]
            # st.write(len(filtered_radar_data))
            # Assuming the ideal score is 5
            if output_selection == 'Score':
                ideal_score = 5
                # Calculate the percentage success for each result
                filtered_radar_data['result_percent'] = (filtered_radar_data['result_numeric'] / ideal_score) * 100
                
                # Ensure result_percent is within 0 to 100
                success_filtered_radar_data = filtered_radar_data[(filtered_radar_data['result_percent'] >= 0) & (filtered_radar_data['result_percent'] <= 100)]
                
                # Calculate the average success rate for each prompt title and sample title
                average_success_rate = success_filtered_radar_data.groupby(['prompt_title', 'sample_title'])['result_percent'].mean().reset_index()

                # Calculate the overall average success rate for each prompt title
                overall_success_rate = success_filtered_radar_data.groupby('prompt_title')['result_percent'].mean().reset_index()

                # Display the diagnostics to ensure data is correct
                with st.expander("Success Rate Diagnostics"):
                    st.write("Diagnostics: After grouping and calculating average success rate")
                    st.write(average_success_rate)

                # Creating a layered spider chart
                fig = go.Figure()

                unique_samples = average_success_rate['sample_title'].unique()
                num_samples = len(unique_samples)
                # Generate a list of colors using Plotly's color palette
                colors = px.colors.qualitative.Plotly[:num_samples]

                for i, sample in enumerate(unique_samples):
                    sample_data = average_success_rate[average_success_rate['sample_title'] == sample]
                    categories = sample_data['prompt_title'].tolist()
                    values = sample_data['result_percent'].tolist()

                    # Append the first value to the end to close the circle in the spider chart
                    values += values[:1]
                    categories += categories[:1]

                    # Add trace for each sample with a different color
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=sample,
                        line=dict(color=colors[i % num_samples])  # Assign color from the generated list
                    ))

                # Add overall success rate as a dotted line plot
                overall_categories = overall_success_rate['prompt_title'].tolist()
                overall_values = overall_success_rate['result_percent'].tolist()

                # Append the first value to the end to close the circle in the spider chart
                overall_values += overall_values[:1]
                overall_categories += overall_categories[:1]

                fig.add_trace(go.Scatterpolar(
                    r=overall_values,
                    theta=overall_categories,
                    mode='lines',
                    line=dict(color='black', dash='dot'),
                    name='Overall Average'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Average Success Rate by Prompt Title and Sample Title"
                )

                st.plotly_chart(fig)
            else:
                ideal_score = 'True'

                # Convert 'True' and 'False' strings to boolean values in the 'result' column
                filtered_data['success'] = filtered_data['result'].map({'True': True, 'False': False})

                # Calculate the count of results that equal to ideal_score (True) compared to total number of results per prompt title
                count_of_results = filtered_data.groupby(['prompt_title', 'sample_title'])['success'].value_counts().unstack(fill_value=0).reset_index()

                # Ensure both 'True' and 'False' columns exist
                if True not in count_of_results.columns:
                    count_of_results[True] = 0
                if False not in count_of_results.columns:
                    count_of_results[False] = 0

                # Calculate total and success ratio
                count_of_results['total'] = count_of_results[True] + count_of_results[False]
                count_of_results['success_ratio'] = (count_of_results[True] / count_of_results['total']) * 100

                # Display the diagnostics to ensure data is correct
                st.write("Diagnostics: After grouping and calculating success ratio")
                st.write(count_of_results)

                # Creating a bar chart
                fig = px.bar(count_of_results, x='sample_title', y='success_ratio', color='prompt_title', title="Success Ratio by Sample Title and Prompt Title", text='success_ratio')
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(xaxis_title='Sample Title', yaxis_title='Success Ratio (%)')
                st.plotly_chart(fig)
                
            # Display title and data table in the main area
            st.title('Experiment Data Viewer')
            
            filtered_data = filtered_data[['result_id', 'result','sample_title', 'prompt_title','result_status','justification','key_stage_slug', 
                                        'subject_slug','lesson_plan_id', 'run_date',
                                                'prompt_lp_params']]
            st.dataframe(filtered_data)

        else: 
            st.write("Please select an experiment to see more options.")
