""" 
Streamlit page for visualising the results of evaluations in the 
AutoEval app.
    
Functionality:
- Visualize evaluation results using interactive plots.
- Filter data based on various parameters.
- Display specific details for selected runs.
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils.formatting import standardize_key_stage, standardize_subject, json_to_html
from utils.common_utils import clear_all_caches
from utils.db_scripts import execute_single_query, get_light_experiment_data, get_full_experiment_data


def fetch_and_preprocess_light_data():
    """
    Fetches and pre-process light experiment data.

    This function retrieves light experiment data and performs the
    following preprocessing steps:

    - Replaces parentheses in the 'experiment_name' column with square
        brackets.
    - Converts 'run_date' column to a string format (YYYY-MM-DD).
    - Creates a new column 'experiment_with_date' combining
        'experiment_name', 'run_date', and 'teacher_name'.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed light
            experiment data.
    """
    light_data = get_light_experiment_data()
    light_data["experiment_name"] = (
        light_data["experiment_name"]
        .str.replace("(", "[", regex=False)
        .str.replace(")", "]", regex=False)
    )
    light_data["run_date"] = pd.to_datetime(light_data["run_date"]).dt.strftime(
        "%Y-%m-%d"
    )

    light_data["experiment_with_date"] = light_data.apply(
        lambda x: (f"{x['experiment_name']} ({x['run_date']}) ({x['teacher_name']})"),
        axis=1,
    )
    return light_data


def fetch_and_preprocess_full_data(selected_experiment_id):
    """
    Fetches and pre-process full experiment data.

    This function retrieves full experiment data for a given experiment
    ID and performs the following preprocessing steps:

    - Standardizes the 'key_stage_slug' and 'subject_slug' columns.
    - Sorts the data by 'run_date' in descending order.
    - Converts 'run_date' column to a string format (YYYY-MM-DD).

    Args:
        selected_experiment_id (int): The ID of the selected experiment.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed full
            experiment data.

    """
    data = get_full_experiment_data(selected_experiment_id)
    data["key_stage_slug"] = data["key_stage_slug"].apply(standardize_key_stage)
    data["subject_slug"] = data["subject_slug"].apply(standardize_subject)
    data["prompt_title_version"] = (
        data["prompt_title"] + " v" + data["prompt_version"].astype(str)
    )
    data = data.sort_values(by="run_date", ascending=False)
    data["run_date"] = pd.to_datetime(data["run_date"]).dt.strftime("%Y-%m-%d")
    return data


def apply_filters(data):
    """
    Apply filters to the experiment data based on user selections.

    This function provides multiple selection options for teachers,
    prompts, and samples to filter the experiment data. If no filters
    are applied, it displays all the data.

    Args:
        data (pandas.DataFrame): The experiment data to be filtered.

    Returns:
        pandas.DataFrame: The filtered experiment data.
    """
    selected_teachers = st.multiselect(
        "Select Teacher",
        options=data["teacher_name"].unique(),
        help="Select one or more teachers to filter the experiments.",
    )

    data["prompt_title_version"] = (
        data["prompt_title"] + " v" + data["prompt_version"].astype(str)
    )

    selected_prompts = st.multiselect(
        "Select Prompt",
        options=data["prompt_title_version"].unique(),
        help="Select one or more prompts (with version) to filter the experiments.",
    )

    selected_samples = st.multiselect(
        "Select Sample",
        options=data["sample_title"].unique(),
        help="Select one or more samples to filter the experiments.",
    )
    if selected_teachers:
        data = data[data["teacher_name"].isin(selected_teachers)]
    if selected_prompts:
        data = data[data["prompt_title_version"].isin(selected_prompts)]
    if selected_samples:
        data = data[data["sample_title"].isin(selected_samples)]

    if not selected_teachers and not selected_prompts and not selected_samples:
        st.write("No filters applied. Showing all data.")

    return data


def display_pie_chart(data, group_by_column, title, column):
    """
    Display a pie chart for the given data grouped by a specified
    column.

    This function groups the data by the specified column, calculates
    the unique count of 'lesson_plan_id', and displays a pie chart using
    Plotly.

    Args:
        data (pandas.DataFrame): The data to be visualized.
        group_by_column (str): The column to group the data by.
        title (str): The title of the pie chart.
        column (streamlit.DeltaGenerator): The Streamlit column object
            to display the chart in.

    Returns:
        None
    """
    grouped_data = (
        data.groupby(group_by_column)["lesson_plan_id"].nunique().reset_index()
    )
    grouped_data.columns = [group_by_column, "count_of_lesson_plans"]
    if not grouped_data.empty:
        fig = px.pie(
            grouped_data,
            values="count_of_lesson_plans",
            names=group_by_column,
            title=title,
        )
        fig.update_layout(width=350, height=350, showlegend=False)
        fig.update_traces(
            textinfo="label",
            hoverinfo="percent+value",
            hovertemplate="%{label}: %{value} (<b>%{percent}</b>)",
        )
        column.plotly_chart(fig)

def plot_key_stage_subject_heatmap(data, key_stage_col, subject_col, prompt_col, value_col):
    """
    Generates and plots a heatmap showing average scores grouped by key stage, subject, and criteria.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data.
        key_stage_col (str): Column name for key stage.
        subject_col (str): Column name for subject.
        prompt_col (str): Column name for criteria or prompt title/version.
        value_col (str): Column name for success ratio or the value to average.
    """
    # Group and aggregate the data
    grouped_data = data.groupby([key_stage_col, subject_col, prompt_col]).agg({value_col: 'mean'}).reset_index()

    # Pivot the data for heatmap
    heatmap_data = grouped_data.pivot_table(
        index=prompt_col,
        columns=[key_stage_col, subject_col],
        values=value_col
    )

    # Add averages
    heatmap_data['Average'] = heatmap_data.mean(axis=1)
    heatmap_data.loc['Average'] = heatmap_data.mean(axis=0)

    # Plot the heatmap
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': 'Average Score'},
        fmt=".2f"
    )
    plt.title('Average Score per Key Stage, Subject, and Criteria')
    plt.xlabel('Key Stage and Subject')
    plt.ylabel('Criteria')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Render the plot using Streamlit
    st.pyplot(plt)
# Set page configuration
st.set_page_config(page_title="Visualise Results", page_icon="🔍")
st.title("🔍 Visualise Results")

# Sidebar button to clear cache
if st.sidebar.button("Clear Cache"):
    clear_all_caches()
    st.sidebar.success("Cache cleared!")

# Fetch light data
light_data = fetch_and_preprocess_light_data()
experiment_description_options = ["Select"] + light_data[
    "experiment_with_date"
].unique().tolist()

experiment = st.multiselect(
    "Select Experiment",
    experiment_description_options,
    help=(
        "Select experiments to view more options." "(Run Date is in YYYY-MM-DD format)"
    ),
)

# Extract the selected experiment_id
selectected_experiments = []
if experiment != "Select":
    for experiment in experiment:
        selected_experiment_name = experiment.split(" (")[0]
        selected_experiment_id = light_data[
            light_data["experiment_name"] == selected_experiment_name
        ]["experiment_id"].iloc[0]
        selectected_experiments.append(selected_experiment_id)
        result_id_input = ""
else:
    selected_experiment_id = None

if selectected_experiments:
    # Fetch full data
    data = pd.DataFrame()
    for selectected_experiment in selectected_experiments:
        data = pd.concat(
            [data, fetch_and_preprocess_full_data(selectected_experiment)],
            ignore_index=True,
        )

    # Filter data
    if st.checkbox("Filter Experiment Data"):
        data = apply_filters(data)

    exp_data = data

    sample_title_options = exp_data["sample_title"].unique().tolist()

    # Use st.selectbox for single selection
    selected_sample = st.selectbox(
        "Select Sample Title",
        sample_title_options
    )

    # Filter the data based on the selected sample
    exp_data = exp_data[exp_data["sample_title"] == selected_sample]


    if experiment != "Select":
        key_stage_options = exp_data["key_stage_slug"].unique().tolist()
        subject_options = exp_data["subject_slug"].unique().tolist()
        prompt_output_format_options = (
            exp_data["prompt_output_format"].unique().tolist()
        )

        st.write("Please select the filters to view the insights.")
        with st.expander("Key Stage and Subject Filters"):
            key_stage = st.multiselect(
                "Select Key Stage", key_stage_options, default=key_stage_options[:]
            )
            subject = st.multiselect(
                "Select Subject", subject_options, default=subject_options[:]
            )

        st.write("Please select the outcome option to view the insights.")
        output_selection = st.selectbox(
            "Select Outcome Type", prompt_output_format_options
        )

        if output_selection:
            

            exp_data = exp_data[
                (exp_data["sample_title"] == selected_sample)
                & (exp_data["prompt_output_format"] == output_selection)
            ]

            prompt_title_options = exp_data["prompt_title_version"].unique().tolist()
            result_status_options = exp_data["result_status"].unique().tolist()
            result_status = st.multiselect(
                "Select Experiment Status", result_status_options, default="SUCCESS"
            )

            exp_data = exp_data[(exp_data["result_status"].isin(result_status))]
            prompt_title = st.multiselect(
                "Select Prompt Title (with Version)",
                prompt_title_options,
                default=prompt_title_options[:],
            )

            exp_data["result"] = pd.to_numeric(exp_data["result"], errors="coerce")
            exp_data.loc[exp_data["prompt_output_format"] == "Boolean", "result"] = (
                exp_data["result"].map({0: "False", 1: "True"})
            )

            mask = (exp_data["prompt_output_format"] != "Boolean") & (
                exp_data["result"].apply(
                    lambda x: isinstance(x, str) or not isinstance(x, (int, float))
                )
            )
            exp_data.loc[mask, "result"] = exp_data.loc[mask, "result"].astype(float)

            outcome_options = exp_data["result"].unique().tolist()
            if output_selection == "Score":
                outcome_options = sorted(outcome_options)

            outcome = st.multiselect(
                "Filter by Result Outcome", outcome_options, default=outcome_options[:]
            )

            filtered_data = exp_data[
                (exp_data["key_stage_slug"].isin(key_stage))
                & (exp_data["subject_slug"].isin(subject))
                & (exp_data["result_status"].isin(result_status_options))
                & (exp_data["prompt_title_version"].isin(prompt_title))
                & (exp_data["result"].isin(outcome))
            ]

            filtered_data["result_numeric"] = (
                filtered_data["result"]
                .map({"TRUE": 1, "FALSE": 0})
                .fillna(filtered_data["result"])
            )
            filtered_data["result_numeric"] = pd.to_numeric(
                filtered_data["result_numeric"], errors="coerce"
            )

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Number of Lesson Plans in Selection: ",
                filtered_data["lesson_plan_id"].nunique(),
            )
            col2.metric("Evaluator Model", filtered_data["llm_model"].iloc[0])

            display_pie_chart(
                filtered_data, "key_stage_slug", "Lesson Plans by Key Stage", col1
            )
            display_pie_chart(
                filtered_data, "subject_slug", "Lesson Plans by Subject", col2
            )

            # Sidebar for justification lookup
            st.sidebar.header("Justification Lookup")
            st.sidebar.write(
                "Please copy the result_id from the Data Viewer to "
                "view the justification."
            )
            result_id_input = st.sidebar.text_input("Enter Result ID")

            if result_id_input:
                result_data = filtered_data[
                    filtered_data["result_id"] == result_id_input
                ]
                if not result_data.empty:
                    lesson_plan_id = result_data["lesson_plan_id"].iloc[0]
                    lesson_plan_data = execute_single_query(
                        "SELECT json FROM lesson_plans WHERE id = %s",
                        (lesson_plan_id,),
                        return_dataframe=True,
                    )

                    if not lesson_plan_data.empty:
                        lesson_plan_json = lesson_plan_data["json"].iloc[0]
                        lesson_plan_dict = json.loads(lesson_plan_json)
                        justification_text = result_data["justification"].iloc[0]
                        lesson_plan_parts = result_data["prompt_lp_params"].iloc[0]

                        st.sidebar.header(f"Lesson Plan ID: {lesson_plan_id}")
                        st.sidebar.header("Justification for Selected Run")
                        st.sidebar.write(f"**Justification**: {justification_text}")
                        st.sidebar.header("**Relevant Lesson Parts**")
                        st.sidebar.write(f"{lesson_plan_parts}")

                        st.sidebar.header("**Relevant Lesson Plan Parts Extracted**:")
                        html_text = ""
                        for key, value in lesson_plan_dict.items():
                            if key in lesson_plan_parts:
                                if isinstance(value, (str, int, float)):
                                    html_text += (
                                        f"<p><strong>{key}</strong>: " f"{value}</p>"
                                    )
                                else:
                                    html_text += (
                                        f"<p><strong>{key}</strong>:</p>"
                                        f"{json_to_html(value, indent=1)}"
                                    )
                        st.sidebar.markdown(html_text, unsafe_allow_html=True)

                        st.sidebar.header("Relevant Lesson Plan")
                        lesson_plan_text = json_to_html(lesson_plan_dict, indent=0)
                        st.sidebar.markdown(lesson_plan_text, unsafe_allow_html=True)

                    else:
                        st.sidebar.error("No data found for this Run ID")
                else:
                    st.sidebar.error("No data found for this Run ID")

            color_mapping = {
                "Gen-claude-3-opus-20240229-0.3": "lightgreen",
                "Gen-gemini-1.5-pro-1": "lightblue",
                "Gen-gpt-4-turbo-0.5": "pink",
                "Gen-gpt-4-turbo-0.7": "lightcoral",
                # Add more mappings as needed
            }

            # Result_data
            # Assuming the ideal score is 5
            if output_selection == "Score":
                IDEAL_SCORE = 5
                # Calculate the percentage success for each result
                filtered_data["result_percent"] = (
                    filtered_data["result_numeric"] / IDEAL_SCORE
                ) * 100

                # Ensure result_percent is within 0 to 100
                success_filtered_data = filtered_data[
                    (filtered_data["result_percent"] >= 0)
                    & (filtered_data["result_percent"] <= 100)
                ]

                # Calculate the average success rate for each prompt
                # title and sample title
                average_success_rate = (
                    success_filtered_data.groupby(
                        ["prompt_title_version", "sample_title"]
                    )["result_percent"]
                    .mean()
                    .reset_index()
                )

                # Calculate the overall average success rate for each
                # prompt title
                overall_success_rate = (
                    success_filtered_data.groupby("prompt_title_version")[
                        "result_percent"
                    ]
                    .mean()
                    .reset_index()
                )

                # Display the diagnostics to ensure data is correct
                with st.expander("Success Rate Diagnostics"):
                    st.write(
                        "Diagnostics: After grouping and calculating "
                        "average success rate"
                    )
                    st.write(average_success_rate)

                # Creating a layered spider chart
                fig = go.Figure()

                unique_samples = average_success_rate["sample_title"].unique()
                num_samples = len(unique_samples)
                # Generate a list of colors using Plotly's color palette
                colors = px.colors.qualitative.Plotly[:num_samples]

                for i, sample in enumerate(unique_samples):
                    sample_data = average_success_rate[
                        average_success_rate["sample_title"] == sample
                    ]
                    categories = sample_data["prompt_title_version"].tolist()
                    values = sample_data["result_percent"].tolist()

                    # Append the first value to the end to close the
                    # circle in the spider chart
                    values += values[:1]
                    categories += categories[:1]

                    # Add trace for each sample with a different color
                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill="toself",
                            name=sample,
                            # Assign color from the generated list
                            line=dict(color=colors[i % num_samples]),
                        )
                    )

                # Add overall success rate as a dotted line plot
                overall_categories = overall_success_rate[
                    "prompt_title_version"
                ].tolist()
                overall_values = overall_success_rate["result_percent"].tolist()

                # Append the first value to the end to close the circle
                # in the spider chart
                overall_values += overall_values[:1]
                overall_categories += overall_categories[:1]

                fig.add_trace(
                    go.Scatterpolar(
                        r=overall_values,
                        theta=overall_categories,
                        mode="lines",
                        line=dict(color="black", dash="dot"),
                        name="Overall Average",
                    )
                )

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title=("Average Success Rate by Prompt Title " "and Sample Title"),
                )

                st.plotly_chart(fig)

                plot_key_stage_subject_heatmap(
                    data=filtered_data,  
                    key_stage_col='key_stage_slug',
                    subject_col='subject_slug',
                    prompt_col='prompt_title',
                    value_col='success_ratio'
                )

            else:
                # Convert 'True' and 'False' strings to boolean values
                # in the 'result' column
                filtered_data["success"] = filtered_data["result"].map(
                    {"True": True, "False": False}
                )
                

                # Calculate the count of successful results compared
                # to the total number of results per prompt title and
                # sample title
                count_of_results = (
                    filtered_data.groupby(["prompt_title_version", 'key_stage_slug', 'subject_slug',"sample_title"])[
                        "success"
                    ]
                    .value_counts()
                    .unstack(fill_value=0)
                    .reset_index()
                )

                # Ensure both 'True' and 'False' columns exist
                if True not in count_of_results.columns:
                    count_of_results[True] = 0
                if False not in count_of_results.columns:
                    count_of_results[False] = 0

                # Calculate total and success ratio
                count_of_results["total"] = (
                    count_of_results[True] + count_of_results[False]
                )
                count_of_results["success_ratio"] = (
                    count_of_results[True] / count_of_results["total"]
                ) * 100

                # Display the diagnostics to ensure data is correct
                st.write("Diagnostics: After grouping and calculating " "success ratio")
                st.write(count_of_results)

                # Creating a bar chart
                fig = px.bar(
                    count_of_results,
                    x="sample_title",
                    y="success_ratio",
                    color="prompt_title_version",
                    title="Success Ratio by Sample Title and Prompt Title",
                    text="success_ratio",
                )
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(
                    xaxis_title="Sample Title", yaxis_title="Success Ratio (%)"
                )
                st.plotly_chart(fig)

                plot_key_stage_subject_heatmap(
                    data=count_of_results,  
                    key_stage_col='key_stage_slug',
                    subject_col='subject_slug',
                    prompt_col='prompt_title_version',
                    value_col='success_ratio'
                )


            # Display title and data table in the main area
            st.subheader("Experiment Data Viewer")

            filtered_data = filtered_data[
                [
                    "result_id",
                    "result",
                    "sample_title",
                    "prompt_title_version",
                    "result_status",
                    "justification",
                    "key_stage_slug",
                    "subject_slug",
                    "lesson_plan_id",
                    "run_date",
                    "prompt_lp_params",
                ]
            ]
            st.dataframe(filtered_data)

        else:
            st.write("Please select an experiment to see more options.")
