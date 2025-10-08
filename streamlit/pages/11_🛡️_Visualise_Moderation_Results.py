""" 
Streamlit page for visualising the results of moderation experiments in the 
AutoEval app.
    
Functionality:
- Visualize moderation results using interactive plots and charts.
- Filter data based on various parameters (experiments, categories, scores).
- Display specific details for selected moderation runs.
- Show category-wise breakdowns and flagged content analysis.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

from utils.formatting import standardize_key_stage, standardize_subject, json_to_html
from utils.common_utils import clear_all_caches
from utils.db_scripts import execute_single_query, get_db_connection
from utils.moderation_utils import load_moderation_categories, process_moderation_categories


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_moderation_experiments_data():
    """
    Fetch moderation experiment data from the database.
    
    Returns:
        pandas.DataFrame: DataFrame containing moderation experiment information
    """
    query = """
        SELECT DISTINCT 
            e.id as experiment_id,
            e.experiment_name,
            e.created_at as run_date,
            e.created_by,
            e.llm_model,
            e.llm_model_temp,
            e.status,
            e.description,
            COUNT(mr.id) as total_results,
            COUNT(CASE WHEN mr.status = 'SUCCESS' THEN 1 END) as successful_results,
            COUNT(CASE WHEN mr.status = 'FAILURE' THEN 1 END) as failed_results
        FROM m_experiments e
        LEFT JOIN m_results mr ON e.id = mr.experiment_id
        WHERE e.experiment_name ILIKE '%moderation%' 
           OR e.experiment_name ILIKE '%Moderating%'
           OR e.description ILIKE '%moderation%'
           OR e.description ILIKE '%Moderating%'
        GROUP BY e.id, e.experiment_name, e.created_at, e.created_by, 
                 e.llm_model, e.llm_model_temp, e.status, e.description
        ORDER BY e.created_at DESC;
    """
    
    try:
        conn = get_db_connection()
        if not conn:
            st.error("Failed to connect to database")
            return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            # Format the date
            df['run_date'] = pd.to_datetime(df['run_date']).dt.strftime('%Y-%m-%d')
            
            # Create experiment display name
            df['experiment_with_date'] = df.apply(
                lambda x: f"{x['experiment_name']} ({x['run_date']}) ({x['created_by']})",
                axis=1
            )
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching moderation experiments: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_moderation_results_data(experiment_ids: List[str]) -> pd.DataFrame:
    """
    Fetch detailed moderation results for selected experiments.
    
    Args:
        experiment_ids: List of experiment IDs to fetch results for
        
    Returns:
        pandas.DataFrame: DataFrame containing detailed moderation results
    """
    if not experiment_ids:
        return pd.DataFrame()
    
    # Convert to string format for SQL IN clause
    experiment_ids_str = "', '".join(experiment_ids)
    
    query = f"""
        SELECT 
            mr.id as result_id,
            mr.experiment_id,
            mr.lesson_plan_id,
            mr.result as scores_json,
            mr.justification,
            mr.status as result_status,
            mr.created_at,
            e.experiment_name,
            e.llm_model,
            e.llm_model_temp,
            e.created_by,
            s.sample_title,
            lp.key_stage,
            lp.subject,
            lp.json as lesson_plan_json
        FROM m_results mr
        JOIN m_experiments e ON mr.experiment_id = e.id
        LEFT JOIN lesson_plans lp ON mr.lesson_plan_id = lp.id
        LEFT JOIN m_sample_lesson_plans slp ON slp.lesson_plan_id = lp.id
        LEFT JOIN m_samples s ON s.id = slp.sample_id
        WHERE mr.experiment_id IN ('{experiment_ids_str}')
        ORDER BY mr.created_at DESC;
    """
    
    try:
        conn = get_db_connection()
        if not conn:
            st.error("Failed to connect to database")
            return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            # Parse JSON scores and justifications
            df = parse_moderation_scores(df)
            df = parse_moderation_justifications(df)
            
            # Standardize key stage and subject
            df['key_stage_slug'] = df['key_stage'].apply(standardize_key_stage)
            df['subject_slug'] = df['subject'].apply(standardize_subject)
            
            # Format date
            df['run_date'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching moderation results: {e}")
        return pd.DataFrame()


def parse_moderation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse JSON scores from the result column into individual category columns.
    
    Args:
        df: DataFrame with scores_json column
        
    Returns:
        pandas.DataFrame: DataFrame with parsed category scores
    """
    try:
        # Load moderation categories to get abbreviations
        categories_data = load_moderation_categories()
        processed_categories, _, _ = process_moderation_categories(categories_data)
        category_abbreviations = [cat['abbreviation'] for cat in processed_categories]
        
        # Parse scores for each row
        parsed_scores = []
        for idx, row in df.iterrows():
            scores_dict = {}
            try:
                if row['scores_json'] and row['scores_json'] != '{}':
                    scores_data = json.loads(row['scores_json'])
                    scores_dict = scores_data
                else:
                    scores_dict = {}
            except (json.JSONDecodeError, TypeError):
                scores_dict = {}
            
            parsed_scores.append(scores_dict)
        
        # Create DataFrame from parsed scores
        scores_df = pd.DataFrame(parsed_scores)
        
        # Add category score columns to main DataFrame
        for abbr in category_abbreviations:
            if abbr in scores_df.columns:
                df[f'score_{abbr}'] = scores_df[abbr]
            else:
                df[f'score_{abbr}'] = None
        
        # Calculate summary statistics
        df['total_categories'] = len(category_abbreviations)
        df['flagged_categories_count'] = df.apply(
            lambda row: sum(1 for abbr in category_abbreviations 
                          if f'score_{abbr}' in df.columns and 
                          pd.notna(row[f'score_{abbr}']) and 
                          row[f'score_{abbr}'] < 5), 
            axis=1
        )
        df['average_score'] = df.apply(
            lambda row: sum(row[f'score_{abbr}'] for abbr in category_abbreviations 
                          if f'score_{abbr}' in df.columns and 
                          pd.notna(row[f'score_{abbr}'])) / 
                       len([abbr for abbr in category_abbreviations 
                           if f'score_{abbr}' in df.columns and 
                           pd.notna(row[f'score_{abbr}'])]), 
            axis=1
        )
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing moderation scores: {e}")
        return df


def parse_moderation_justifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse JSON justifications from the justification column.
    
    Args:
        df: DataFrame with justification column
        
    Returns:
        pandas.DataFrame: DataFrame with parsed justification data
    """
    try:
        justifications_data = []
        flagged_categories_data = []
        summary_data = []
        
        for idx, row in df.iterrows():
            try:
                if row['justification']:
                    justification_json = json.loads(row['justification'])
                    justifications_data.append(justification_json.get('justifications', {}))
                    flagged_categories_data.append(justification_json.get('flagged_categories', []))
                    summary_data.append(justification_json.get('summary', ''))
                else:
                    justifications_data.append({})
                    flagged_categories_data.append([])
                    summary_data.append('')
            except (json.JSONDecodeError, TypeError):
                justifications_data.append({})
                flagged_categories_data.append([])
                summary_data.append('')
        
        df['justifications_dict'] = justifications_data
        df['flagged_categories_list'] = flagged_categories_data
        df['justification_summary'] = summary_data
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing moderation justifications: {e}")
        return df


def create_moderation_score_distribution_chart(data: pd.DataFrame, category_abbr: str) -> None:
    """
    Create a distribution chart for a specific moderation category.
    
    Args:
        data: DataFrame containing moderation results
        category_abbr: Category abbreviation to plot
    """
    score_col = f'score_{category_abbr}'
    
    if score_col not in data.columns:
        st.warning(f"No data available for category: {category_abbr}")
        return
    
    # Filter out null values
    scores_data = data[score_col].dropna()
    
    if scores_data.empty:
        st.warning(f"No valid scores found for category: {category_abbr}")
        return
    
    # Create histogram
    fig = px.histogram(
        scores_data, 
        x=score_col,
        nbins=5,
        title=f"Score Distribution for: {category_abbr}",
        labels={score_col: 'Score', 'count': 'Number of Lesson Plans'},
        color_discrete_sequence=["#ff6b6b"]
    )
    
    fig.update_layout(
        xaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_moderation_heatmap(data: pd.DataFrame, categories: List[str]) -> None:
    """
    Create a heatmap showing average scores across categories and key stages/subjects.
    
    Args:
        data: DataFrame containing moderation results
        categories: List of category abbreviations to include
    """
    if data.empty:
        st.warning("No data available for heatmap")
        return
    
    # Prepare data for heatmap
    heatmap_data = []
    
    KEY_STAGE_COL = 'Key Stage'
    SUBJECT_COL = 'Subject'
    
    for _, row in data.iterrows():
        key_stage = row.get('key_stage_slug', 'Unknown')
        subject = row.get('subject_slug', 'Unknown')
        
        for category in categories:
            score_col = f'score_{category}'
            if score_col in data.columns and pd.notna(row[score_col]):
                heatmap_data.append({
                    KEY_STAGE_COL: key_stage,
                    SUBJECT_COL: subject,
                    'Category': category,
                    'Score': row[score_col]
                })
    
    if not heatmap_data:
        st.warning("No valid data for heatmap")
        return
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Calculate average scores
    avg_scores = heatmap_df.groupby([KEY_STAGE_COL, SUBJECT_COL, 'Category'])['Score'].mean().reset_index()
    
    # Create pivot table for heatmap
    KEY_STAGE_LABEL = 'Key Stage'
    SUBJECT_LABEL = 'Subject'
    pivot_data = avg_scores.pivot_table(
        index=[KEY_STAGE_LABEL, SUBJECT_LABEL], 
        columns='Category', 
        values='Score', 
        fill_value=5.0
    )
    
    # Create heatmap
    fig, _ = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap='RdYlGn', 
        vmin=1, 
        vmax=5,
        fmt='.1f',
        cbar_kws={'label': 'Average Score (1=Heavily Involved, 5=Not Involved)'}
    )
    
    plt.title('Average Moderation Scores by Key Stage and Subject')
    plt.xlabel('Moderation Categories')
    plt.ylabel('Key Stage & Subject')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig)


def create_flagged_content_analysis(data: pd.DataFrame) -> None:
    """
    Create visualizations for flagged content analysis.
    
    Args:
        data: DataFrame containing moderation results
    """
    if data.empty:
        st.warning("No data available for flagged content analysis")
        return
    
    # Count flagged lessons
    flagged_lessons = data[data['flagged_categories_count'] > 0]
    total_lessons = len(data)
    flagged_count = len(flagged_lessons)
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Lessons", total_lessons)
    with col2:
        st.metric("Flagged Lessons", flagged_count)
    with col3:
        flagged_percentage = (flagged_count / total_lessons * 100) if total_lessons > 0 else 0
        st.metric("Flagged Percentage", f"{flagged_percentage:.1f}%")
    with col4:
        avg_flagged_categories = flagged_lessons['flagged_categories_count'].mean() if flagged_count > 0 else 0
        st.metric("Avg Flagged Categories", f"{avg_flagged_categories:.1f}")
    
    if flagged_count > 0:
        # Distribution of flagged categories count
        fig = px.histogram(
            flagged_lessons,
            x='flagged_categories_count',
            title='Distribution of Number of Flagged Categories per Lesson',
            labels={'flagged_categories_count': 'Number of Flagged Categories', 'count': 'Number of Lessons'},
            color_discrete_sequence=['#ff6b6b']
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top flagged categories
        st.subheader("Most Frequently Flagged Categories")
        
        # Count how many times each category was flagged
        category_flag_counts = {}
        for _, row in flagged_lessons.iterrows():
            flagged_cats = row.get('flagged_categories_list', [])
            for cat in flagged_cats:
                category_flag_counts[cat] = category_flag_counts.get(cat, 0) + 1
        
        if category_flag_counts:
            FLAGGED_COUNT_LABEL = 'Flagged Count'
            flag_df = pd.DataFrame([
                {'Category': cat, FLAGGED_COUNT_LABEL: count} 
                for cat, count in category_flag_counts.items()
            ]).sort_values(FLAGGED_COUNT_LABEL, ascending=False)
            
            fig = px.bar(
                flag_df.head(10),
                x='Category',
                y=FLAGGED_COUNT_LABEL,
                title='Top 10 Most Frequently Flagged Categories',
                color=FLAGGED_COUNT_LABEL,
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# Set page configuration
st.set_page_config(page_title="Visualise Moderation Results", page_icon="🛡️")
st.title("🛡️ Visualise Moderation Results")

# Sidebar buttons for cache management
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Clear Cache"):
        clear_all_caches()
        st.session_state.moderation_experiments_loaded = False
        st.session_state.moderation_experiments_data = pd.DataFrame()
        st.sidebar.success("Cache cleared!")
with col2:
    if st.button("Refresh Data"):
        st.session_state.moderation_experiments_loaded = False
        st.session_state.moderation_experiments_data = pd.DataFrame()
        st.sidebar.success("Data refresh initiated!")

# Initialize session state for caching
if 'moderation_experiments_loaded' not in st.session_state:
    st.session_state.moderation_experiments_loaded = False
if 'moderation_experiments_data' not in st.session_state:
    st.session_state.moderation_experiments_data = pd.DataFrame()

# Load experiments data only when needed
if not st.session_state.moderation_experiments_loaded:
    with st.spinner("Loading moderation experiments..."):
        experiments_data = get_moderation_experiments_data()
        st.session_state.moderation_experiments_data = experiments_data
        st.session_state.moderation_experiments_loaded = True
else:
    experiments_data = st.session_state.moderation_experiments_data

if experiments_data.empty:
    st.warning("No moderation experiments found. Please run some moderation experiments first.")
    st.info("Go to the 'Run Moderations' page to create moderation experiments.")
else:
    st.success(f"Found {len(experiments_data)} moderation experiments")
    
    # Experiment selection
    experiment_options = ["Select"] + experiments_data['experiment_with_date'].unique().tolist()
    
    selected_experiments = st.multiselect(
        "Select Moderation Experiments",
        experiment_options,
        help="Select one or more moderation experiments to analyze"
    )
    
    if selected_experiments and "Select" not in selected_experiments:
        # Get experiment IDs
        selected_experiment_ids = []
        for exp_name in selected_experiments:
            exp_id = experiments_data[experiments_data['experiment_with_date'] == exp_name]['experiment_id'].iloc[0]
            selected_experiment_ids.append(str(exp_id))
        
        # Fetch detailed results only when experiments are selected
        with st.spinner("Loading moderation results..."):
            results_data = get_moderation_results_data(selected_experiment_ids)
        
        if not results_data.empty:
            st.success(f"Loaded {len(results_data)} moderation results")
            
            # Load moderation categories for reference
            try:
                categories_data = load_moderation_categories()
                processed_categories, _, _ = process_moderation_categories(categories_data)
                category_abbreviations = [cat['abbreviation'] for cat in processed_categories]
                
                # Filter data section
                st.subheader("Filter Results")
                
                # Key stage filter
                key_stage_options = results_data['key_stage_slug'].unique().tolist()
                selected_key_stages = st.multiselect(
                    "Filter by Key Stage",
                    key_stage_options,
                    default=key_stage_options
                )
                
                # Subject filter
                subject_options = results_data['subject_slug'].unique().tolist()
                selected_subjects = st.multiselect(
                    "Filter by Subject",
                    subject_options,
                    default=subject_options
                )
                
                # Status filter
                status_options = results_data['result_status'].unique().tolist()
                selected_statuses = st.multiselect(
                    "Filter by Status",
                    status_options,
                    default=status_options
                )
                
                # Apply filters
                filtered_data = results_data[
                    (results_data['key_stage_slug'].isin(selected_key_stages)) &
                    (results_data['subject_slug'].isin(selected_subjects)) &
                    (results_data['result_status'].isin(selected_statuses))
                ]
                
                if not filtered_data.empty:
                    # Overview metrics
                    st.subheader("Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Results", len(filtered_data))
                    with col2:
                        successful_count = len(filtered_data[filtered_data['result_status'] == 'SUCCESS'])
                        st.metric("Successful", successful_count)
                    with col3:
                        failed_count = len(filtered_data[filtered_data['result_status'] == 'FAILURE'])
                        st.metric("Failed", failed_count)
                    with col4:
                        avg_score = filtered_data['average_score'].mean() if 'average_score' in filtered_data.columns else 0
                        st.metric("Avg Score", f"{avg_score:.2f}")
                    
                    # Flagged content analysis
                    st.subheader("Flagged Content Analysis")
                    create_flagged_content_analysis(filtered_data)
                    
                    # Category score distributions
                    st.subheader("Category Score Distributions")
                    
                    # Select categories to display
                    selected_categories = st.multiselect(
                        "Select Categories to Display",
                        category_abbreviations,
                        default=category_abbreviations[:5]  # Show first 5 by default
                    )
                    
                    if selected_categories:
                        # Create columns for category charts
                        cols = st.columns(min(len(selected_categories), 3))
                        
                        for i, category in enumerate(selected_categories):
                            with cols[i % 3]:
                                create_moderation_score_distribution_chart(filtered_data, category)
                    
                    # Heatmap visualization
                    st.subheader("Score Heatmap")
                    create_moderation_heatmap(filtered_data, category_abbreviations)
                    
                    # Detailed results table
                    st.subheader("Detailed Results")
                    
                    # Select columns to display
                    display_columns = [
                        'result_id', 'experiment_name', 'sample_title', 
                        'key_stage_slug', 'subject_slug', 'result_status',
                        'flagged_categories_count', 'average_score', 'justification_summary'
                    ]
                    
                    # Add category score columns
                    for category in category_abbreviations:
                        score_col = f'score_{category}'
                        if score_col in filtered_data.columns:
                            display_columns.append(score_col)
                    
                    # Filter to only existing columns
                    available_columns = [col for col in display_columns if col in filtered_data.columns]
                    
                    # Display table
                    st.dataframe(
                        filtered_data[available_columns],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Individual result details
                    st.subheader("Individual Result Details")
                    
                    result_id_input = st.text_input("Enter Result ID to view details")
                    
                    if result_id_input:
                        result_details = filtered_data[filtered_data['result_id'] == result_id_input]
                        
                        if not result_details.empty:
                            result = result_details.iloc[0]
                            
                            st.write(f"**Result ID:** {result['result_id']}")
                            st.write(f"**Lesson Plan ID:** {result['lesson_plan_id']}")
                            st.write(f"**Sample:** {result['sample_title']}")
                            st.write(f"**Key Stage:** {result['key_stage_slug']}")
                            st.write(f"**Subject:** {result['subject_slug']}")
                            st.write(f"**Status:** {result['result_status']}")
                            
                            # Display category scores
                            st.write("**Category Scores:**")
                            scores_data = []
                            for category in category_abbreviations:
                                score_col = f'score_{category}'
                                if score_col in result and pd.notna(result[score_col]):
                                    scores_data.append({
                                        'Category': category,
                                        'Score': result[score_col],
                                        'Flagged': result[score_col] < 5
                                    })
                            
                            if scores_data:
                                scores_df = pd.DataFrame(scores_data)
                                st.dataframe(scores_df, use_container_width=True)
                            
                            # Display justifications
                            if result.get('justifications_dict'):
                                st.write("**Justifications:**")
                                justifications = result['justifications_dict']
                                for category, justification in justifications.items():
                                    st.write(f"**{category}:** {justification}")
                            
                            # Display lesson plan content
                            if st.checkbox("Show Lesson Plan Content"):
                                if result.get('lesson_plan_json'):
                                    try:
                                        lesson_plan = json.loads(result['lesson_plan_json'])
                                        st.json(lesson_plan)
                                    except (json.JSONDecodeError, TypeError):
                                        st.write("Unable to parse lesson plan JSON")
                                else:
                                    st.write("No lesson plan content available")
                        else:
                            st.warning(f"No result found with ID: {result_id_input}")
                
                else:
                    st.warning("No results match the selected filters")
            
            except Exception as e:
                st.error(f"Error loading moderation categories: {e}")
        
        else:
            st.warning("No detailed results found for the selected experiments")
    
    else:
        st.info("Please select one or more moderation experiments to view results")
