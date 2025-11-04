#!/usr/bin/env python3
"""
Model Combination Analysis Page

This page analyzes results from experiments with different model combinations,
providing heatmaps and radar charts to visualize scores across different runs.

Usage: Access via Streamlit navigation menu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="Model Combination Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .heatmap-container {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_results_file(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load the model combination results CSV file."""
    if file_path is None:
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent / "moderation" / "small_dataset_results_all_combinations.csv",
            Path(__file__).parent.parent.parent / "streamlit" / "moderation" / "small_dataset_results_all_combinations.csv",
            Path.cwd() / "moderation" / "small_dataset_results_all_combinations.csv",
            Path.cwd() / "small_dataset_results_all_combinations.csv",
        ]
        
        for path in possible_paths:
            if path.exists():
                file_path = str(path)
                break
        
        if file_path is None:
            st.error("Could not find small_dataset_results_all_combinations.csv")
            st.info("Please upload the file or place it in the moderation folder")
            return None
    
    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def extract_score_columns(df: pd.DataFrame) -> List[str]:
    """Extract all score column names from the dataframe."""
    score_columns = [col for col in df.columns if col.startswith('score_')]
    return sorted(score_columns)

def clean_run_name(run_name: str) -> str:
    """Clean run name for better display."""
    if pd.isna(run_name):
        return "Unknown"
    run_name = str(run_name)
    
    # Limit input length to prevent DoS
    if len(run_name) > 1000:
        run_name = run_name[:1000]
    
    # Replace underscores with spaces and capitalize
    run_name = run_name.replace('_', ' ').replace('gemini 2 5', 'gemini-2.5').replace('gpt 4 1', 'gpt-4.1').replace('gpt 4o', 'gpt-4o').replace('gpt 5', 'gpt-5')
    
    # Split by 'comprehensive' if present - use string methods instead of regex to avoid backtracking
    run_name_lower = run_name.lower()
    comprehensive_word = 'comprehensive'
    
    if comprehensive_word in run_name_lower:
        # Find the position of 'comprehensive' (case-insensitive)
        idx = run_name_lower.find(comprehensive_word)
        if idx != -1:
            # Find word boundaries: start by skipping leading whitespace before 'comprehensive'
            word_start = idx
            while word_start > 0 and run_name[word_start - 1].isspace():
                word_start -= 1
            
            # Find end: after 'comprehensive' and skip trailing whitespace
            word_end = idx + len(comprehensive_word)
            while word_end < len(run_name) and run_name[word_end].isspace():
                word_end += 1
            
            # Split at the word boundaries
            before = run_name[:word_start].strip()
            after = run_name[word_end:].strip()
            
            # Only format if both parts exist
            if before and after:
                skimmed = before.replace('skimmed ', '').strip()
                comprehensive = after.strip()
                return f"{skimmed} → {comprehensive}"
    
    return run_name.title()

def calculate_average_scores(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """Calculate average scores for each run and category."""
    avg_scores = df.groupby('run_name')[score_columns].mean().reset_index()
    return avg_scores

def create_heatmap(df: pd.DataFrame, score_columns: List[str], title: str = "Average Scores Heatmap", show_percentage: bool = False) -> go.Figure:
    """Create a heatmap showing average scores across runs and categories with sample sizes.
    
    Args:
        df: DataFrame with moderation results
        score_columns: List of score column names
        title: Title for the heatmap
        show_percentage: If True, convert scores (1-5) to percentages (0%, 25%, 50%, 75%, 100%)
    """
    # Calculate averages
    avg_df = calculate_average_scores(df, score_columns)
    
    if len(avg_df) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available for heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate sample sizes (count of non-null values) for each category-run combination
    sample_sizes = {}
    for run_name in df['run_name'].unique():
        run_df = df[df['run_name'] == run_name]
        sample_sizes[run_name] = {}
        for col in score_columns:
            non_null_count = run_df[col].notna().sum()
            sample_sizes[run_name][col] = non_null_count
    
    # Clean run names for display
    avg_df['run_name_display'] = avg_df['run_name'].apply(clean_run_name)
    
    # Prepare data for heatmap - transpose so categories are rows, runs are columns
    heatmap_data = avg_df[score_columns].values.T.copy()
    
    # Build text labels with sample sizes
    text_data = []
    sample_size_data = []
    
    # Convert to percentages if requested (1->0%, 2->25%, 3->50%, 4->75%, 5->100%)
    if show_percentage:
        # Convert scores: percentage = (score - 1) * 25
        heatmap_data_pct = (heatmap_data - 1) * 25
        for i, row in enumerate(heatmap_data_pct):
            category_col = score_columns[i]
            text_row = []
            sample_row = []
            for j, val in enumerate(row):
                run_name = avg_df.iloc[j]['run_name']
                sample_size = sample_sizes[run_name].get(category_col, 0)
                if not np.isnan(val):
                    text_row.append(f"{val:.1f}%<br>(n={sample_size})")
                    sample_row.append(sample_size)
                else:
                    text_row.append("")
                    sample_row.append(0)
            text_data.append(text_row)
            sample_size_data.append(sample_row)
        heatmap_data = heatmap_data_pct
        colorbar_title = "Average Score (%)"
        hovertemplate = '<b>%{y}</b><br>%{x}<br>Score: %{z:.1f}%<br>Sample Size: %{customdata}<extra></extra>'
    else:
        for i, row in enumerate(heatmap_data):
            category_col = score_columns[i]
            text_row = []
            sample_row = []
            for j, val in enumerate(row):
                run_name = avg_df.iloc[j]['run_name']
                sample_size = sample_sizes[run_name].get(category_col, 0)
                if not np.isnan(val):
                    text_row.append(f"{val:.2f}<br>(n={sample_size})")
                    sample_row.append(sample_size)
                else:
                    text_row.append("")
                    sample_row.append(0)
            text_data.append(text_row)
            sample_size_data.append(sample_row)
        colorbar_title = "Average Score"
        hovertemplate = '<b>%{y}</b><br>%{x}<br>Score: %{z:.2f}<br>Sample Size: %{customdata}<extra></extra>'
    
    # Extract category names (remove 'score_' prefix)
    category_names = [col.replace('score_', '').upper() for col in score_columns]
    
    # Create heatmap trace with minimal parameters first
    # Add colorbar and text after to avoid validation issues
    heatmap_trace = go.Heatmap(
        z=heatmap_data,
        x=avg_df['run_name_display'].values,
        y=category_names,
        colorscale='RdYlGn',
        reversescale=True,  # Lower scores (red) are worse, higher (green) are better
        customdata=sample_size_data,
        hoverongaps=False,
        hovertemplate=hovertemplate
    )
    
    fig = go.Figure(data=heatmap_trace)
    
    # Add text and colorbar via update_traces to avoid validation issues
    update_dict = {
        "text": text_data,
        "texttemplate": "%{text}",
        "textfont": {"size": 14}  # Increased font size for better visibility
    }
    
    # Only add colorbar if title is provided
    # Use simple title assignment unless specific Plotly version compatibility is required
    if colorbar_title:
        update_dict["colorbar"] = dict(title=str(colorbar_title))
    
    fig.update_traces(**update_dict)
    
    fig.update_layout(
        title=title,
        xaxis_title="Model Combinations",
        yaxis_title="Categories",
        height=max(1000, len(category_names) * 50 + 300),  # Increased height significantly for better font visibility
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),  # Reverse Y axis so top is first category
        margin=dict(l=120, r=50, t=120, b=250)  # Increased margins to accommodate larger text
    )
    
    return fig

def create_radar_chart(df: pd.DataFrame, score_columns: List[str], run_names: List[str], title: str = "Radar Chart Comparison", show_percentage: bool = False) -> go.Figure:
    """Create a radar chart comparing scores across multiple runs, optionally in percentage format."""
    # Calculate averages for each run
    avg_df = calculate_average_scores(df, score_columns)
    
    # Filter to selected runs
    selected_df = avg_df[avg_df['run_name'].isin(run_names)]
    
    # Extract category names
    category_names = [col.replace('score_', '').upper() for col in score_columns]
    
    # Create radar chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for idx, row in selected_df.iterrows():
        run_name = clean_run_name(row['run_name'])
        values = [row[col] if not pd.isna(row[col]) else 0 for col in score_columns]
        
        # Convert to percentage if requested
        if show_percentage:
            values = [(v - 1) * 25 for v in values]  # Convert 1-5 to 0-100%
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=category_names,
            fill='toself',
            name=run_name,
            line=dict(color=colors[idx % len(colors)]),
            opacity=0.7
        ))
    
    # Set range based on whether percentage is shown
    if show_percentage:
        radial_range = [0, 100]
        radial_title = "Score (%)"
    else:
        radial_range = [0, 5]
        radial_title = "Score"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=radial_range,
                title=radial_title
            )),
        showlegend=True,
        title=title,
        height=600,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig

def parse_human_expected(expected_str: Any) -> Tuple[List[str], List[str]]:
    """
    Parse Human_expected column to extract:
    - Categories that should be detected (positive expectations)
    - Categories that should NOT be detected (negative expectations, 'no X' format)
    
    Returns: (positive_categories, negative_categories)
    """
    if pd.isna(expected_str) or expected_str == "" or expected_str is None:
        return [], []
    
    expected_str = str(expected_str).strip()
    positive_categories = []
    negative_categories = []
    
    # Handle multiple categories (comma-separated)
    parts = [p.strip() for p in expected_str.split(',')]
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check for negative expectations (e.g., "no R", "no T", "no N", "No R")
        if part.lower().startswith('no '):
            category = part[3:].strip()
            if category:
                # Extract the category letter(s) - could be "R", "T", "N" or "T3" etc.
                category = category.upper()
                # Take first letter if multiple (e.g., "T3" -> "T")
                if len(category) > 1 and category[0].isalpha():
                    category = category[0]
                negative_categories.append(category.lower())
        else:
            # Positive expectation: normalize to lowercase and preserve full code
            category = part.strip().upper()
            if category:
                # Preserve full category code (e.g., "N2" -> "n2", "N7" -> "n7", "T3" -> "t3")
                # This ensures "N2" != "N4" and "T3" != "T7" for accurate comparison
                positive_categories.append(category.lower())
    
    return positive_categories, negative_categories

def parse_flagged_categories(flagged_str: Any) -> List[str]:
    """Parse flagged categories from JSON string or list."""
    if pd.isna(flagged_str) or flagged_str == "" or flagged_str is None:
        return []
    
    try:
        if isinstance(flagged_str, str):
            flagged_list = json.loads(flagged_str)
        else:
            flagged_list = flagged_str
        
        if isinstance(flagged_list, list):
            # Normalize to lowercase for comparison
            return [str(cat).lower() for cat in flagged_list]
        else:
            return []
    except (json.JSONDecodeError, TypeError):
        return []

def calculate_detection_metrics(row: pd.Series) -> Dict[str, Any]:
    """
    Calculate detection metrics for a single row by comparing Human_expected with flagged categories.
    
    Returns:
        Dictionary with metrics: TP, FP, FN, TN, precision, recall, F1, accuracy, match_status
    """
    human_expected_str = row.get('Human_expected', '')
    flagged_str = row.get('comprehensive_flagged_categories', '')
    
    # Parse expectations and flagged categories
    positive_expected, negative_expected = parse_human_expected(human_expected_str)
    flagged_categories = parse_flagged_categories(flagged_str)
    
    # Normalize category abbreviations - preserve full code for accurate comparison
    def normalize_category(cat: str) -> str:
        """
        Normalize category for comparison - preserve full code including numbers.
        Examples:
        - "N2" -> "n2" (Human_expected format)
        - "n4" -> "n4" (flagged category abbreviation)
        - "n/current-conflicts" -> "n" (if code format, extract prefix)
        - "n2/current-conflicts" -> "n2" (extract full prefix before slash)
        - "u1" -> "u1" (preserve full code)
        - "u" -> "u" (already just a letter)
        """
        cat = str(cat).lower().strip()
        # If it's a code format (letter/name), extract just the prefix
        if '/' in cat:
            # Extract the full prefix before the slash (e.g., "n2/current-conflicts" -> "n2")
            return cat.split('/')[0]
        # If it's an abbreviation format (letter + number/letter), preserve full code
        # e.g., "n2" -> "n2", "n4" -> "n4", "t3" -> "t3", "u1" -> "u1"
        # Only extract letter if it's a single letter (e.g., "n" -> "n")
        if len(cat) == 1 and cat.isalpha():
            return cat
        # Preserve full code for abbreviations (e.g., "n2", "u1", "t3")
        # This ensures "n2" != "n4" and "u1" != "u2"
        return cat
    
    positive_expected_normalized = [normalize_category(cat) for cat in positive_expected]
    negative_expected_normalized = [normalize_category(cat) for cat in negative_expected]
    flagged_normalized = [normalize_category(cat) for cat in flagged_categories]
    
    # Calculate metrics
    # True Positives: Expected categories that were detected
    true_positives = sum(1 for cat in positive_expected_normalized if cat in flagged_normalized)
    
    # False Negatives: Expected categories that were NOT detected
    false_negatives = len(positive_expected_normalized) - true_positives
    
    # False Positives: Categories detected that were NOT expected (and not in negative expected)
    unexpected_detected = [cat for cat in flagged_normalized if cat not in positive_expected_normalized]
    # Remove those that were explicitly expected NOT to be detected (these are also false positives)
    false_positives = [cat for cat in unexpected_detected if cat not in negative_expected_normalized]
    false_positive_count = len(false_positives)
    
    # True Negatives: Categories expected NOT to be detected that were indeed NOT detected
    true_negatives = sum(1 for cat in negative_expected_normalized if cat not in flagged_normalized)
    
    # Additional false positives: categories that were explicitly expected NOT to be detected but were detected
    explicit_false_positives = sum(1 for cat in negative_expected_normalized if cat in flagged_normalized)
    false_positive_count += explicit_false_positives
    
    # Calculate precision, recall, F1
    total_detected = len(flagged_normalized)
    total_expected = len(positive_expected_normalized)
    
    precision = true_positives / total_detected if total_detected > 0 else 0.0
    recall = true_positives / total_expected if total_expected > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    total_cases = true_positives + true_negatives + false_positive_count + false_negatives
    accuracy = (true_positives + true_negatives) / total_cases if total_cases > 0 else 0.0
    
    # Determine match status
    if total_expected == 0 and false_positive_count == 0 and total_detected == 0:
        match_status = "Perfect"  # No expectations and nothing detected
    elif true_positives == total_expected and false_positive_count == 0:
        match_status = "Perfect"  # All expected detected, no unexpected
    elif true_positives == total_expected:
        match_status = "Correct + Extra"  # All expected detected, but some extra
    elif false_positive_count == 0:
        match_status = "Partial"  # Some expected detected, no unexpected
    else:
        match_status = "Incorrect"  # Missing expected or has unexpected
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positive_count,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'match_status': match_status,
        'total_expected': total_expected,
        'total_detected': total_detected,
        'expected_categories': positive_expected,
        'flagged_categories': flagged_categories,
        'unexpected_detected': false_positives + [cat for cat in negative_expected_normalized if cat in flagged_normalized]
    }

def analyze_lesson_plan_performance(df: pd.DataFrame, lesson_id_column: str = 'id', comparison_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Analyze performance metrics for each lesson plan across all model combinations.
    
    Args:
        df: Original dataframe with results
        lesson_id_column: Column name to identify unique lesson plans
        comparison_df: Optional dataframe with Human_expected comparison metrics
    
    Returns:
        DataFrame with aggregated metrics per lesson plan
    """
    if lesson_id_column not in df.columns:
        # Try alternative column names
        possible_id_cols = [c for c in df.columns if 'id' in c.lower()]
        if possible_id_cols:
            lesson_id_column = possible_id_cols[0]
        else:
            st.warning(f"No ID column found to group lesson plans. Using row index.")
            # Create a temporary ID based on first few characters of lesson_plan if available
            if 'lesson_plan' in df.columns:
                df['_temp_id'] = df['lesson_plan'].astype(str).str[:50]
                lesson_id_column = '_temp_id'
            else:
                return pd.DataFrame()
    
    lesson_plan_metrics = []
    
    # Merge comparison metrics if available
    if comparison_df is not None and 'row_index' in comparison_df.columns:
        # Add row_index to df based on DataFrame index
        df_with_index = df.copy()
        df_with_index['_temp_row_index'] = df_with_index.index
        
        # Merge comparison metrics
        df_with_metrics = df_with_index.merge(
            comparison_df[['row_index', 'precision', 'recall', 'f1_score', 'accuracy', 'match_status']],
            left_on='_temp_row_index',
            right_on='row_index',
            how='left'
        )
        df_with_metrics = df_with_metrics.drop(columns=['_temp_row_index'])
    else:
        df_with_metrics = df.copy()
    
    for lesson_id in df[lesson_id_column].unique():
        lesson_df = df_with_metrics[df_with_metrics[lesson_id_column] == lesson_id].copy()
        
        if len(lesson_df) == 0:
            continue
        
        # Get first row for lesson plan metadata
        first_row = lesson_df.iloc[0]
        
        # Count runs for this lesson plan
        num_runs = len(lesson_df)
        
        # Calculate average detection metrics if available
        avg_precision = lesson_df['precision'].mean() if 'precision' in lesson_df.columns and lesson_df['precision'].notna().any() else None
        avg_recall = lesson_df['recall'].mean() if 'recall' in lesson_df.columns and lesson_df['recall'].notna().any() else None
        avg_f1 = lesson_df['f1_score'].mean() if 'f1_score' in lesson_df.columns and lesson_df['f1_score'].notna().any() else None
        avg_accuracy = lesson_df['accuracy'].mean() if 'accuracy' in lesson_df.columns and lesson_df['accuracy'].notna().any() else None
        
        # Count successful runs
        successful_runs = sum(lesson_df['final_status'] == 'SUCCESS') if 'final_status' in lesson_df.columns else 0
        
        # Get best and worst runs (by F1 score if available, otherwise by success status)
        if 'f1_score' in lesson_df.columns and lesson_df['f1_score'].notna().any():
            best_run_idx = lesson_df['f1_score'].idxmax()
            best_run = lesson_df.loc[best_run_idx, 'run_name'] if 'run_name' in lesson_df.columns else 'Unknown'
            best_f1 = lesson_df.loc[best_run_idx, 'f1_score']
            
            worst_run_idx = lesson_df['f1_score'].idxmin()
            worst_run = lesson_df.loc[worst_run_idx, 'run_name'] if 'run_name' in lesson_df.columns else 'Unknown'
            worst_f1 = lesson_df.loc[worst_run_idx, 'f1_score']
        else:
            # Fallback: use success status
            successful_runs_df = lesson_df[lesson_df['final_status'] == 'SUCCESS'] if 'final_status' in lesson_df.columns else lesson_df
            if len(successful_runs_df) > 0:
                best_run = successful_runs_df.iloc[0]['run_name'] if 'run_name' in successful_runs_df.columns else 'Unknown'
                best_f1 = None
            else:
                best_run = lesson_df.iloc[0]['run_name'] if 'run_name' in lesson_df.columns else 'Unknown'
                best_f1 = None
            
            failed_runs_df = lesson_df[lesson_df['final_status'] == 'FAILED'] if 'final_status' in lesson_df.columns else pd.DataFrame()
            if len(failed_runs_df) > 0:
                worst_run = failed_runs_df.iloc[0]['run_name'] if 'run_name' in failed_runs_df.columns else 'Unknown'
                worst_f1 = None
            else:
                worst_run = lesson_df.iloc[0]['run_name'] if 'run_name' in lesson_df.columns else 'Unknown'
                worst_f1 = None
        
        # Calculate average times and tokens
        avg_skimmed_time = lesson_df['skimmed_time'].mean() if 'skimmed_time' in lesson_df.columns and lesson_df['skimmed_time'].notna().any() else None
        avg_comprehensive_time = lesson_df['comprehensive_time'].mean() if 'comprehensive_time' in lesson_df.columns and lesson_df['comprehensive_time'].notna().any() else None
        total_input_tokens = None
        if 'skimmed_input_tokens' in lesson_df.columns and 'comprehensive_input_tokens' in lesson_df.columns:
            total_input_tokens = (lesson_df['skimmed_input_tokens'].sum() + lesson_df['comprehensive_input_tokens'].sum())
        
        # Get categories detected (consensus across runs)
        all_detected = []
        for idx, row in lesson_df.iterrows():
            detected = parse_flagged_categories(row.get('comprehensive_flagged_categories', ''))
            all_detected.extend(detected)
        
        # Count frequency of each detected category
        category_counts = Counter(all_detected)
        most_common_categories = [cat for cat, count in category_counts.most_common(5)]
        
        # Check if human expected exists
        human_expected = first_row.get('Human_expected', '')
        has_human_expected = pd.notna(human_expected) and str(human_expected).strip() != ''
        
        # Calculate average scores across all score columns
        score_columns = [col for col in lesson_df.columns if col.startswith('score_')]
        avg_scores = {}
        for col in score_columns:
            valid_scores = lesson_df[col].dropna()
            if len(valid_scores) > 0:
                avg_scores[col] = valid_scores.mean()
        
        lesson_plan_metrics.append({
            'lesson_id': lesson_id,
            'topic': first_row.get('Topic', 'Unknown'),
            'num_runs': num_runs,
            'successful_runs': successful_runs,
            'failed_runs': num_runs - successful_runs,
            'success_rate': (successful_runs / num_runs * 100) if num_runs > 0 else 0,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'avg_accuracy': avg_accuracy,
            'best_run': best_run,
            'best_f1': best_f1,
            'worst_run': worst_run,
            'worst_f1': worst_f1,
            'avg_skimmed_time': avg_skimmed_time,
            'avg_comprehensive_time': avg_comprehensive_time,
            'total_input_tokens': total_input_tokens,
            'most_common_categories': ', '.join(most_common_categories) if most_common_categories else 'None',
            'num_categories_detected': len(set(all_detected)),
            'human_expected': human_expected if has_human_expected else None,
            'has_human_expected': has_human_expected,
            'avg_scores_dict': avg_scores  # Store as dict for later use
        })
    
    return pd.DataFrame(lesson_plan_metrics)

def analyze_human_expected_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze Human_expected vs moderation outputs for all rows."""
    results = []
    
    for idx, row in df.iterrows():
        metrics = calculate_detection_metrics(row)
        metrics['row_index'] = idx
        metrics['run_name'] = row.get('run_name', '')
        metrics['human_expected'] = row.get('Human_expected', '')
        results.append(metrics)
    
    return pd.DataFrame(results)

def create_category_summary(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """Create summary statistics for each category across all runs."""
    summary_data = []
    
    for col in score_columns:
        category = col.replace('score_', '').upper()
        valid_scores = df[col].dropna()
        
        if len(valid_scores) > 0:
            summary_data.append({
                'Category': category,
                'Mean': valid_scores.mean(),
                'Median': valid_scores.median(),
                'Std': valid_scores.std(),
                'Min': valid_scores.min(),
                'Max': valid_scores.max(),
                'Count': len(valid_scores),
                'Non-Null %': (len(valid_scores) / len(df)) * 100
            })
    
    return pd.DataFrame(summary_data)

def create_run_summary(df: pd.DataFrame, comparison_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Create summary statistics for each run, including Human_expected metrics if available."""
    summary_data = []
    
    for run_name in df['run_name'].unique():
        run_df = df[df['run_name'] == run_name]
        
        # Get score columns
        score_columns = [col for col in run_df.columns if col.startswith('score_')]
        
        # Calculate metrics
        total_lessons = len(run_df)
        successful = sum(run_df['final_status'] == 'SUCCESS') if 'final_status' in run_df.columns else total_lessons
        failed = sum(run_df['final_status'] == 'FAILED') if 'final_status' in run_df.columns else 0
        
        # Calculate average scores
        avg_scores = {}
        for col in score_columns:
            valid_scores = run_df[col].dropna()
            if len(valid_scores) > 0:
                avg_scores[col] = valid_scores.mean()
        
        avg_score = np.mean(list(avg_scores.values())) if avg_scores else None
        avg_score_pct = ((avg_score - 1) * 25) if avg_score is not None else None  # Convert to percentage
        
        # Timing metrics
        avg_skimmed_time = run_df['skimmed_time'].dropna().mean() if 'skimmed_time' in run_df.columns else None
        avg_comprehensive_time = run_df['comprehensive_time'].dropna().mean() if 'comprehensive_time' in run_df.columns else None
        total_time = (avg_skimmed_time + avg_comprehensive_time) if (avg_skimmed_time is not None and avg_comprehensive_time is not None) else None
        
        # Token metrics
        total_input_tokens = (run_df['skimmed_input_tokens'].sum() + run_df['comprehensive_input_tokens'].sum()) if 'skimmed_input_tokens' in run_df.columns else None
        total_output_tokens = (run_df['skimmed_output_tokens'].sum() + run_df['comprehensive_output_tokens'].sum()) if 'skimmed_output_tokens' in run_df.columns else None
        total_tokens = (total_input_tokens + total_output_tokens) if (total_input_tokens is not None and total_output_tokens is not None) else None
        
        # Human_expected metrics if comparison_df is available
        human_expected_metrics = {}
        if comparison_df is not None:
            run_comparison = comparison_df[comparison_df['run_name'] == run_name]
            if len(run_comparison) > 0:
                human_expected_metrics = {
                    'Avg Precision': run_comparison['precision'].mean(),
                    'Avg Recall': run_comparison['recall'].mean(),
                    'Avg F1 Score': run_comparison['f1_score'].mean(),
                    'Avg Accuracy': run_comparison['accuracy'].mean(),
                    'Perfect Matches': (run_comparison['match_status'] == 'Perfect').sum(),
                    'Perfect Match Rate %': ((run_comparison['match_status'] == 'Perfect').sum() / len(run_comparison) * 100) if len(run_comparison) > 0 else 0
                }
        
        summary_data.append({
            'Run Name': clean_run_name(run_name),
            'Total Lessons': total_lessons,
            'Successful': successful,
            'Failed': failed,
            'Success Rate %': (successful / total_lessons * 100) if total_lessons > 0 else 0,
            'Avg Score': avg_score,
            'Avg Score (%)': avg_score_pct,
            'Avg Skimmed Time (s)': avg_skimmed_time,
            'Avg Comprehensive Time (s)': avg_comprehensive_time,
            'Total Avg Time (s)': total_time,
            'Total Input Tokens': total_input_tokens,
            'Total Output Tokens': total_output_tokens,
            'Total Tokens': total_tokens,
            **human_expected_metrics
        })
    
    return pd.DataFrame(summary_data)

def main():
    st.title("📈 Model Combination Analysis")
    st.markdown("Analyze and compare results from different model combinations using heatmaps and radar charts.")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Source")
        uploaded_file = st.file_uploader("Upload results CSV", type=['csv'], key="model_combination_file")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows from uploaded file")
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")
                df = None
        else:
            df = load_results_file()
    
    if df is None or len(df) == 0:
        st.warning("No data loaded. Please upload a file or ensure small_dataset_results_all_combinations.csv exists.")
        return
    
    # Extract score columns
    score_columns = extract_score_columns(df)
    
    if len(score_columns) == 0:
        st.error("No score columns found in the dataset!")
        return
    
    st.success(f"Found {len(score_columns)} score categories and {len(df['run_name'].unique())} unique runs")
    
    # Check if Human_expected column exists
    has_human_expected = 'Human_expected' in df.columns
    
    # Main content tabs
    if has_human_expected:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Heatmap", "📈 Radar Charts", "📋 Run Summary", "📊 Category Summary", "✅ Expected vs Detected", "📚 Lesson Plan Performance", "🎯 Model Accuracy Comparison"])
    else:
        tab1, tab2, tab3, tab4, tab6, tab7 = st.tabs(["📊 Heatmap", "📈 Radar Charts", "📋 Run Summary", "📊 Category Summary", "📚 Lesson Plan Performance", "🎯 Model Accuracy Comparison"])
        tab5 = None
    
    with tab1:
        st.header("📊 Average Scores Heatmap")
        st.markdown("This heatmap shows the average scores for each category across different model combinations. View in both raw scores and percentage format.")
        
        # Options for heatmap
        col1, col2 = st.columns(2)
        with col1:
            show_values = st.checkbox("Show values on heatmap", value=True, key="heatmap_show_values")
        with col2:
            show_percentage = st.checkbox("Show percentage version", value=False, key="heatmap_show_percentage")
        
        # Filter runs if desired
        all_runs = sorted(df['run_name'].unique())
        selected_runs = st.multiselect(
            "Select runs to display (leave empty for all)",
            options=all_runs,
            default=[],
            format_func=clean_run_name
        )
        
        heatmap_df = df[df['run_name'].isin(selected_runs)] if selected_runs else df
        
        if len(heatmap_df) == 0:
            st.warning("No data selected for heatmap")
        else:
            # Quick insights
            st.subheader("🎯 Quick Insights")
            avg_df = calculate_average_scores(heatmap_df, score_columns)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Best overall average
                avg_df['avg_all'] = avg_df[score_columns].mean(axis=1)
                best_run = avg_df.loc[avg_df['avg_all'].idxmax()]
                best_pct = ((best_run['avg_all'] - 1) * 25)
                st.metric("Best Overall Score", f"{best_pct:.1f}%", clean_run_name(best_run['run_name']))
            
            with col2:
                # Best category average
                category_means = {}
                for col in score_columns:
                    category_means[col] = avg_df[col].mean()
                best_category_col = max(category_means, key=category_means.get) if category_means else None
                if best_category_col:
                    cat_name = best_category_col.replace('score_', '').upper()
                    best_cat_pct = ((category_means[best_category_col] - 1) * 25)
                    st.metric("Best Category", cat_name, f"{best_cat_pct:.1f}%")
            
            with col3:
                # Most variable category
                category_std = {}
                for col in score_columns:
                    category_std[col] = avg_df[col].std()
                most_variable = max(category_std, key=category_std.get) if category_std else None
                if most_variable:
                    cat_name = most_variable.replace('score_', '').upper()
                    st.metric("Most Variable", cat_name, f"Std: {category_std[most_variable]:.2f}")
            
            with col4:
                # Overall average
                overall_avg = avg_df[score_columns].values.mean()
                overall_avg_pct = ((overall_avg - 1) * 25)
                st.metric("Overall Average", f"{overall_avg_pct:.1f}%", f"{len(avg_df)} runs")
            
            # Heatmaps
            # Create and display heatmap (raw scores)
            fig = create_heatmap(heatmap_df, score_columns, "Average Scores Heatmap by Model Combination", show_percentage=False)
            if not show_values:
                fig.update_traces(texttemplate='', textfont=None)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentage version if requested
            if show_percentage:
                st.subheader("Percentage Version")
                st.markdown("Percentage conversion: 1 = 0%, 2 = 25%, 3 = 50%, 4 = 75%, 5 = 100%")
                fig_pct = create_heatmap(heatmap_df, score_columns, "Average Scores Heatmap (Percentage) by Model Combination", show_percentage=True)
                if not show_values:
                    fig_pct.update_traces(texttemplate='', textfont=None)
                st.plotly_chart(fig_pct, use_container_width=True)
            
            # Enhanced data table with sample sizes
            st.subheader("📊 Detailed Data Table with Sample Sizes")
            st.markdown("Average scores and sample sizes for each category and model combination.")
            
            # Calculate averages and sample sizes
            avg_df = calculate_average_scores(heatmap_df, score_columns)
            enhanced_table_data = []
            
            for idx, row in avg_df.iterrows():
                run_name = row['run_name']
                run_df = heatmap_df[heatmap_df['run_name'] == run_name]
                
                for col in score_columns:
                    avg_score = row[col]
                    sample_size = run_df[col].notna().sum()
                    category = col.replace('score_', '').upper()
                    
                    if not pd.isna(avg_score):
                        # Calculate percentage
                        pct_score = (avg_score - 1) * 25
                        
                        enhanced_table_data.append({
                            'Category': category,
                            'Run': clean_run_name(run_name),
                            'Average Score': f"{avg_score:.2f}",
                            'Percentage (%)': f"{pct_score:.1f}%",
                            'Sample Size (n)': sample_size,
                            'Min Score': run_df[col].min() if sample_size > 0 else np.nan,
                            'Max Score': run_df[col].max() if sample_size > 0 else np.nan,
                            'Std Dev': run_df[col].std() if sample_size > 1 else np.nan
                        })
            
            if enhanced_table_data:
                enhanced_df = pd.DataFrame(enhanced_table_data)
                # Format min, max, std dev - handle numeric values properly
                for col in ['Min Score', 'Max Score', 'Std Dev']:
                    if col in enhanced_df.columns:
                        enhanced_df[col] = enhanced_df[col].apply(
                            lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A"
                        )
                
                st.dataframe(enhanced_df, use_container_width=True)
                
                # Download button
                csv = enhanced_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enhanced Data Table CSV",
                    data=csv,
                    file_name="average_scores_with_sample_sizes.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.header("📈 Radar Chart Comparison")
        st.markdown("Compare scores across different model combinations using radar charts. View in both raw scores and percentage format.")
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            show_percentage = st.checkbox("Show percentage version", value=False, key="radar_show_percentage")
        with col2:
            show_table = st.checkbox("Show detailed comparison table", value=True, key="radar_show_table")
        
        # Select runs for comparison
        all_runs = sorted(df['run_name'].unique())
        selected_runs_radar = st.multiselect(
            "Select runs to compare (select 2-5 for best visualization)",
            options=all_runs,
            default=all_runs[:3] if len(all_runs) >= 3 else all_runs,
            format_func=clean_run_name,
            key="radar_runs"
        )
        
        if len(selected_runs_radar) == 0:
            st.warning("Please select at least one run to display")
        elif len(selected_runs_radar) > 10:
            st.warning("⚠️ Too many runs selected (max 10). Displaying first 10.")
            selected_runs_radar = selected_runs_radar[:10]
        else:
            # Quick insights
            st.subheader("🎯 Quick Insights")
            avg_df = calculate_average_scores(df, score_columns)
            comparison_df = avg_df[avg_df['run_name'].isin(selected_runs_radar)].copy()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Find best overall average score
                comparison_df['avg_all'] = comparison_df[score_columns].mean(axis=1)
                best_avg = comparison_df.loc[comparison_df['avg_all'].idxmax()]
                best_avg_pct = ((best_avg['avg_all'] - 1) * 25)
                st.metric("Best Overall Score", f"{best_avg_pct:.1f}%", clean_run_name(best_avg['run_name']))
            
            with col2:
                # Find category with highest variation
                category_std = {}
                for col in score_columns:
                    category_std[col] = comparison_df[col].std()
                most_variable = max(category_std, key=category_std.get) if category_std else None
                if most_variable:
                    cat_name = most_variable.replace('score_', '').upper()
                    st.metric("Most Variable Category", cat_name, f"Std: {category_std[most_variable]:.2f}")
            
            with col3:
                # Find most consistent category
                least_variable = min(category_std, key=category_std.get) if category_std else None
                if least_variable:
                    cat_name = least_variable.replace('score_', '').upper()
                    st.metric("Most Consistent Category", cat_name, f"Std: {category_std[least_variable]:.2f}")
            
            with col4:
                # Average across all selected runs
                overall_avg = comparison_df[score_columns].values.mean()
                overall_avg_pct = ((overall_avg - 1) * 25)
                st.metric("Average Across Runs", f"{overall_avg_pct:.1f}%", f"{len(selected_runs_radar)} runs")
            
            # Create radar chart (raw scores)
            st.subheader("📊 Radar Chart: Raw Scores (1-5 scale)")
            fig = create_radar_chart(df, score_columns, selected_runs_radar, "Radar Chart: Score Comparison Across Runs")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentage version if requested
            if show_percentage:
                st.subheader("📊 Radar Chart: Percentage Version (0-100% scale)")
                st.markdown("Percentage conversion: 1 = 0%, 2 = 25%, 3 = 50%, 4 = 75%, 5 = 100%")
                
                # Create percentage version
                fig_pct = create_radar_chart(df, score_columns, selected_runs_radar, "Radar Chart: Score Comparison (Percentage)", show_percentage=True)
                st.plotly_chart(fig_pct, use_container_width=True)
            
            # Show detailed comparison table
            if show_table:
                st.subheader("📋 Detailed Score Comparison")
                comparison_df['run_name_display'] = comparison_df['run_name'].apply(clean_run_name)
                
                # Create enhanced table with percentages
                display_data = []
                for idx, row in comparison_df.iterrows():
                    row_data = {'Run Name': row['run_name_display']}
                    for col in score_columns:
                        category = col.replace('score_', '').upper()
                        score = row[col]
                        score_pct = ((score - 1) * 25) if pd.notna(score) else None
                        row_data[f'{category} (Score)'] = f"{score:.2f}" if pd.notna(score) else "N/A"
                        row_data[f'{category} (%)'] = f"{score_pct:.1f}%" if score_pct is not None else "N/A"
                    display_data.append(row_data)
                
                if display_data:
                    display_df = pd.DataFrame(display_data)
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download button
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Radar Comparison CSV",
                        data=csv,
                        file_name="radar_comparison.csv",
                        mime="text/csv"
                    )
                
                # Also show original format
                st.subheader("📊 Original Format (Transposed)")
                display_cols = ['run_name_display'] + score_columns
                transposed_df = comparison_df[display_cols].set_index('run_name_display').T
                transposed_df.columns.name = 'Category / Run'
                st.dataframe(transposed_df, use_container_width=True)
    
    with tab3:
        st.header("📋 Run Summary Statistics")
        st.markdown("Comprehensive overview statistics for each model combination run, including Human_expected metrics if available.")
        
        # Get comparison_df if Human_expected is available
        comparison_df = None
        if has_human_expected:
            with st.spinner("Analyzing Human_expected metrics..."):
                comparison_df = analyze_human_expected_comparison(df)
        
        summary_df = create_run_summary(df, comparison_df)
        
        if len(summary_df) > 0:
            # Quick insights
            st.subheader("🎯 Quick Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_success = summary_df.loc[summary_df['Success Rate %'].idxmax()]
                st.metric("Best Success Rate", f"{best_success['Success Rate %']:.1f}%", best_success['Run Name'])
            
            with col2:
                best_score = summary_df.loc[summary_df['Avg Score (%)'].idxmax()] if 'Avg Score (%)' in summary_df.columns else None
                if best_score is not None and pd.notna(best_score['Avg Score (%)']):
                    st.metric("Best Avg Score", f"{best_score['Avg Score (%)']:.1f}%", best_score['Run Name'])
            
            with col3:
                fastest = summary_df.loc[summary_df['Total Avg Time (s)'].idxmin()] if 'Total Avg Time (s)' in summary_df.columns else None
                if fastest is not None and pd.notna(fastest['Total Avg Time (s)']):
                    st.metric("Fastest Run", f"{fastest['Total Avg Time (s)']:.2f}s", fastest['Run Name'])
            
            with col4:
                if 'Avg F1 Score' in summary_df.columns:
                    best_f1 = summary_df.loc[summary_df['Avg F1 Score'].idxmax()]
                    st.metric("Best F1 Score", f"{best_f1['Avg F1 Score']*100:.1f}%", best_f1['Run Name'])
            
            # Main summary table
            st.subheader("📊 Detailed Summary Table")
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Run Summary CSV",
                data=csv,
                file_name="run_summary.csv",
                mime="text/csv"
            )
            
            # Enhanced Visualizations
            st.subheader("📈 Visualizations")
            
            # Row 1: Success Rate and Score Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                fig_success = px.bar(
                    summary_df,
                    x='Run Name',
                    y='Success Rate %',
                    title="Success Rate by Run",
                    color='Success Rate %',
                    color_continuous_scale='RdYlGn',
                    text='Success Rate %'
                )
                fig_success.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_success.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_success, use_container_width=True)
            
            with col2:
                if 'Avg Score (%)' in summary_df.columns:
                    fig_score = px.bar(
                        summary_df,
                        x='Run Name',
                        y='Avg Score (%)',
                        title="Average Score (%) by Run",
                        color='Avg Score (%)',
                        color_continuous_scale='RdYlGn',
                        text='Avg Score (%)'
                    )
                    fig_score.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_score.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_score, use_container_width=True)
            
            # Row 2: Timing Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                fig_time = px.bar(
                    summary_df,
                    x='Run Name',
                    y=['Avg Skimmed Time (s)', 'Avg Comprehensive Time (s)'],
                    title="Average Processing Time by Run",
                    barmode='group',
                    color_discrete_map={'Avg Skimmed Time (s)': '#1f77b4', 'Avg Comprehensive Time (s)': '#ff7f0e'}
                )
                fig_time.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                if 'Total Avg Time (s)' in summary_df.columns:
                    fig_total_time = px.bar(
                        summary_df,
                        x='Run Name',
                        y='Total Avg Time (s)',
                        title="Total Average Time by Run",
                        color='Total Avg Time (s)',
                        color_continuous_scale='RdYlGn_r',  # Reversed: lower is better
                        text='Total Avg Time (s)'
                    )
                    fig_total_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
                    fig_total_time.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_total_time, use_container_width=True)
            
            # Row 3: Human_expected Metrics (if available)
            if 'Avg F1 Score' in summary_df.columns:
                st.subheader("🎯 Human Expected Detection Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_metrics = go.Figure()
                    fig_metrics.add_trace(go.Bar(
                        name='Precision',
                        x=summary_df['Run Name'],
                        y=summary_df['Avg Precision'] * 100,
                        marker_color='#1f77b4',
                        text=summary_df['Avg Precision'].apply(lambda x: f"{x*100:.1f}%"),
                        textposition='outside'
                    ))
                    fig_metrics.add_trace(go.Bar(
                        name='Recall',
                        x=summary_df['Run Name'],
                        y=summary_df['Avg Recall'] * 100,
                        marker_color='#ff7f0e',
                        text=summary_df['Avg Recall'].apply(lambda x: f"{x*100:.1f}%"),
                        textposition='outside'
                    ))
                    fig_metrics.add_trace(go.Bar(
                        name='F1 Score',
                        x=summary_df['Run Name'],
                        y=summary_df['Avg F1 Score'] * 100,
                        marker_color='#2ca02c',
                        text=summary_df['Avg F1 Score'].apply(lambda x: f"{x*100:.1f}%"),
                        textposition='outside'
                    ))
                    fig_metrics.update_layout(
                        title='Precision, Recall, and F1 Score by Run',
                        xaxis_title='Run Name',
                        yaxis_title='Score (%)',
                        barmode='group',
                        height=500,
                        xaxis=dict(tickangle=-45),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_metrics, use_container_width=True)
                
                with col2:
                    fig_perfect = px.bar(
                        summary_df,
                        x='Run Name',
                        y='Perfect Match Rate %',
                        title="Perfect Match Rate by Run",
                        color='Perfect Match Rate %',
                        color_continuous_scale='RdYlGn',
                        text='Perfect Match Rate %'
                    )
                    fig_perfect.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_perfect.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig_perfect, use_container_width=True)
            
            # Row 4: Token Usage (if available)
            if 'Total Tokens' in summary_df.columns:
                st.subheader("💻 Token Usage")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_tokens = px.bar(
                        summary_df,
                        x='Run Name',
                        y=['Total Input Tokens', 'Total Output Tokens'],
                        title="Token Usage by Run",
                        barmode='group',
                        color_discrete_map={'Total Input Tokens': '#9467bd', 'Total Output Tokens': '#8c564b'}
                    )
                    fig_tokens.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_tokens, use_container_width=True)
                
                with col2:
                    fig_total_tokens = px.bar(
                        summary_df,
                        x='Run Name',
                        y='Total Tokens',
                        title="Total Token Usage by Run",
                        color='Total Tokens',
                        color_continuous_scale='Viridis',
                        text='Total Tokens'
                    )
                    fig_total_tokens.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_total_tokens.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_total_tokens, use_container_width=True)
    
    with tab4:
        st.header("📊 Category Summary Statistics")
        st.markdown("Comprehensive summary statistics for each moderation category across all runs, including percentage scores.")
        
        category_summary = create_category_summary(df, score_columns)
        
        if len(category_summary) > 0:
            # Add percentage columns
            category_summary['Mean (%)'] = ((category_summary['Mean'] - 1) * 25).round(1)
            category_summary['Median (%)'] = ((category_summary['Median'] - 1) * 25).round(1)
            category_summary['Min (%)'] = ((category_summary['Min'] - 1) * 25).round(1)
            category_summary['Max (%)'] = ((category_summary['Max'] - 1) * 25).round(1)
            
            # Quick insights
            st.subheader("🎯 Quick Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_category = category_summary.loc[category_summary['Mean (%)'].idxmax()]
                st.metric("Best Category", best_category['Category'], f"{best_category['Mean (%)']:.1f}%")
            
            with col2:
                worst_category = category_summary.loc[category_summary['Mean (%)'].idxmin()]
                st.metric("Worst Category", worst_category['Category'], f"{worst_category['Mean (%)']:.1f}%")
            
            with col3:
                most_samples = category_summary.loc[category_summary['Count'].idxmax()]
                st.metric("Most Samples", most_samples['Category'], f"{int(most_samples['Count'])}")
            
            with col4:
                avg_all = category_summary['Mean (%)'].mean()
                st.metric("Average Across All", f"{avg_all:.1f}%", f"{len(category_summary)} categories")
            
            # Main summary table
            st.subheader("📊 Detailed Category Summary")
            
            # Reorder columns for better display
            display_cols = ['Category', 'Mean', 'Mean (%)', 'Median', 'Median (%)', 
                           'Min', 'Min (%)', 'Max', 'Max (%)', 'Std', 'Count', 'Non-Null %']
            available_cols = [col for col in display_cols if col in category_summary.columns]
            st.dataframe(category_summary[available_cols], use_container_width=True)
            
            # Download button
            csv = category_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Category Summary CSV",
                data=csv,
                file_name="category_summary.csv",
                mime="text/csv"
            )
            
            # Enhanced Visualizations
            st.subheader("📈 Visualizations")
            
            # Row 1: Score Comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig_category_mean = px.bar(
                    category_summary,
                    x='Category',
                    y='Mean (%)',
                    title="Average Score (%) by Category",
                    color='Mean (%)',
                    color_continuous_scale='RdYlGn',
                    text='Mean (%)'
                )
                fig_category_mean.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_category_mean.update_layout(height=500, xaxis_tickangle=-45, yaxis_title='Score (%)')
                st.plotly_chart(fig_category_mean, use_container_width=True)
            
            with col2:
                fig_category_range = go.Figure()
                fig_category_range.add_trace(go.Bar(
                    name='Mean',
                    x=category_summary['Category'],
                    y=category_summary['Mean (%)'],
                    marker_color='#2ca02c',
                    text=category_summary['Mean (%)'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside'
                ))
                fig_category_range.add_trace(go.Bar(
                    name='Min',
                    x=category_summary['Category'],
                    y=category_summary['Min (%)'],
                    marker_color='#d62728',
                    text=category_summary['Min (%)'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside'
                ))
                fig_category_range.add_trace(go.Bar(
                    name='Max',
                    x=category_summary['Category'],
                    y=category_summary['Max (%)'],
                    marker_color='#1f77b4',
                    text=category_summary['Max (%)'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside'
                ))
                fig_category_range.update_layout(
                    title='Score Range (Min, Mean, Max) by Category',
                    xaxis_title='Category',
                    yaxis_title='Score (%)',
                    barmode='group',
                    height=500,
                    xaxis=dict(tickangle=-45),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_category_range, use_container_width=True)
            
            # Row 2: Distribution and Sample Size
            col1, col2 = st.columns(2)
            
            with col1:
                fig_std = px.bar(
                    category_summary,
                    x='Category',
                    y='Std',
                    title="Standard Deviation by Category",
                    color='Std',
                    color_continuous_scale='RdYlGn_r',  # Reversed: lower std is better (more consistent)
                    text='Std'
                )
                fig_std.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_std.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_std, use_container_width=True)
            
            with col2:
                fig_samples = px.bar(
                    category_summary,
                    x='Category',
                    y='Count',
                    title="Sample Size by Category",
                    color='Count',
                    color_continuous_scale='Blues',
                    text='Count'
                )
                fig_samples.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig_samples.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_samples, use_container_width=True)
            
            # Row 3: Box plot showing distribution
            st.subheader("📊 Score Distribution by Category")
            box_data = []
            for _, row in category_summary.iterrows():
                category = row['Category']
                category_col = f"score_{category.lower()}"
                if category_col in df.columns:
                    scores = df[category_col].dropna()
                    if len(scores) > 0:
                        for score in scores:
                            box_data.append({
                                'Category': category,
                                'Score': score,
                                'Score (%)': (score - 1) * 25
                            })
            
            if box_data:
                box_df = pd.DataFrame(box_data)
                fig_box = px.box(
                    box_df,
                    x='Category',
                    y='Score (%)',
                    title="Score Distribution by Category",
                    color='Category',
                    points="outliers"
                )
                fig_box.update_layout(height=600, xaxis_tickangle=-45, yaxis_title='Score (%)', showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
    
    if tab5 is not None:
        with tab5:
            st.header("✅ Human Expected vs Detected Categories")
            st.markdown("Compare moderation outputs against human expectations to evaluate detection accuracy.")
            
            # Analyze the comparison
            with st.spinner("Analyzing Human_expected vs detected categories..."):
                comparison_df = analyze_human_expected_comparison(df)
            
            st.success(f"Analyzed {len(comparison_df)} rows from {len(comparison_df['run_name'].unique())} model combinations")
            
            # Quick Insights Section
            st.subheader("🎯 Quick Insights")
            
            # Calculate insights
            total_lessons = len(comparison_df)
            perfect_count = (comparison_df['match_status'] == 'Perfect').sum()
            perfect_pct = (perfect_count / total_lessons * 100) if total_lessons > 0 else 0
            
            # Best and worst performing runs
            run_metrics_summary = comparison_df.groupby('run_name').agg({
                'f1_score': 'mean',
                'precision': 'mean',
                'recall': 'mean',
                'accuracy': 'mean',
                'match_status': lambda x: (x == 'Perfect').sum()
            }).reset_index()
            run_metrics_summary.columns = ['run_name', 'avg_f1', 'avg_precision', 'avg_recall', 'avg_accuracy', 'perfect_count']
            
            best_run = run_metrics_summary.loc[run_metrics_summary['avg_f1'].idxmax()]
            worst_run = run_metrics_summary.loc[run_metrics_summary['avg_f1'].idxmin()]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Perfect Matches", f"{perfect_count}", f"{perfect_pct:.1f}% of total")
            
            with col2:
                st.metric("Best Run (F1)", clean_run_name(best_run['run_name']), f"{best_run['avg_f1']:.3f}")
            
            with col3:
                st.metric("Worst Run (F1)", clean_run_name(worst_run['run_name']), f"{worst_run['avg_f1']:.3f}")
            
            with col4:
                total_runs = len(comparison_df['run_name'].unique())
                st.metric("Model Combinations", total_runs, f"{total_lessons} total rows")
            
            # Overview metrics with better formatting
            st.subheader("📊 Overall Detection Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_precision = comparison_df['precision'].mean()
                st.metric("Average Precision", f"{avg_precision*100:.1f}%", f"{avg_precision:.3f}")
            
            with col2:
                avg_recall = comparison_df['recall'].mean()
                st.metric("Average Recall", f"{avg_recall*100:.1f}%", f"{avg_recall:.3f}")
            
            with col3:
                avg_f1 = comparison_df['f1_score'].mean()
                st.metric("Average F1 Score", f"{avg_f1*100:.1f}%", f"{avg_f1:.3f}")
            
            with col4:
                avg_accuracy = comparison_df['accuracy'].mean()
                st.metric("Average Accuracy", f"{avg_accuracy*100:.1f}%", f"{avg_accuracy:.3f}")
            
            # Match status distribution
            st.subheader("Match Status Distribution")
            match_status_counts = comparison_df['match_status'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_match = px.pie(
                    values=match_status_counts.values,
                    names=match_status_counts.index,
                    title="Distribution of Match Status",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_match.update_layout(height=400)
                st.plotly_chart(fig_match, use_container_width=True)
            
            with col2:
                # Summary table
                summary_data = []
                for status in match_status_counts.index:
                    status_df = comparison_df[comparison_df['match_status'] == status]
                    summary_data.append({
                        'Status': status,
                        'Count': len(status_df),
                        'Percentage': (len(status_df) / len(comparison_df)) * 100,
                        'Avg Precision': status_df['precision'].mean(),
                        'Avg Recall': status_df['recall'].mean(),
                        'Avg F1': status_df['f1_score'].mean()
                    })
                
                summary_table = pd.DataFrame(summary_data)
                st.dataframe(summary_table, use_container_width=True)
            
            # Metrics by run
            st.subheader("📈 Detection Metrics by Run")
            
            # Calculate metrics by run
            run_metrics_list = []
            for run_name in comparison_df['run_name'].unique():
                run_df = comparison_df[comparison_df['run_name'] == run_name]
                run_metrics_list.append({
                    'run_name': run_name,
                    'precision': run_df['precision'].mean(),
                    'recall': run_df['recall'].mean(),
                    'f1_score': run_df['f1_score'].mean(),
                    'accuracy': run_df['accuracy'].mean(),
                    'true_positives': run_df['true_positives'].mean(),
                    'false_positives': run_df['false_positives'].mean(),
                    'false_negatives': run_df['false_negatives'].mean(),
                    'perfect_matches': (run_df['match_status'] == 'Perfect').sum(),
                    'perfect_pct': (run_df['match_status'] == 'Perfect').sum() / len(run_df) * 100,
                    'total_rows': len(run_df)
                })
            
            run_metrics = pd.DataFrame(run_metrics_list)
            
            run_metrics['run_name_display'] = run_metrics['run_name'].apply(clean_run_name)
            run_metrics = run_metrics.sort_values('f1_score', ascending=False)
            
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Multi-metric comparison chart
                fig_metrics = go.Figure()
                
                fig_metrics.add_trace(go.Bar(
                    name='Precision',
                    x=run_metrics['run_name_display'],
                    y=run_metrics['precision'] * 100,
                    marker_color='#1f77b4',
                    text=run_metrics['precision'].apply(lambda x: f"{x*100:.1f}%"),
                    textposition='outside'
                ))
                fig_metrics.add_trace(go.Bar(
                    name='Recall',
                    x=run_metrics['run_name_display'],
                    y=run_metrics['recall'] * 100,
                    marker_color='#ff7f0e',
                    text=run_metrics['recall'].apply(lambda x: f"{x*100:.1f}%"),
                    textposition='outside'
                ))
                fig_metrics.add_trace(go.Bar(
                    name='F1 Score',
                    x=run_metrics['run_name_display'],
                    y=run_metrics['f1_score'] * 100,
                    marker_color='#2ca02c',
                    text=run_metrics['f1_score'].apply(lambda x: f"{x*100:.1f}%"),
                    textposition='outside'
                ))
                
                fig_metrics.update_layout(
                    title='Precision, Recall, and F1 Score by Run',
                    xaxis_title='Model Combination',
                    yaxis_title='Score (%)',
                    barmode='group',
                    height=500,
                    xaxis=dict(tickangle=-45),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            with col2:
                # Perfect match rate by run
                fig_perfect = px.bar(
                    run_metrics,
                    x='run_name_display',
                    y='perfect_pct',
                    title="Perfect Match Rate by Run",
                    color='perfect_pct',
                    color_continuous_scale='RdYlGn',
                    labels={'perfect_pct': 'Perfect Match Rate (%)', 'run_name_display': 'Run Name'},
                    text='perfect_pct'
                )
                fig_perfect.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_perfect.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig_perfect, use_container_width=True)
            
            # Enhanced metrics table
            display_cols = {
                'run_name_display': 'Run Name',
                'precision': 'Precision (%)',
                'recall': 'Recall (%)',
                'f1_score': 'F1 Score (%)',
                'accuracy': 'Accuracy (%)',
                'perfect_matches': 'Perfect Matches',
                'perfect_pct': 'Perfect Rate (%)',
                'true_positives': 'Avg TP',
                'false_positives': 'Avg FP',
                'false_negatives': 'Avg FN',
                'total_rows': 'Total Rows'
            }
            
            # Convert 0-1 scale to percentages for display
            run_metrics_display = run_metrics[list(display_cols.keys())].copy()
            
            # Convert metrics to percentages (multiply by 100)
            for col in ['precision', 'recall', 'f1_score', 'accuracy']:
                if col in run_metrics_display.columns:
                    run_metrics_display[col] = run_metrics_display[col] * 100
            
            run_metrics_display = run_metrics_display.rename(columns=display_cols)
            
            # Format percentage columns
            for col in ['Precision (%)', 'Recall (%)', 'F1 Score (%)', 'Accuracy (%)', 'Perfect Rate (%)']:
                if col in run_metrics_display.columns:
                    run_metrics_display[col] = run_metrics_display[col].apply(lambda x: f"{float(x):.1f}%")
            
            st.dataframe(run_metrics_display, use_container_width=True)
            
            # Download button for run metrics
            csv_run_metrics = run_metrics_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Run Metrics CSV",
                data=csv_run_metrics,
                file_name="run_metrics_detection.csv",
                mime="text/csv"
            )
            
            # Percentage Heatmap by Category and Run
            st.subheader("📊 Percentage Heatmap: Metrics by Category and Run")
            st.markdown("Heatmap showing detection metrics as percentages across different categories and model combinations.")
            
            # Explanation of how percentages are calculated
            with st.expander("📖 How are these percentages calculated?", expanded=True):
                st.markdown("""
                ### Understanding TP, FP, FN, TN with Multiple Categories:
                
                Since we check for **multiple subcategories** per lesson plan, each category is counted individually:
                
                **Example: Human expects "N7, T3" and model detects "N7, T3, U1"**
                - **TP (True Positives) = 2**: N7 ✓ (expected AND detected), T3 ✓ (expected AND detected)
                - **FP (False Positives) = 1**: U1 ✗ (detected but NOT expected)
                - **FN (False Negatives) = 0**: No expected categories were missed
                - **TN (True Negatives)**: Count of categories expected NOT to be detected (via "no X") that were correctly NOT detected
                
                **Example: Human expects "N7, T3" and model detects only "N7"**
                - **TP = 1**: N7 ✓ (expected AND detected)
                - **FP = 0**: No unexpected detections
                - **FN = 1**: T3 ✗ (expected but NOT detected)
                - **TN**: Count of "no X" categories correctly not detected
                
                **Example: Human expects "N7" and model detects "N7, T3, U1"**
                - **TP = 1**: N7 ✓ (expected AND detected)
                - **FP = 2**: T3 ✗, U1 ✗ (detected but NOT expected)
                - **FN = 0**: All expected categories were detected
                - **TN**: Count of "no X" categories correctly not detected
                
                ### Match Status Classification:
                
                The match status is determined by comparing TP, FP, and FN:
                
                - **Perfect**: TP = Total Expected AND FP = 0 (all expected detected, no unexpected)
                - **Correct + Extra**: TP = Total Expected BUT FP > 0 (all expected detected, but some extra categories)
                - **Partial**: TP < Total Expected BUT FP = 0 (some expected detected, no unexpected)
                - **Incorrect**: TP < Total Expected AND FP > 0 (missing expected categories OR has unexpected)
                
                ### Calculation Process:
                
                **Step 1: Calculate Base Metrics for Each Row**
                For each lesson plan row, we compare the `Human_expected` categories with the `comprehensive_flagged_categories` 
                to calculate detection metrics:
                
                - **True Positives (TP)**: Count of categories that were expected AND detected
                - **False Positives (FP)**: Count of categories that were detected but NOT expected (including those explicitly expected NOT to be detected)
                - **False Negatives (FN)**: Count of categories that were expected but NOT detected
                - **True Negatives (TN)**: Count of categories expected NOT to be detected (via "no X" format) that were indeed NOT detected
                
                **Step 2: Calculate Metrics (0-1 scale)**
                - **Precision** = TP / (TP + FP) = "Of all categories detected, how many were correct?"
                - **Recall** = TP / (TP + FN) = "Of all expected categories, how many were detected?"
                - **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) = Harmonic mean of Precision and Recall
                - **Accuracy** = (TP + TN) / (TP + TN + FP + FN) = "Overall correctness including expected detections and non-detections"
                
                **Step 3: Group by Category and Run**
                - For each category (e.g., "N7", "T3", "NO_R") and each model combination (run):
                  - Collect all metric values (precision, recall, F1, or accuracy) for rows where that category appears in `Human_expected`
                  - Calculate the **average** of these metric values
                
                **Step 4: Convert to Percentage**
                - Multiply the average metric (0-1 scale) by 100 to get percentage (0-100%)
                - Example: If average F1 score = 0.75, then percentage = 75.0%
                
                ### Detailed Example with Multiple Categories:
                
                **Scenario 1: Perfect Match**
                - Human expects: "N7, T3"
                - Model detects: "N7, T3"
                - Match Status: Perfect ✓
                
                **Scenario 2: Correct + Extra**
                - Human expects: "N7"
                - Model detects: "N7, T3, U1"
                - Match Status: Correct + Extra (all expected detected, but extra categories)
                
                **Scenario 3: Partial Match**
                - Human expects: "N7, T3"
                - Model detects: "N7"
                - Match Status: Partial (some expected detected, no unexpected)
                
                **Scenario 4: Incorrect**
                - Human expects: "N7, T3"
                - Model detects: "N6, U1"
                - Match Status: Incorrect (missing expected AND has unexpected)
                
                **Sample Size (n)**: The number of lesson plans that have this category in their `Human_expected` field.
                """)
                
                # Add a visualization showing TP/FP/FN breakdown by match status
                st.markdown("### 📊 Match Status Breakdown:")
                
                # Get match status distribution with TP/FP/FN averages
                match_status_stats = []
                for status in ['Perfect', 'Correct + Extra', 'Partial', 'Incorrect']:
                    status_df = comparison_df[comparison_df['match_status'] == status]
                    if len(status_df) > 0:
                        match_status_stats.append({
                            'Match Status': status,
                            'Count': len(status_df),
                            'Avg TP': status_df['true_positives'].mean(),
                            'Avg FP': status_df['false_positives'].mean(),
                            'Avg FN': status_df['false_negatives'].mean(),
                            'Avg Precision': status_df['precision'].mean() * 100,
                            'Avg Recall': status_df['recall'].mean() * 100,
                            'Avg F1': status_df['f1_score'].mean() * 100
                        })
                
                if match_status_stats:
                    stats_df = pd.DataFrame(match_status_stats)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Create a bar chart showing TP/FP/FN by match status
                    fig_breakdown = go.Figure()
                    
                    stats_df_for_chart = stats_df.copy()
                    fig_breakdown.add_trace(go.Bar(
                        name='Avg TP',
                        x=stats_df_for_chart['Match Status'],
                        y=stats_df_for_chart['Avg TP'],
                        marker_color='#2ca02c',
                        text=stats_df_for_chart['Avg TP'].round(2),
                        textposition='outside'
                    ))
                    fig_breakdown.add_trace(go.Bar(
                        name='Avg FP',
                        x=stats_df_for_chart['Match Status'],
                        y=stats_df_for_chart['Avg FP'],
                        marker_color='#d62728',
                        text=stats_df_for_chart['Avg FP'].round(2),
                        textposition='outside'
                    ))
                    fig_breakdown.add_trace(go.Bar(
                        name='Avg FN',
                        x=stats_df_for_chart['Match Status'],
                        y=stats_df_for_chart['Avg FN'],
                        marker_color='#ff7f0e',
                        text=stats_df_for_chart['Avg FN'].round(2),
                        textposition='outside'
                    ))
                    
                    fig_breakdown.update_layout(
                        title='Average TP, FP, FN by Match Status',
                        xaxis_title='Match Status',
                        yaxis_title='Average Count',
                        barmode='group',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # All metrics to display
            metric_options = {
                'Accuracy': 'accuracy',
                'F1 Score': 'f1_score',
                'Precision': 'precision',
                'Recall': 'recall'
            }
            
            # Helper to extract category code
            def extract_category_code_for_heatmap(cat: str) -> str:
                cat = str(cat).strip().upper()
                if len(cat) >= 2 and cat[0].isalpha() and (cat[1].isdigit() or (len(cat) > 2 and cat[1:].isdigit())):
                    return cat
                if len(cat) == 1 and cat.isalpha():
                    return cat
                return cat
            
            # Loop through all metrics and display each heatmap
            for selected_metric, metric_column in metric_options.items():
                st.markdown(f"### {selected_metric} Heatmap")
                
                # Build category-based metrics data for this metric
                category_metrics_data = {}
                
                # Build data structure: {category: {run: [metric_values]}}
                for idx, row in comparison_df.iterrows():
                    run_name = row.get('run_name', 'Unknown')
                    human_expected = row.get('human_expected', '')
                    metric_value = row.get(metric_column, 0)
                    positive_expected, negative_expected = parse_human_expected(human_expected)
                    
                    # Process positive expectations
                    for expected_cat in positive_expected:
                        cat_key = extract_category_code_for_heatmap(expected_cat)
                        if cat_key not in category_metrics_data:
                            category_metrics_data[cat_key] = {}
                        if run_name not in category_metrics_data[cat_key]:
                            category_metrics_data[cat_key][run_name] = []
                        category_metrics_data[cat_key][run_name].append(metric_value)
                    
                    # Process negative expectations
                    for not_expected_cat in negative_expected:
                        cat_key = extract_category_code_for_heatmap(not_expected_cat)
                        cat_key_negative = f"NO_{cat_key}"
                        if cat_key_negative not in category_metrics_data:
                            category_metrics_data[cat_key_negative] = {}
                        if run_name not in category_metrics_data[cat_key_negative]:
                            category_metrics_data[cat_key_negative][run_name] = []
                        category_metrics_data[cat_key_negative][run_name].append(metric_value)
                
                # Create heatmap data
                if category_metrics_data:
                    # Sort categories
                    def category_sort_key_heatmap(cat: str):
                        if cat.startswith('NO_'):
                            return (1, cat)
                        import re
                        match = re.match(r'([A-Z])(\d+)', cat)
                        if match:
                            letter, num = match.groups()
                            return (0, letter, int(num))
                        return (0, cat)
                    
                    all_categories_sorted = sorted(category_metrics_data.keys(), key=category_sort_key_heatmap)
                    all_runs_sorted = sorted(comparison_df['run_name'].unique())
                    
                    # Build heatmap matrix: rows = categories, columns = runs
                    heatmap_matrix = []
                    sample_size_matrix = []
                    category_names_display = []
                    
                    for cat_name in all_categories_sorted:
                        category_names_display.append(cat_name)
                        row_values = []
                        sample_row = []
                        
                        for run_name in all_runs_sorted:
                            if run_name in category_metrics_data[cat_name]:
                                # Calculate average metric for this category-run combination
                                values = category_metrics_data[cat_name][run_name]
                                sample_size = len(values)
                                avg_value = np.mean(values) if values else 0
                                # Convert to percentage (multiply by 100)
                                row_values.append(avg_value * 100)
                                sample_row.append(sample_size)
                            else:
                                row_values.append(np.nan)
                                sample_row.append(0)
                        
                        heatmap_matrix.append(row_values)
                        sample_size_matrix.append(sample_row)
                    
                    if heatmap_matrix:
                        # Create text labels with sample sizes
                        text_data = []
                        for i, row in enumerate(heatmap_matrix):
                            text_row = []
                            for j, val in enumerate(row):
                                sample_size = sample_size_matrix[i][j]
                                if not np.isnan(val) and sample_size > 0:
                                    text_row.append(f"{val:.1f}%<br>(n={sample_size})")
                                else:
                                    text_row.append("")
                            text_data.append(text_row)
                        
                        # Create heatmap with increased font size
                        # Create heatmap without colorbar initially to avoid validation issues
                        heatmap_trace = go.Heatmap(
                            z=heatmap_matrix,
                            x=[clean_run_name(run_name) for run_name in all_runs_sorted],
                            y=category_names_display,
                            colorscale='RdYlGn',
                            reversescale=False,  # Higher is better (green)
                            text=text_data,
                            texttemplate='%{text}',
                            textfont={"size": 14},  # Increased font size from 8 to 14
                            customdata=sample_size_matrix,
                            hoverongaps=False,
                            hovertemplate=f'<b>%{{y}}</b><br>%{{x}}<br>{selected_metric}: %{{z:.1f}}%<br>Sample Size: %{{customdata}}<extra></extra>'
                        )
                        
                        fig_heatmap_metrics = go.Figure(data=heatmap_trace)
                        
                        # Set colorbar title via update_traces to avoid validation issues
                        fig_heatmap_metrics.update_traces(
                            colorbar=dict(title=dict(text=f"{selected_metric} (%)"))
                        )
                        
                        fig_heatmap_metrics.update_layout(
                            title=f"{selected_metric} Heatmap by Category and Run<br><sub>Values show {selected_metric} (%) and sample size (n)</sub>",
                            xaxis_title="Model Combinations",
                            yaxis_title="Categories",
                            height=max(800, len(category_names_display) * 30 + 200),
                            xaxis=dict(tickangle=-45),
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=120, r=50, t=120, b=200)
                        )
                        
                        st.plotly_chart(fig_heatmap_metrics, use_container_width=True)
                        st.markdown("---")  # Separator between metrics
            
            # Category Performance Pie Charts by Run (collapsible)
            with st.expander("📊 Category Performance by Run (Pie Charts)", expanded=False):
                st.markdown("Pie charts showing match status distribution (Perfect, Correct + Extra, Partial, Incorrect) for each category in each run.")
                
                # Build category performance data using match_status: {category: {run: {Perfect, Correct + Extra, Partial, Incorrect}}}
                category_performance_by_category = {}
                
                # Helper to extract category code
                def extract_category_code(cat: str) -> str:
                    cat = str(cat).strip().upper()
                    if len(cat) >= 2 and cat[0].isalpha() and (cat[1].isdigit() or (len(cat) > 2 and cat[1:].isdigit())):
                        return cat
                    if len(cat) == 1 and cat.isalpha():
                        return cat
                    return cat
                
                # Normalize for comparison
                def normalize_for_comparison(cat: str) -> str:
                    cat = str(cat).lower().strip()
                    if '/' in cat:
                        return cat.split('/')[0]
                    if len(cat) > 1 and cat[0].isalpha() and (cat[1].isdigit() or cat[1].isalpha()):
                        return cat[0]
                    if len(cat) == 1 and cat.isalpha():
                        return cat
                    return cat
                
                for idx, row in comparison_df.iterrows():
                    run_name = row.get('run_name', 'Unknown')
                    human_expected = row.get('human_expected', '')
                    match_status = row.get('match_status', 'Unknown')
                    positive_expected, negative_expected = parse_human_expected(human_expected)
                    
                    # Get flagged categories
                    flagged_str = row.get('comprehensive_flagged_categories', '')
                    flagged_categories = parse_flagged_categories(flagged_str)
                    flagged_norm = [normalize_for_comparison(cat) for cat in flagged_categories]
                    
                    # Process positive expectations
                    for expected_cat in positive_expected:
                        cat_key = extract_category_code(expected_cat)
                        if cat_key not in category_performance_by_category:
                            category_performance_by_category[cat_key] = {}
                        if run_name not in category_performance_by_category[cat_key]:
                            category_performance_by_category[cat_key][run_name] = {
                                'Perfect': 0,
                                'Correct + Extra': 0,
                                'Partial': 0,
                                'Incorrect': 0,
                                'Unknown': 0
                            }
                        
                        # Count by match status
                        if match_status in category_performance_by_category[cat_key][run_name]:
                            category_performance_by_category[cat_key][run_name][match_status] += 1
                        else:
                            category_performance_by_category[cat_key][run_name]['Unknown'] += 1
                    
                    # Process negative expectations
                    for not_expected_cat in negative_expected:
                        cat_key = extract_category_code(not_expected_cat)
                        cat_key_negative = f"NO_{cat_key}"
                        if cat_key_negative not in category_performance_by_category:
                            category_performance_by_category[cat_key_negative] = {}
                        if run_name not in category_performance_by_category[cat_key_negative]:
                            category_performance_by_category[cat_key_negative][run_name] = {
                                'Perfect': 0,
                                'Correct + Extra': 0,
                                'Partial': 0,
                                'Incorrect': 0,
                                'Unknown': 0
                            }
                        
                        # Count by match status
                        if match_status in category_performance_by_category[cat_key_negative][run_name]:
                            category_performance_by_category[cat_key_negative][run_name][match_status] += 1
                        else:
                            category_performance_by_category[cat_key_negative][run_name]['Unknown'] += 1
                
                # Create pie charts organized by category
                if category_performance_by_category:
                    # Select categories to display
                    def category_sort_key(cat: str):
                        if cat.startswith('NO_'):
                            return (1, cat)
                        import re
                        match = re.match(r'([A-Z])(\d+)', cat)
                        if match:
                            letter, num = match.groups()
                            return (0, letter, int(num))
                        return (0, cat)
                    
                    all_categories = sorted(category_performance_by_category.keys(), key=category_sort_key)
                    selected_categories_for_pie = st.multiselect(
                        "Select categories to display (leave empty for all)",
                        options=all_categories,
                        default=all_categories[:min(6, len(all_categories))],  # Reduced default to 6
                        format_func=lambda x: x
                    )
                    
                    if not selected_categories_for_pie:
                        selected_categories_for_pie = all_categories
                    
                    # Get all runs
                    all_runs_list = sorted(comparison_df['run_name'].unique())
                    selected_runs_for_pie = st.multiselect(
                        "Select runs to display (leave empty for all)",
                        options=all_runs_list,
                        default=all_runs_list[:min(4, len(all_runs_list))],  # Default to first 4 runs
                        format_func=clean_run_name
                    )
                    
                    if not selected_runs_for_pie:
                        selected_runs_for_pie = all_runs_list[:min(4, len(all_runs_list))]  # Limit to 4 if all selected
                    
                    # Match status colors
                    status_colors = {
                        'Perfect': '#2ca02c',  # Green
                        'Correct + Extra': '#ffd700',  # Gold
                        'Partial': '#ff7f0e',  # Orange
                        'Incorrect': '#d62728',  # Red
                        'Unknown': '#808080'  # Gray
                    }
                    
                    # Create pie charts for each category
                    num_cols_per_category = min(4, len(selected_runs_for_pie))  # Max 4 charts per row
                    
                    for cat_name in selected_categories_for_pie:
                        if cat_name not in category_performance_by_category:
                            continue
                        
                        cat_data = category_performance_by_category[cat_name]
                        valid_runs = [r for r in selected_runs_for_pie if r in cat_data]
                        
                        if not valid_runs:
                            continue
                        
                        st.markdown(f"### Category: {cat_name}")
                        
                        # Create rows of pie charts for this category
                        for row_start in range(0, len(valid_runs), num_cols_per_category):
                            row_runs = valid_runs[row_start:row_start + num_cols_per_category]
                            cols = st.columns(len(row_runs))
                            
                            for col_idx, run_name in enumerate(row_runs):
                                if col_idx >= len(cols):
                                    break
                                
                                run_stats = cat_data[run_name]
                                total = sum(run_stats.values())
                                
                                if total > 0:
                                    # Prepare pie chart data
                                    pie_labels = []
                                    pie_values = []
                                    pie_colors = []
                                    
                                    # Order: Perfect, Correct + Extra, Partial, Incorrect, Unknown
                                    status_order = ['Perfect', 'Correct + Extra', 'Partial', 'Incorrect', 'Unknown']
                                    for status in status_order:
                                        count = run_stats.get(status, 0)
                                        if count > 0:
                                            pct = (count / total) * 100
                                            pie_labels.append(f"{status}\n({pct:.1f}%)")
                                            pie_values.append(count)  # Use count, not percentage
                                            pie_colors.append(status_colors.get(status, '#808080'))
                                    
                                    with cols[col_idx]:
                                        if pie_values:
                                            # Create pie chart for this run and category
                                            fig_pie = go.Figure(data=[go.Pie(
                                                labels=pie_labels,
                                                values=pie_values,
                                                hole=0.3,
                                                marker_colors=pie_colors,
                                                textinfo='label+percent',
                                                textposition='outside',
                                                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                                            )])
                                            
                                            fig_pie.update_layout(
                                                title=f"{clean_run_name(run_name)}<br>{cat_name}",
                                                height=450,
                                                showlegend=True,
                                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                                            )
                                            st.plotly_chart(fig_pie, use_container_width=True)
                                            
                                            # Show summary metrics below chart
                                            perfect_count = run_stats.get('Perfect', 0)
                                            perfect_pct = (perfect_count / total * 100) if total > 0 else 0
                                            st.metric("Perfect Match Rate", f"{perfect_pct:.1f}%", f"{perfect_count}/{total}")
                        
                        # Summary table for this category across all runs
                        category_summary_data = []
                        for run_name in selected_runs_for_pie:
                            if run_name in cat_data:
                                run_stats = cat_data[run_name]
                                total = sum(run_stats.values())
                                if total > 0:
                                    perfect_count = run_stats.get('Perfect', 0)
                                    correct_extra_count = run_stats.get('Correct + Extra', 0)
                                    partial_count = run_stats.get('Partial', 0)
                                    incorrect_count = run_stats.get('Incorrect', 0)
                                    
                                    perfect_pct = (perfect_count / total) * 100
                                    category_summary_data.append({
                                        'Run': clean_run_name(run_name),
                                        'Perfect': f"{perfect_count} ({perfect_pct:.1f}%)",
                                        'Correct + Extra': f"{correct_extra_count}",
                                        'Partial': f"{partial_count}",
                                        'Incorrect': f"{incorrect_count}",
                                        'Total': total
                                    })
                        
                        if category_summary_data:
                            category_summary_df = pd.DataFrame(category_summary_data)
                            st.dataframe(category_summary_df, use_container_width=True)
                        
                        st.divider()
            
            # Detailed comparison table with enhanced features
            st.subheader("🔍 Detailed Comparison Table")
            
            # Enhanced filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_status = st.multiselect(
                    "Filter by Match Status",
                    options=comparison_df['match_status'].unique(),
                    default=[]
                )
            with col2:
                filter_runs = st.multiselect(
                    "Filter by Run",
                    options=sorted(df['run_name'].unique()),
                    default=[],
                    format_func=clean_run_name
                )
            with col3:
                # Filter by metric thresholds
                metric_filter = st.selectbox(
                    "Filter by Metric Threshold",
                    options=["None", "F1 Score > 0.8", "F1 Score > 0.9", "Precision > 0.8", "Recall > 0.8", "Perfect Matches Only"],
                    index=0
                )
            
            filtered_comparison = comparison_df.copy()
            if filter_status:
                filtered_comparison = filtered_comparison[filtered_comparison['match_status'].isin(filter_status)]
            if filter_runs:
                filtered_comparison = filtered_comparison[filtered_comparison['run_name'].isin(filter_runs)]
            if metric_filter != "None":
                if metric_filter == "F1 Score > 0.8":
                    filtered_comparison = filtered_comparison[filtered_comparison['f1_score'] > 0.8]
                elif metric_filter == "F1 Score > 0.9":
                    filtered_comparison = filtered_comparison[filtered_comparison['f1_score'] > 0.9]
                elif metric_filter == "Precision > 0.8":
                    filtered_comparison = filtered_comparison[filtered_comparison['precision'] > 0.8]
                elif metric_filter == "Recall > 0.8":
                    filtered_comparison = filtered_comparison[filtered_comparison['recall'] > 0.8]
                elif metric_filter == "Perfect Matches Only":
                    filtered_comparison = filtered_comparison[filtered_comparison['match_status'] == 'Perfect']
            
            # Summary of filtered results
            st.info(f"Showing {len(filtered_comparison)} of {len(comparison_df)} rows ({(len(filtered_comparison)/len(comparison_df)*100):.1f}%)")
            
            # Display selected rows
            col1, col2 = st.columns(2)
            with col1:
                num_rows = st.selectbox("Number of rows to display", [10, 25, 50, 100, 500, 1000, "All"], index=2)
            with col2:
                sort_options = ["F1 Score", "Precision", "Recall", "Accuracy", "Match Status", "Run Name"]
                if 'Topic' in df.columns:
                    sort_options.append("Topic")
                sort_by = st.selectbox(
                    "Sort by",
                    options=sort_options,
                    index=0
                )
            
            # Create display dataframe
            base_cols = [
                'run_name', 'row_index', 'human_expected', 'match_status', 
                'precision', 'recall', 'f1_score', 'accuracy',
                'true_positives', 'false_positives', 'false_negatives',
                'total_expected', 'total_detected'
            ]
            
            # Add flagged_categories if available in comparison_df
            if 'flagged_categories' in filtered_comparison.columns:
                base_cols.append('flagged_categories')
            
            display_df = filtered_comparison[base_cols].copy()
            
            # Merge with original df to get dataset, lesson_plan, Topic columns and comprehensive_flagged_categories
            if 'row_index' in display_df.columns:
                # Create a mapping from row_index to original df columns
                for idx, row in display_df.iterrows():
                    row_idx = int(row.get('row_index', idx))
                    if 0 <= row_idx < len(df):
                        original_row = df.iloc[row_idx]
                        
                        # Add columns from original df
                        if 'dataset' in df.columns:
                            display_df.at[idx, 'dataset'] = original_row.get('dataset', '')
                        if 'lesson_plan' in df.columns:
                            lesson_plan_value = original_row.get('lesson_plan', '')
                            # Format lesson_plan - if it's a JSON string, show a truncated version
                            if isinstance(lesson_plan_value, str) and len(lesson_plan_value) > 100:
                                try:
                                    lesson_plan_json = json.loads(lesson_plan_value)
                                    display_df.at[idx, 'lesson_plan'] = json.dumps(lesson_plan_json, indent=2, ensure_ascii=False)[:200] + "..."
                                except (json.JSONDecodeError, ValueError, TypeError):
                                    display_df.at[idx, 'lesson_plan'] = lesson_plan_value[:200] + "..."
                            else:
                                display_df.at[idx, 'lesson_plan'] = lesson_plan_value
                        if 'Topic' in df.columns:
                            display_df.at[idx, 'Topic'] = original_row.get('Topic', '')
                        
                        # Get comprehensive_flagged_categories from original df
                        if 'comprehensive_flagged_categories' in df.columns:
                            flagged_cats = original_row.get('comprehensive_flagged_categories', '')
                            display_df.at[idx, 'comprehensive_flagged_categories'] = flagged_cats
                        elif 'moderation_flagged_categories' in df.columns:
                            flagged_cats = original_row.get('moderation_flagged_categories', '')
                            display_df.at[idx, 'comprehensive_flagged_categories'] = flagged_cats
            
            display_df['run_name_display'] = display_df['run_name'].apply(clean_run_name)
            display_df = display_df.drop('run_name', axis=1)
            
            # Format flagged_categories for display - prefer comprehensive_flagged_categories from original df
            def format_flagged_categories(x):
                if pd.isna(x) or x == '' or x is None:
                    return ''
                if isinstance(x, list):
                    return ', '.join([str(cat) for cat in x])
                if isinstance(x, str):
                    try:
                        # Try to parse as JSON
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return ', '.join([str(cat) for cat in parsed])
                        return str(parsed)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        return str(x)
                return str(x)
            
            # Use comprehensive_flagged_categories if available, otherwise use flagged_categories
            if 'comprehensive_flagged_categories' in display_df.columns:
                display_df['detected_categories'] = display_df['comprehensive_flagged_categories'].apply(format_flagged_categories)
                display_df = display_df.drop('comprehensive_flagged_categories', axis=1)
            elif 'flagged_categories' in display_df.columns:
                display_df['detected_categories'] = display_df['flagged_categories'].apply(format_flagged_categories)
                display_df = display_df.drop('flagged_categories', axis=1)
            
            # Sort by selected column
            sort_mapping = {
                "F1 Score": 'f1_score',
                "Precision": 'precision',
                "Recall": 'recall',
                "Accuracy": 'accuracy',
                "Match Status": 'match_status',
                "Run Name": 'run_name_display',
                "Topic": 'Topic'
            }
            if sort_by in sort_mapping:
                if sort_by == "Match Status":
                    status_order = {'Perfect': 0, 'Correct + Extra': 1, 'Partial': 2, 'Incorrect': 3}
                    display_df['_sort_order'] = display_df['match_status'].map(status_order).fillna(99)
                    display_df = display_df.sort_values('_sort_order', ascending=True)
                    display_df = display_df.drop('_sort_order', axis=1)
                else:
                    display_df = display_df.sort_values(sort_mapping[sort_by], ascending=False)
            
            # Reorder columns - add dataset, lesson_plan, Topic, and detected_categories
            base_cols = ['run_name_display']
            
            # Add dataset, Topic if they exist
            if 'dataset' in display_df.columns:
                base_cols.append('dataset')
            if 'Topic' in display_df.columns:
                base_cols.append('Topic')
            
            # Add human_expected and detected_categories
            base_cols.extend(['human_expected'])
            if 'detected_categories' in display_df.columns:
                base_cols.append('detected_categories')
            
            # Add lesson_plan if it exists
            if 'lesson_plan' in display_df.columns:
                base_cols.append('lesson_plan')
            
            # Add remaining columns
            base_cols.extend([
                'match_status', 
                'precision', 'recall', 'f1_score', 'accuracy',
                'true_positives', 'false_positives', 'false_negatives',
                'total_expected', 'total_detected'
            ])
            
            # Only include columns that actually exist
            cols = [col for col in base_cols if col in display_df.columns]
            display_df = display_df[cols]
            
            # Rename columns for display
            column_mapping = {
                'run_name_display': 'Run Name',
                'dataset': 'Dataset',
                'Topic': 'Topic',
                'human_expected': 'Human Expected',
                'detected_categories': 'Detected Categories',
                'lesson_plan': 'Lesson Plan',
                'match_status': 'Match Status',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1 Score',
                'accuracy': 'Accuracy',
                'true_positives': 'TP',
                'false_positives': 'FP',
                'false_negatives': 'FN',
                'total_expected': 'Expected',
                'total_detected': 'Detected'
            }
            
            display_df = display_df.rename(columns=column_mapping)
            
            # Format percentage columns
            for col in ['Precision', 'Recall', 'F1 Score', 'Accuracy']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{float(x)*100:.1f}%")
            
            # Display rows
            if num_rows == "All":
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(display_df.head(num_rows), use_container_width=True)
            
            # Download button - include all columns from display_df
            download_df = display_df.copy()
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Comparison CSV",
                data=csv,
                file_name="human_expected_comparison.csv",
                mime="text/csv"
            )
            
            # Expandable section for row-by-row details
            st.subheader("Detailed Row Analysis")
            
            # Create options with more descriptive labels
            row_options = []
            for idx in range(len(filtered_comparison)):
                row_data = filtered_comparison.iloc[idx]
                run_name_clean = clean_run_name(row_data.get('run_name', ''))
                status = row_data.get('match_status', 'Unknown')
                human_expected = str(row_data.get('human_expected', ''))[:30]
                row_options.append((idx, f"{status} | {run_name_clean[:20]} | Expected: {human_expected}"))
            
            if row_options:
                selected_option = st.selectbox(
                    "Select row to analyze",
                    options=range(len(row_options)),
                    format_func=lambda x: row_options[x][1]
                )
                
                if selected_option < len(filtered_comparison):
                    row_data = filtered_comparison.iloc[selected_option]
                    # Get original row index from comparison data
                    original_row_idx = int(row_data.get('row_index', selected_option))
                    
                    # Ensure we don't go out of bounds
                    if 0 <= original_row_idx < len(df):
                        original_row = df.iloc[original_row_idx]
                        # Parse negative expectations from original row
                        _, negative_expected = parse_human_expected(original_row.get('Human_expected', ''))
                    else:
                        _, negative_expected = [], []
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Human Expected:**")
                        st.write(f"- Positive (should detect): {row_data['expected_categories'] if row_data['expected_categories'] else 'None'}")
                        st.write(f"- Negative (should NOT detect): {[f'no {cat.upper()}' for cat in negative_expected] if negative_expected else 'None'}")
                        st.write("**Flagged Categories:**")
                        st.write(row_data['flagged_categories'] if row_data['flagged_categories'] else 'None')
                    
                    with col2:
                        st.write("**Metrics:**")
                        st.write(f"- Precision: {row_data['precision']:.3f}")
                        st.write(f"- Recall: {row_data['recall']:.3f}")
                        st.write(f"- F1 Score: {row_data['f1_score']:.3f}")
                        st.write(f"- Accuracy: {row_data['accuracy']:.3f}")
                        st.write(f"- **TP:** {row_data['true_positives']} | **FP:** {row_data['false_positives']} | **FN:** {row_data['false_negatives']}")
                        
                        if row_data.get('unexpected_detected'):
                            st.write(f"**⚠️ Unexpected Detected:** {row_data['unexpected_detected']}")
                        
                        # Show match status
                        status_color = {
                            'Perfect': '🟢',
                            'Correct + Extra': '🟡',
                            'Partial': '🟠',
                            'Incorrect': '🔴'
                        }
                        status_icon = status_color.get(row_data['match_status'], '⚪')
                        st.write(f"**Match Status:** {status_icon} {row_data['match_status']}")
    
    if tab6 is not None:
        with tab6:
            st.header("📚 Lesson Plan Performance Analysis")
            st.markdown("Analyze performance metrics for each lesson plan across all model combinations.")
            
            # Get comparison metrics if available
            comparison_df = None
            if has_human_expected:
                with st.spinner("Calculating detection metrics..."):
                    comparison_df = analyze_human_expected_comparison(df)
            
            # Analyze lesson plan performance
            with st.spinner("Analyzing lesson plan performance..."):
                # Determine lesson ID column
                lesson_id_col = 'id' if 'id' in df.columns else None
                if lesson_id_col is None:
                    possible_id_cols = [c for c in df.columns if 'id' in c.lower()]
                    if possible_id_cols:
                        lesson_id_col = possible_id_cols[0]
                
                lesson_performance_df = analyze_lesson_plan_performance(df, lesson_id_column=lesson_id_col or 'id', comparison_df=comparison_df)
            
            if lesson_performance_df.empty:
                st.warning("No lesson plan performance data available. Please ensure the dataset contains lesson plan identifiers.")
                return
            
            st.success(f"Analyzed {len(lesson_performance_df)} unique lesson plans")
            
            # Quick Insights Section
            st.subheader("🎯 Quick Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Best performing lesson plan
            if 'avg_f1' in lesson_performance_df.columns and lesson_performance_df['avg_f1'].notna().any():
                best_lesson = lesson_performance_df.loc[lesson_performance_df['avg_f1'].idxmax()]
                with col1:
                    st.metric("Best F1 Score", f"{best_lesson['avg_f1']*100:.1f}%", f"Topic: {best_lesson.get('topic', 'N/A')}")
            else:
                best_success = lesson_performance_df.loc[lesson_performance_df['success_rate'].idxmax()]
                with col1:
                    st.metric("Best Success Rate", f"{best_success['success_rate']:.1f}%", f"Topic: {best_success.get('topic', 'N/A')}")
            
            # Worst performing lesson plan
            if 'avg_f1' in lesson_performance_df.columns and lesson_performance_df['avg_f1'].notna().any():
                worst_lesson = lesson_performance_df.loc[lesson_performance_df['avg_f1'].idxmin()]
                with col2:
                    st.metric("Worst F1 Score", f"{worst_lesson['avg_f1']*100:.1f}%", f"Topic: {worst_lesson.get('topic', 'N/A')}")
            else:
                worst_success = lesson_performance_df.loc[lesson_performance_df['success_rate'].idxmin()]
                with col2:
                    st.metric("Lowest Success Rate", f"{worst_success['success_rate']:.1f}%", f"Topic: {worst_success.get('topic', 'N/A')}")
            
            # Most consistent lesson plan
            if 'avg_f1' in lesson_performance_df.columns:
                # Calculate std dev of F1 scores (lower is more consistent)
                lesson_performance_df['f1_std'] = lesson_performance_df.get('f1_std', np.nan)
                if lesson_performance_df['f1_std'].notna().any():
                    most_consistent = lesson_performance_df.loc[lesson_performance_df['f1_std'].idxmin()]
                    with col3:
                        st.metric("Most Consistent", f"Std: {most_consistent['f1_std']:.3f}", f"Topic: {most_consistent.get('topic', 'N/A')}")
                else:
                    with col3:
                        total_lesson_plans = len(lesson_performance_df)
                        st.metric("Total Lesson Plans", total_lesson_plans)
            else:
                with col3:
                    total_lesson_plans = len(lesson_performance_df)
                    st.metric("Total Lesson Plans", total_lesson_plans)
            
            # Average metrics
            with col4:
                if 'avg_f1' in lesson_performance_df.columns:
                    avg_f1_overall = lesson_performance_df['avg_f1'].dropna().mean()
                    st.metric("Avg F1 Score", f"{avg_f1_overall*100:.1f}%" if not pd.isna(avg_f1_overall) else "N/A")
                else:
                    avg_success_rate = lesson_performance_df['success_rate'].mean()
                    st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
            
            # Overview metrics with better formatting
            st.subheader("📊 Overall Lesson Plan Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_success_rate = lesson_performance_df['success_rate'].mean()
                st.metric("Average Success Rate", f"{avg_success_rate:.1f}%", f"{avg_success_rate:.3f}")
            
            with col2:
                total_lesson_plans = len(lesson_performance_df)
                total_runs = lesson_performance_df['num_runs'].sum()
                st.metric("Total Lesson Plans", total_lesson_plans, f"{total_runs} total runs")
            
            with col3:
                if 'avg_f1' in lesson_performance_df.columns:
                    avg_f1_overall = lesson_performance_df['avg_f1'].dropna().mean()
                    st.metric("Average F1 Score", f"{avg_f1_overall*100:.1f}%" if not pd.isna(avg_f1_overall) else "N/A", f"{avg_f1_overall:.3f}" if not pd.isna(avg_f1_overall) else "N/A")
                else:
                    st.metric("Average F1 Score", "N/A")
            
            with col4:
                avg_num_runs = lesson_performance_df['num_runs'].mean()
                max_runs = lesson_performance_df['num_runs'].max()
                st.metric("Avg Runs per Lesson", f"{avg_num_runs:.1f}", f"Max: {max_runs}")
            
            # Enhanced visualizations
            st.subheader("📈 Performance Distribution Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Success rate distribution
                fig_success_dist = px.histogram(
                    lesson_performance_df,
                    x='success_rate',
                    nbins=20,
                    title="Success Rate Distribution",
                    labels={'success_rate': 'Success Rate (%)', 'count': 'Number of Lesson Plans'},
                    color_discrete_sequence=['#2ca02c']
                )
                fig_success_dist.update_traces(marker=dict(color='#2ca02c', line=dict(color='#1f77b4', width=1)))
                fig_success_dist.update_layout(height=400)
                st.plotly_chart(fig_success_dist, use_container_width=True)
            
            with col2:
                # F1 Score distribution if available
                if 'avg_f1' in lesson_performance_df.columns and lesson_performance_df['avg_f1'].notna().any():
                    fig_f1_dist = px.histogram(
                        lesson_performance_df,
                        x='avg_f1',
                        nbins=20,
                        title="F1 Score Distribution",
                        labels={'avg_f1': 'F1 Score', 'count': 'Number of Lesson Plans'},
                        color_discrete_sequence=['#2ca02c']
                    )
                    fig_f1_dist.update_traces(marker=dict(color='#2ca02c', line=dict(color='#1f77b4', width=1)))
                    fig_f1_dist.update_layout(height=400, xaxis=dict(tickformat='.2f'))
                    st.plotly_chart(fig_f1_dist, use_container_width=True)
                else:
                    # Number of runs distribution
                    fig_runs_dist = px.histogram(
                        lesson_performance_df,
                        x='num_runs',
                        nbins=20,
                        title="Number of Runs Distribution",
                        labels={'num_runs': 'Number of Runs', 'count': 'Number of Lesson Plans'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_runs_dist.update_traces(marker=dict(color='#1f77b4', line=dict(color='#0d47a1', width=1)))
                    fig_runs_dist.update_layout(height=400)
                    st.plotly_chart(fig_runs_dist, use_container_width=True)
            
            # Top and bottom performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 10 Performers")
                if 'avg_f1' in lesson_performance_df.columns and lesson_performance_df['avg_f1'].notna().any():
                    top_performers = lesson_performance_df.nlargest(10, 'avg_f1')[['lesson_id', 'topic', 'avg_f1', 'success_rate', 'num_runs']].copy()
                    top_performers['avg_f1'] = top_performers['avg_f1'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
                    top_performers.columns = ['Lesson ID', 'Topic', 'F1 Score', 'Success Rate (%)', 'Num Runs']
                    st.dataframe(top_performers, use_container_width=True)
                else:
                    top_performers = lesson_performance_df.nlargest(10, 'success_rate')[['lesson_id', 'topic', 'success_rate', 'num_runs']].copy()
                    top_performers.columns = ['Lesson ID', 'Topic', 'Success Rate (%)', 'Num Runs']
                    st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.subheader("⚠️ Bottom 10 Performers")
                if 'avg_f1' in lesson_performance_df.columns and lesson_performance_df['avg_f1'].notna().any():
                    bottom_performers = lesson_performance_df.nsmallest(10, 'avg_f1')[['lesson_id', 'topic', 'avg_f1', 'success_rate', 'num_runs']].copy()
                    bottom_performers['avg_f1'] = bottom_performers['avg_f1'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
                    bottom_performers.columns = ['Lesson ID', 'Topic', 'F1 Score', 'Success Rate (%)', 'Num Runs']
                    st.dataframe(bottom_performers, use_container_width=True)
                else:
                    bottom_performers = lesson_performance_df.nsmallest(10, 'success_rate')[['lesson_id', 'topic', 'success_rate', 'num_runs']].copy()
                    bottom_performers.columns = ['Lesson ID', 'Topic', 'Success Rate (%)', 'Num Runs']
                    st.dataframe(bottom_performers, use_container_width=True)
            
            # Topic-wise analysis if Topic column exists
            if 'topic' in lesson_performance_df.columns:
                st.subheader("📊 Topic-wise Performance Analysis")
                
                topic_stats = lesson_performance_df.groupby('topic').agg({
                    'success_rate': ['mean', 'std', 'count'],
                    'num_runs': 'mean'
                }).reset_index()
                topic_stats.columns = ['Topic', 'Avg Success Rate', 'Std Success Rate', 'Count', 'Avg Runs']
                
                if 'avg_f1' in lesson_performance_df.columns:
                    topic_f1 = lesson_performance_df.groupby('topic')['avg_f1'].agg(['mean', 'std']).reset_index()
                    topic_f1.columns = ['Topic', 'Avg F1', 'Std F1']
                    topic_stats = topic_stats.merge(topic_f1, on='Topic', how='left')
                
                # Sort by performance
                sort_col = 'Avg F1' if 'Avg F1' in topic_stats.columns else 'Avg Success Rate'
                topic_stats = topic_stats.sort_values(sort_col, ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_topic = px.bar(
                        topic_stats,
                        x='Topic',
                        y=sort_col if 'Avg F1' in topic_stats.columns else 'Avg Success Rate',
                        title=f"Performance by Topic ({sort_col})",
                        color=sort_col if 'Avg F1' in topic_stats.columns else 'Avg Success Rate',
                        color_continuous_scale='RdYlGn',
                        text=sort_col if 'Avg F1' in topic_stats.columns else 'Avg Success Rate'
                    )
                    if 'Avg F1' in topic_stats.columns:
                        fig_topic.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    else:
                        fig_topic.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_topic.update_layout(height=500, xaxis_tickangle=-45, yaxis_title=sort_col)
                    st.plotly_chart(fig_topic, use_container_width=True)
                
                with col2:
                    st.dataframe(topic_stats, use_container_width=True)
                    
                    csv_topic = topic_stats.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Topic Stats CSV",
                        data=csv_topic,
                        file_name="topic_performance_stats.csv",
                        mime="text/csv"
                    )
            
            # Create pivot table: one row per lesson plan, columns for each run's metrics
            st.subheader("📊 Lesson Plan Performance Comparison")
            st.markdown("One row per lesson plan, with columns showing metrics for each run (model combination).")
            
            # Enhanced Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_topic = st.multiselect(
                    "Filter by Topic",
                    options=sorted(lesson_performance_df['topic'].unique()) if 'topic' in lesson_performance_df.columns else [],
                    default=[],
                    key="lesson_perf_filter_topic"
                )
            with col2:
                # Filter by success rate threshold
                min_success_rate = st.slider(
                    "Minimum Success Rate (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5,
                    key="lesson_perf_min_success"
                )
            with col3:
                # Show unique lesson plans count
                total_lessons = len(lesson_performance_df)
                st.metric("Unique Lesson Plans", total_lessons)
            
            # Filter lesson plans
            filtered_lesson_df = lesson_performance_df.copy()
            if filter_topic and 'topic' in filtered_lesson_df.columns:
                filtered_lesson_df = filtered_lesson_df[filtered_lesson_df['topic'].isin(filter_topic)]
            if min_success_rate > 0:
                filtered_lesson_df = filtered_lesson_df[filtered_lesson_df['success_rate'] >= min_success_rate]
            
            st.info(f"Showing {len(filtered_lesson_df)} of {len(lesson_performance_df)} lesson plans ({(len(filtered_lesson_df)/len(lesson_performance_df)*100):.1f}%)")
            
            # Get all unique runs
            all_runs = sorted(df['run_name'].unique()) if 'run_name' in df.columns else []
            
            # Create comparison metrics map for quick lookup
            comparison_metrics_map = {}
            if comparison_df is not None:
                for comp_idx, comp_row in comparison_df.iterrows():
                    original_row_idx = comp_row.get('row_index')
                    if original_row_idx is not None and original_row_idx < len(df):
                        original_row = df.iloc[original_row_idx]
                        lesson_id = original_row.get('id')
                        run_name = comp_row.get('run_name')
                        if lesson_id and run_name:
                            key = (lesson_id, run_name)
                            comparison_metrics_map[key] = {
                                'match_status': comp_row.get('match_status', 'Unknown'),
                                'precision': comp_row.get('precision'),
                                'recall': comp_row.get('recall'),
                                'f1_score': comp_row.get('f1_score'),
                                'accuracy': comp_row.get('accuracy')
                            }
            
            # Build pivot table
            pivot_data = []
            
            for lesson_idx, lesson_row in filtered_lesson_df.iterrows():
                lesson_id = lesson_row['lesson_id']
                
                # Get all runs for this lesson plan
                lesson_runs_df = df[df['id'] == lesson_id] if 'id' in df.columns else pd.DataFrame()
                
                if lesson_runs_df.empty:
                    continue
                
                # Get lesson plan content (from first run)
                first_run_row = lesson_runs_df.iloc[0]
                lesson_plan_content = first_run_row.get('lesson_plan', '')
                lesson_plan_str = ''
                if pd.notna(lesson_plan_content) and lesson_plan_content:
                    try:
                        if isinstance(lesson_plan_content, str):
                            lesson_plan_json = json.loads(lesson_plan_content)
                            lesson_plan_str = json.dumps(lesson_plan_json, indent=2, ensure_ascii=False)
                        else:
                            lesson_plan_str = json.dumps(lesson_plan_content, indent=2, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        lesson_plan_str = str(lesson_plan_content)
                
                # Get Human_expected
                human_expected = first_run_row.get('Human_expected', '')
                
                # Start row with lesson plan info
                row_data = {
                    'Lesson ID': lesson_id,
                    'Topic': lesson_row['topic'],
                    'Human Expected': human_expected,
                    'Lesson Plan': lesson_plan_str,
                    'Num Runs': len(lesson_runs_df),
                    'Success Rate': lesson_row['success_rate']
                }
                
                # Add columns for each run
                for run_name in all_runs:
                    run_clean_name = clean_run_name(run_name)
                    
                    # Find this run's data for this lesson plan
                    run_row = lesson_runs_df[lesson_runs_df['run_name'] == run_name]
                    
                    if not run_row.empty:
                        run_row = run_row.iloc[0]
                        
                        # Get flagged categories
                        flagged_str = run_row.get('comprehensive_flagged_categories', '')
                        flagged_categories = parse_flagged_categories(flagged_str)
                        
                        # Get comparison metrics
                        key = (lesson_id, run_name)
                        metrics = comparison_metrics_map.get(key, {})
                        
                        # Parse Human_expected for comparison
                        positive_expected, negative_expected = parse_human_expected(human_expected)
                        
                        # Normalize categories for comparison
                        def normalize_category(cat: str) -> str:
                            cat = str(cat).lower().strip()
                            if '/' in cat:
                                return cat.split('/')[0]
                            if len(cat) > 1 and cat[0].isalpha() and (cat[1].isdigit() or cat[1].isalpha()):
                                return cat[0]
                            if len(cat) == 1 and cat.isalpha():
                                return cat
                            return cat
                        
                        positive_expected_norm = [normalize_category(cat) for cat in positive_expected]
                        flagged_norm = [normalize_category(cat) for cat in flagged_categories]
                        
                        expected_detected = bool([cat for cat in positive_expected_norm if cat in flagged_norm])
                        
                        # Add run-specific columns
                        row_data[f'{run_clean_name} - Flagged Categories'] = ', '.join(flagged_categories) if flagged_categories else 'None'
                        row_data[f'{run_clean_name} - Expected Detected'] = '✅' if expected_detected else '❌'
                        row_data[f'{run_clean_name} - Match Status'] = metrics.get('match_status', 'Unknown')
                        row_data[f'{run_clean_name} - F1 Score'] = f"{metrics.get('f1_score', 0):.3f}" if metrics.get('f1_score') is not None else "N/A"
                        row_data[f'{run_clean_name} - Precision'] = f"{metrics.get('precision', 0):.3f}" if metrics.get('precision') is not None else "N/A"
                        row_data[f'{run_clean_name} - Recall'] = f"{metrics.get('recall', 0):.3f}" if metrics.get('recall') is not None else "N/A"
                        row_data[f'{run_clean_name} - Status'] = run_row.get('final_status', 'Unknown')
                    else:
                        # Run not found for this lesson plan
                        row_data[f'{run_clean_name} - Flagged Categories'] = 'N/A'
                        row_data[f'{run_clean_name} - Expected Detected'] = 'N/A'
                        row_data[f'{run_clean_name} - Match Status'] = 'N/A'
                        row_data[f'{run_clean_name} - F1 Score'] = 'N/A'
                        row_data[f'{run_clean_name} - Precision'] = 'N/A'
                        row_data[f'{run_clean_name} - Recall'] = 'N/A'
                        row_data[f'{run_clean_name} - Status'] = 'N/A'
                
                pivot_data.append(row_data)
            
            # Create DataFrame
            pivot_table = pd.DataFrame(pivot_data)
            
            if not pivot_table.empty:
                # Enhanced sorting options
                col1, col2 = st.columns(2)
                with col1:
                    sort_by = st.selectbox(
                        "Sort by",
                        options=["Topic", "Success Rate", "Num Runs", "Lesson ID"] + 
                                ([f"{run_clean_name} - F1 Score" for run_clean_name in [clean_run_name(r) for r in all_runs]] if all_runs else []),
                        index=0,
                        key="lesson_perf_sort_by"
                    )
                with col2:
                    num_rows_display = st.selectbox(
                        "Number of rows to display",
                        options=[10, 25, 50, 100, 500, 1000, "All"],
                        index=2,
                        key="lesson_perf_num_rows"
                    )
                
                # Sort table
                if sort_by in pivot_table.columns:
                    if sort_by == "Success Rate":
                        pivot_table = pivot_table.sort_values(sort_by, ascending=False)
                    elif sort_by.endswith(" - F1 Score"):
                        # Sort F1 scores descending
                        pivot_table = pivot_table.sort_values(sort_by, ascending=False, na_position='last')
                    else:
                        pivot_table = pivot_table.sort_values(sort_by, ascending=True, na_position='last')
                else:
                    # Default sort by topic
                    pivot_table = pivot_table.sort_values('Topic', ascending=True)
                
                # Display table with row limit
                if num_rows_display == "All":
                    st.dataframe(pivot_table, use_container_width=True, height=600)
                else:
                    st.dataframe(pivot_table.head(num_rows_display), use_container_width=True, height=600)
                
                # Download button
                csv = pivot_table.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Table CSV",
                    data=csv,
                    file_name="lesson_plan_performance_comparison.csv",
                    mime="text/csv"
                )
                
                st.info(f"📊 Displaying {len(pivot_table)} lesson plans with metrics for {len(all_runs)} runs each.")
                
                # Heatmap section for correct identifications
                st.divider()
                st.subheader("🔥 Correct Detection Heatmaps")
                
                # Heatmap 1: Correct detection per lesson plan per run
                st.markdown("**Heatmap 1: Correct Detection by Lesson Plan**")
                st.markdown("Shows whether each run correctly identified the expected category for each lesson plan.")
                
                # Build heatmap data
                heatmap_data = []
                heatmap_lesson_ids = []
                heatmap_topics = []
                
                for lesson_idx, lesson_row in filtered_lesson_df.iterrows():
                    lesson_id = lesson_row['lesson_id']
                    lesson_runs_df = df[df['id'] == lesson_id] if 'id' in df.columns else pd.DataFrame()
                    
                    if lesson_runs_df.empty:
                        continue
                    
                    first_run_row = lesson_runs_df.iloc[0]
                    human_expected = first_run_row.get('Human_expected', '')
                    positive_expected, _ = parse_human_expected(human_expected)
                    
                    # Normalize expected categories
                    def normalize_category(cat: str) -> str:
                        cat = str(cat).lower().strip()
                        if '/' in cat:
                            return cat.split('/')[0]
                        if len(cat) > 1 and cat[0].isalpha() and (cat[1].isdigit() or cat[1].isalpha()):
                            return cat[0]
                        if len(cat) == 1 and cat.isalpha():
                            return cat
                        return cat
                    
                    positive_expected_norm = [normalize_category(cat) for cat in positive_expected]
                    
                    row_values = []
                    for run_name in all_runs:
                        run_row = lesson_runs_df[lesson_runs_df['run_name'] == run_name]
                        
                        if not run_row.empty:
                            run_row = run_row.iloc[0]
                            flagged_str = run_row.get('comprehensive_flagged_categories', '')
                            flagged_categories = parse_flagged_categories(flagged_str)
                            flagged_norm = [normalize_category(cat) for cat in flagged_categories]
                            
                            # Check if expected was detected
                            expected_detected = bool([cat for cat in positive_expected_norm if cat in flagged_norm])
                            row_values.append(1 if expected_detected else 0)
                        else:
                            row_values.append(0)  # N/A case
                    
                    heatmap_data.append(row_values)
                    heatmap_lesson_ids.append(lesson_id)
                    heatmap_topics.append(lesson_row['topic'])
                
                if heatmap_data:
                    # Create heatmap DataFrame
                    heatmap_df = pd.DataFrame(
                        heatmap_data,
                        index=[f"{topic} ({str(lesson_id)[:10]}...)" for topic, lesson_id in zip(heatmap_topics, heatmap_lesson_ids)],
                        columns=[clean_run_name(run_name) for run_name in all_runs]
                    )
                    
                    # Create Plotly heatmap
                    fig_heatmap = px.imshow(
                        heatmap_df,
                        labels=dict(x="Model Combination (Run)", y="Lesson Plan", color="Correct Detection"),
                        title="Correct Detection Heatmap (1 = Correct, 0 = Incorrect/Missing)",
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        text_auto=True
                    )
                    fig_heatmap.update_layout(
                        height=max(600, len(heatmap_df) * 20),
                        xaxis_title="Run (Model Combination)",
                        yaxis_title="Lesson Plan"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Heatmap 2: Summary by expected category type
                st.markdown("**Heatmap 2: Accuracy by Expected Category Type**")
                st.markdown("Shows how often each category type was correctly identified across all runs.")
                
                # Group by expected category type
                category_summary = {}
                
                for lesson_idx, lesson_row in filtered_lesson_df.iterrows():
                    lesson_id = lesson_row['lesson_id']
                    lesson_runs_df = df[df['id'] == lesson_id] if 'id' in df.columns else pd.DataFrame()
                    
                    if lesson_runs_df.empty:
                        continue
                    
                    first_run_row = lesson_runs_df.iloc[0]
                    human_expected = first_run_row.get('Human_expected', '')
                    positive_expected, _ = parse_human_expected(human_expected)
                    
                    # Normalize expected categories
                    def normalize_category(cat: str) -> str:
                        cat = str(cat).lower().strip()
                        if '/' in cat:
                            return cat.split('/')[0]
                        if len(cat) > 1 and cat[0].isalpha() and (cat[1].isdigit() or cat[1].isalpha()):
                            return cat[0]
                        if len(cat) == 1 and cat.isalpha():
                            return cat
                        return cat
                    
                    for expected_cat in positive_expected:
                        expected_cat_norm = normalize_category(expected_cat)
                        cat_key = expected_cat_norm.upper() if len(expected_cat_norm) == 1 else expected_cat.upper()
                        
                        if cat_key not in category_summary:
                            category_summary[cat_key] = {run_name: {'correct': 0, 'total': 0} for run_name in all_runs}
                        
                        # Check each run
                        for run_name in all_runs:
                            run_row = lesson_runs_df[lesson_runs_df['run_name'] == run_name]
                            
                            if not run_row.empty:
                                run_row = run_row.iloc[0]
                                flagged_str = run_row.get('comprehensive_flagged_categories', '')
                                flagged_categories = parse_flagged_categories(flagged_str)
                                flagged_norm = [normalize_category(cat) for cat in flagged_categories]
                                
                                category_summary[cat_key][run_name]['total'] += 1
                                if expected_cat_norm in flagged_norm:
                                    category_summary[cat_key][run_name]['correct'] += 1
                
                # Build category heatmap data
                if category_summary:
                    category_heatmap_data = []
                    category_names = sorted(category_summary.keys())
                    
                    for cat_name in category_names:
                        row_values = []
                        for run_name in all_runs:
                            stats = category_summary[cat_name][run_name]
                            if stats['total'] > 0:
                                accuracy = stats['correct'] / stats['total']
                                row_values.append(accuracy)
                            else:
                                row_values.append(None)  # No data
                        category_heatmap_data.append(row_values)
                    
                    # Create category heatmap DataFrame
                    category_heatmap_df = pd.DataFrame(
                        category_heatmap_data,
                        index=category_names,
                        columns=[clean_run_name(run_name) for run_name in all_runs]
                    )
                    
                    # Create Plotly heatmap
                    fig_category_heatmap = px.imshow(
                        category_heatmap_df,
                        labels=dict(x="Model Combination (Run)", y="Expected Category", color="Accuracy"),
                        title="Category Detection Accuracy Heatmap (Percentage of correct identifications)",
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        text_auto=".2f"
                    )
                    fig_category_heatmap.update_layout(
                        height=max(400, len(category_names) * 50),
                        xaxis_title="Run (Model Combination)",
                        yaxis_title="Expected Category Type"
                    )
                    st.plotly_chart(fig_category_heatmap, use_container_width=True)
                    
                    # Show summary table
                    st.markdown("**Category Detection Summary Table**")
                    summary_table_data = []
                    for cat_name in category_names:
                        for run_name in all_runs:
                            stats = category_summary[cat_name][run_name]
                            if stats['total'] > 0:
                                summary_table_data.append({
                                    'Category': cat_name,
                                    'Run': clean_run_name(run_name),
                                    'Correct': stats['correct'],
                                    'Total': stats['total'],
                                    'Accuracy': f"{stats['correct'] / stats['total'] * 100:.1f}%"
                                })
                    
                    if summary_table_data:
                        summary_table_df = pd.DataFrame(summary_table_data)
                        st.dataframe(summary_table_df, use_container_width=True)
                        
                        # Download button for summary
                        csv_summary = summary_table_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Category Summary CSV",
                            data=csv_summary,
                            file_name="category_detection_summary.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("No data available. Please ensure the dataset contains lesson plan identifiers.")
    
    if tab7 is not None:
        with tab7:
            st.header("🎯 Model Accuracy Comparison")
            st.markdown("Compare model accuracy across different runs using heatmaps and graphs.")
            
            if not has_human_expected:
                st.warning("⚠️ This tab requires the 'Human_expected' column. Please upload data that includes human expectations.")
                return
            
            # Get comparison metrics if available
            with st.spinner("Calculating accuracy metrics..."):
                comparison_df = analyze_human_expected_comparison(df)
            
            # Get all unique runs
            all_runs = sorted(df['run_name'].unique()) if 'run_name' in df.columns else []
            
            if len(all_runs) == 0:
                st.error("No runs found in the data.")
                return
            
            # Build category accuracy data
            category_accuracy_data = {}
            run_summary_data = {}
            
            # Helper function to extract full category code (e.g., "T3" -> "T3", "N7" -> "N7")
            def extract_category_code(cat: str) -> str:
                """Extract the category code, preserving numbers (e.g., T1, T3, N7)"""
                cat = str(cat).strip().upper()
                # If it's already in format like "T3", "N7", return as is
                if len(cat) >= 2 and cat[0].isalpha() and (cat[1].isdigit() or (len(cat) > 2 and cat[1:].isdigit())):
                    return cat
                # If it's like "T", "N", return as is
                if len(cat) == 1 and cat.isalpha():
                    return cat
                # If it has a slash, try to extract abbreviation from the code
                if '/' in cat:
                    # Try to find matching category in detailed categories JSON
                    try:
                        categories_json_path = Path(__file__).parent.parent / "data" / "moderation_categories.json"
                        if categories_json_path.exists():
                            with open(categories_json_path, 'r', encoding='utf-8') as f:
                                detailed_categories = json.load(f)
                            for cat_data in detailed_categories:
                                if cat_data.get('code') == cat.lower():
                                    abbr = cat_data.get('abbreviation', '').upper()
                                    if abbr:
                                        return abbr
                    except:
                        pass
                    # Fallback: extract prefix
                    return cat.split('/')[0].upper()
                return cat
            
            # Helper function to normalize for comparison (extract main letter for matching)
            def normalize_for_comparison(cat: str) -> str:
                """Normalize category to main letter for comparison"""
                cat = str(cat).lower().strip()
                if '/' in cat:
                    return cat.split('/')[0]
                if len(cat) > 1 and cat[0].isalpha() and (cat[1].isdigit() or cat[1].isalpha()):
                    return cat[0]
                if len(cat) == 1 and cat.isalpha():
                    return cat
                return cat
            
            for idx, row in df.iterrows():
                run_name = row.get('run_name', 'Unknown')
                human_expected = row.get('Human_expected', '')
                positive_expected, negative_expected = parse_human_expected(human_expected)
                
                # Get flagged categories
                flagged_str = row.get('comprehensive_flagged_categories', '')
                flagged_categories = parse_flagged_categories(flagged_str)
                
                # Initialize run summary
                if run_name not in run_summary_data:
                    run_summary_data[run_name] = {
                        'total': 0,
                        'correct': 0,
                        'incorrect': 0
                    }
                
                # Process positive expectations (should be detected)
                for expected_cat in positive_expected:
                    # Use full category code for display (T1, T3, N7, etc.)
                    cat_key = extract_category_code(expected_cat)
                    
                    if cat_key not in category_accuracy_data:
                        category_accuracy_data[cat_key] = {run_name: {'correct': 0, 'total': 0} for run_name in all_runs}
                    
                    category_accuracy_data[cat_key][run_name]['total'] += 1
                    run_summary_data[run_name]['total'] += 1
                    
                    # Check if detected (using normalized comparison)
                    expected_cat_norm = normalize_for_comparison(expected_cat)
                    flagged_norm = [normalize_for_comparison(cat) for cat in flagged_categories]
                    
                    if expected_cat_norm in flagged_norm:
                        category_accuracy_data[cat_key][run_name]['correct'] += 1
                        run_summary_data[run_name]['correct'] += 1
                    else:
                        run_summary_data[run_name]['incorrect'] += 1
                
                # Process negative expectations (should NOT be detected)
                for not_expected_cat in negative_expected:
                    # Use full category code for display (e.g., "no R" becomes "R", "no T3" becomes "T3")
                    cat_key = extract_category_code(not_expected_cat)
                    # Add prefix to distinguish from positive expectations
                    cat_key_negative = f"NO_{cat_key}"
                    
                    if cat_key_negative not in category_accuracy_data:
                        category_accuracy_data[cat_key_negative] = {run_name: {'correct': 0, 'total': 0} for run_name in all_runs}
                    
                    category_accuracy_data[cat_key_negative][run_name]['total'] += 1
                    run_summary_data[run_name]['total'] += 1
                    
                    # Check if correctly NOT detected (using normalized comparison)
                    not_expected_cat_norm = normalize_for_comparison(not_expected_cat)
                    flagged_norm = [normalize_for_comparison(cat) for cat in flagged_categories]
                    
                    if not_expected_cat_norm not in flagged_norm:
                        # Correctly NOT detected
                        category_accuracy_data[cat_key_negative][run_name]['correct'] += 1
                        run_summary_data[run_name]['correct'] += 1
                    else:
                        # Incorrectly detected (false positive)
                        run_summary_data[run_name]['incorrect'] += 1
            
            # Filter options for categories
            col1, col2 = st.columns(2)
            with col1:
                show_positive_only = st.checkbox("Show only positive expectations (hide NO_* categories)", value=False, key="model_accuracy_show_positive")
            with col2:
                show_negative_only = st.checkbox("Show only negative expectations (NO_* categories only)", value=False, key="model_accuracy_show_negative")
            
            # Filter categories based on selections
            if category_accuracy_data:
                if show_positive_only:
                    filtered_categories = {k: v for k, v in category_accuracy_data.items() if not k.startswith('NO_')}
                elif show_negative_only:
                    filtered_categories = {k: v for k, v in category_accuracy_data.items() if k.startswith('NO_')}
                else:
                    filtered_categories = category_accuracy_data
            else:
                filtered_categories = {}
            
            # Visualization options
            visualization_type = st.radio(
                "Select Visualization Type",
                options=["Heatmap - Category Accuracy", "Heatmap - Correct Detection", "Bar Chart - Overall Accuracy", "Bar Chart - Category-wise Accuracy", "Radar Chart - Category Performance"],
                horizontal=True
            )
            
            if visualization_type == "Heatmap - Category Accuracy":
                st.subheader("🔥 Category Accuracy Heatmap")
                st.markdown("Shows accuracy percentage for each category across different runs. Categories with NO_ prefix represent negative expectations (should NOT be detected).")
                
                if filtered_categories:
                    # Build heatmap data - sort categories with numeric sort for better ordering
                    def category_sort_key(cat: str):
                        """Sort categories: first letter, then number, NO_ categories last"""
                        if cat.startswith('NO_'):
                            return (1, cat)  # Put NO_ categories after
                        # Extract letter and number for proper sorting (T1, T2, T3, T10...)
                        import re
                        match = re.match(r'([A-Z])(\d+)', cat)
                        if match:
                            letter, num = match.groups()
                            return (0, letter, int(num))  # Sort by letter, then numeric
                        return (0, cat)
                    
                    category_names = sorted(filtered_categories.keys(), key=category_sort_key)
                    heatmap_data = []
                    
                    for cat_name in category_names:
                        row_values = []
                        for run_name in all_runs:
                            stats = filtered_categories[cat_name][run_name]
                            if stats['total'] > 0:
                                accuracy = stats['correct'] / stats['total']
                                row_values.append(accuracy)
                            else:
                                row_values.append(None)
                        heatmap_data.append(row_values)
                    
                    # Create DataFrame
                    accuracy_df = pd.DataFrame(
                        heatmap_data,
                        index=category_names,
                        columns=[clean_run_name(run_name) for run_name in all_runs]
                    )
                    
                    # Create heatmap
                    fig = px.imshow(
                        accuracy_df,
                        labels=dict(x="Model Combination (Run)", y="Category", color="Accuracy"),
                        title="Category Detection Accuracy Heatmap",
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        text_auto=".2f",
                        zmin=0,
                        zmax=1
                    )
                    fig.update_layout(
                        height=max(500, len(category_names) * 50),
                        xaxis_title="Run (Model Combination)",
                        yaxis_title="Expected Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.markdown("**Detailed Accuracy Data**")
                    st.dataframe(accuracy_df, use_container_width=True)
            
            elif visualization_type == "Heatmap - Correct Detection":
                st.subheader("🔥 Correct Detection Heatmap")
                st.markdown("Shows count of correct detections for each category across different runs. Categories with NO_ prefix represent negative expectations.")
                
                if filtered_categories:
                    # Build heatmap data (correct counts) - use same sorting
                    def category_sort_key(cat: str):
                        """Sort categories: first letter, then number, NO_ categories last"""
                        if cat.startswith('NO_'):
                            return (1, cat)
                        import re
                        match = re.match(r'([A-Z])(\d+)', cat)
                        if match:
                            letter, num = match.groups()
                            return (0, letter, int(num))
                        return (0, cat)
                    
                    category_names = sorted(filtered_categories.keys(), key=category_sort_key)
                    heatmap_data = []
                    
                    for cat_name in category_names:
                        row_values = []
                        for run_name in all_runs:
                            stats = filtered_categories[cat_name][run_name]
                            row_values.append(stats['correct'])
                        heatmap_data.append(row_values)
                    
                    # Create DataFrame
                    correct_df = pd.DataFrame(
                        heatmap_data,
                        index=category_names,
                        columns=[clean_run_name(run_name) for run_name in all_runs]
                    )
                    
                    # Create heatmap
                    fig = px.imshow(
                        correct_df,
                        labels=dict(x="Model Combination (Run)", y="Category", color="Correct Count"),
                        title="Correct Detection Count Heatmap",
                        color_continuous_scale='YlGnBu',
                        aspect="auto",
                        text_auto=True
                    )
                    fig.update_layout(
                        height=max(500, len(category_names) * 50),
                        xaxis_title="Run (Model Combination)",
                        yaxis_title="Expected Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.markdown("**Correct Detection Counts**")
                    st.dataframe(correct_df, use_container_width=True)
            
            elif visualization_type == "Bar Chart - Overall Accuracy":
                st.subheader("📊 Overall Accuracy by Run")
                st.markdown("Shows overall accuracy percentage for each run.")
                
                if run_summary_data:
                    run_accuracies = []
                    for run_name in all_runs:
                        stats = run_summary_data[run_name]
                        if stats['total'] > 0:
                            accuracy = stats['correct'] / stats['total']
                            run_accuracies.append({
                                'Run': clean_run_name(run_name),
                                'Accuracy %': accuracy * 100,
                                'Correct': stats['correct'],
                                'Total': stats['total']
                            })
                    
                    if run_accuracies:
                        accuracy_df = pd.DataFrame(run_accuracies)
                        accuracy_df = accuracy_df.sort_values('Accuracy %', ascending=False)
                        
                        # Bar chart
                        fig = px.bar(
                            accuracy_df,
                            x='Run',
                            y='Accuracy %',
                            title="Overall Accuracy by Model Combination",
                            color='Accuracy %',
                            color_continuous_scale='RdYlGn',
                            text='Accuracy %',
                            text_auto='.1f'
                        )
                        fig.update_layout(
                            height=500,
                            xaxis_tickangle=-45,
                            xaxis_title="Model Combination (Run)",
                            yaxis_title="Accuracy (%)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Data table
                        st.dataframe(accuracy_df, use_container_width=True)
            
            elif visualization_type == "Bar Chart - Category-wise Accuracy":
                st.subheader("📊 Category-wise Accuracy Comparison")
                st.markdown("Compare accuracy for each category across different runs. Categories with NO_ prefix represent negative expectations.")
                
                if filtered_categories:
                    # Allow selecting categories with better sorting
                    def category_sort_key(cat: str):
                        """Sort categories: first letter, then number, NO_ categories last"""
                        if cat.startswith('NO_'):
                            return (1, cat)
                        import re
                        match = re.match(r'([A-Z])(\d+)', cat)
                        if match:
                            letter, num = match.groups()
                            return (0, letter, int(num))
                        return (0, cat)
                    
                    category_names = sorted(filtered_categories.keys(), key=category_sort_key)
                    selected_categories = st.multiselect(
                        "Select categories to display (leave empty for all)",
                        options=category_names,
                        default=category_names[:min(10, len(category_names))]  # Default to first 10
                    )
                    
                    if selected_categories:
                        # Build data for bar chart
                        chart_data = []
                        for cat_name in selected_categories:
                            for run_name in all_runs:
                                stats = filtered_categories[cat_name][run_name]
                                if stats['total'] > 0:
                                    accuracy = stats['correct'] / stats['total']
                                    chart_data.append({
                                        'Category': cat_name,
                                        'Run': clean_run_name(run_name),
                                        'Accuracy %': accuracy * 100,
                                        'Correct': stats['correct'],
                                        'Total': stats['total']
                                    })
                        
                        if chart_data:
                            chart_df = pd.DataFrame(chart_data)
                            
                            # Grouped bar chart
                            fig = px.bar(
                                chart_df,
                                x='Category',
                                y='Accuracy %',
                                color='Run',
                                title="Category-wise Accuracy by Run",
                                barmode='group',
                                text='Accuracy %',
                                text_auto='.1f'
                            )
                            fig.update_layout(
                                height=600,
                                xaxis_title="Category",
                                yaxis_title="Accuracy (%)",
                                legend_title="Run"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Data table
                            st.dataframe(chart_df, use_container_width=True)
            
            elif visualization_type == "Radar Chart - Category Performance":
                st.subheader("📈 Radar Chart - Category Performance")
                st.markdown("Compare category performance across selected runs using a radar chart. Categories with NO_ prefix represent negative expectations.")
                
                if filtered_categories:
                    # Sort categories properly
                    def category_sort_key(cat: str):
                        """Sort categories: first letter, then number, NO_ categories last"""
                        if cat.startswith('NO_'):
                            return (1, cat)
                        import re
                        match = re.match(r'([A-Z])(\d+)', cat)
                        if match:
                            letter, num = match.groups()
                            return (0, letter, int(num))
                        return (0, cat)
                    
                    category_names = sorted(filtered_categories.keys(), key=category_sort_key)
                    
                    # Select runs to compare
                    selected_runs = st.multiselect(
                        "Select runs to compare (up to 10)",
                        options=all_runs,
                        default=all_runs[:min(len(all_runs), 5)],
                        format_func=clean_run_name,
                        max_selections=10
                    )
                    
                    if selected_runs:
                        # Build radar chart data
                        fig = go.Figure()
                        
                        for run_name in selected_runs:
                            run_accuracies = []
                            for cat_name in category_names:
                                stats = filtered_categories[cat_name][run_name]
                                if stats['total'] > 0:
                                    accuracy = stats['correct'] / stats['total']
                                    run_accuracies.append(accuracy)
                                else:
                                    run_accuracies.append(0)
                            
                            fig.add_trace(go.Scatterpolar(
                                r=run_accuracies,
                                theta=category_names,
                                fill='toself',
                                name=clean_run_name(run_name)
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1],
                                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    ticktext=['0%', '20%', '40%', '60%', '80%', '100%']
                                )
                            ),
                            showlegend=True,
                            title="Category Performance Radar Chart",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary table
                        summary_data = []
                        for cat_name in category_names:
                            for run_name in selected_runs:
                                stats = filtered_categories[cat_name][run_name]
                                if stats['total'] > 0:
                                    accuracy = stats['correct'] / stats['total']
                                    summary_data.append({
                                        'Category': cat_name,
                                        'Run': clean_run_name(run_name),
                                        'Accuracy %': f"{accuracy * 100:.1f}%",
                                        'Correct': stats['correct'],
                                        'Total': stats['total']
                                    })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
            
            # Overall summary statistics
            st.divider()
            st.subheader("📊 Summary Statistics")
            
            if run_summary_data:
                summary_cols = st.columns(min(5, len(all_runs)))
                for idx, run_name in enumerate(all_runs[:5]):
                    with summary_cols[idx]:
                        stats = run_summary_data[run_name]
                        if stats['total'] > 0:
                            accuracy = stats['correct'] / stats['total']
                            st.metric(
                                clean_run_name(run_name),
                                f"{accuracy * 100:.1f}%",
                                f"{stats['correct']}/{stats['total']}"
                            )
            
            # Download button
            st.divider()
            if filtered_categories:
                # Create comprehensive accuracy table with proper sorting
                def category_sort_key(cat: str):
                    """Sort categories: first letter, then number, NO_ categories last"""
                    if cat.startswith('NO_'):
                        return (1, cat)
                    import re
                    match = re.match(r'([A-Z])(\d+)', cat)
                    if match:
                        letter, num = match.groups()
                        return (0, letter, int(num))
                    return (0, cat)
                
                accuracy_table_data = []
                for cat_name in sorted(filtered_categories.keys(), key=category_sort_key):
                    for run_name in all_runs:
                        stats = filtered_categories[cat_name][run_name]
                        if stats['total'] > 0:
                            accuracy = stats['correct'] / stats['total']
                            accuracy_table_data.append({
                                'Category': cat_name,
                                'Run': clean_run_name(run_name),
                                'Correct': stats['correct'],
                                'Total': stats['total'],
                                'Accuracy %': accuracy * 100
                            })
                
                if accuracy_table_data:
                    accuracy_table_df = pd.DataFrame(accuracy_table_data)
                    csv = accuracy_table_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Accuracy Data CSV",
                        data=csv,
                        file_name="model_accuracy_comparison.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()

