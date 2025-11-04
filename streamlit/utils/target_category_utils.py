"""
Utility functions for analyzing target category accuracy in moderation results.
"""

import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
import ast


def parse_json_column(series: pd.Series) -> pd.Series:
    """Safely parse JSON columns, handling both string and dict types."""
    def safe_parse(x):
        if pd.isna(x) or x == '':
            return []
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(x)
                except (json.JSONDecodeError, ValueError):
                    return []
        elif isinstance(x, list):
            return x
        else:
            return []
    
    return series.apply(safe_parse)


def extract_category_code(target_category: str) -> Optional[str]:
    """
    Extract category code from target category string.
    
    Examples:
    - "t/creating-biological-weapons" -> "t"
    - "l/discriminatory-behaviour" -> "l"
    - "u/upsetting-content" -> "u"
    """
    if pd.isna(target_category) or not target_category:
        return None
    
    if '/' in target_category:
        return target_category.split('/')[0]
    
    # If it's already a code (like "u1", "t1"), extract the letter part
    if len(target_category) > 0 and target_category[0].isalpha():
        return target_category[0].lower()
    
    return None


def normalize_flagged_category(category: str) -> Optional[str]:
    """
    Normalize flagged category to match target category format.
    
    Examples:
    - "u1" -> "u"
    - "t1" -> "t"
    - "l1" -> "l"
    - "e1" -> "e"
    """
    if pd.isna(category) or not category:
        return None
    
    category_str = str(category).lower()
    
    # If it's already a single letter, return it
    if len(category_str) == 1 and category_str.isalpha():
        return category_str
    
    # Extract first letter if it's like "u1", "t1", etc.
    if len(category_str) > 0 and category_str[0].isalpha():
        return category_str[0].lower()
    
    return None


def is_target_category_met(row: pd.Series) -> Tuple[bool, Optional[str], List[str]]:
    """
    Check if the target category was correctly identified in the flagged categories.
    
    Returns:
        Tuple of (is_met, target_category_code, flagged_categories)
    """
    target_category = row.get('target_category', None)
    
    if pd.isna(target_category):
        return False, None, []
    
    target_code = extract_category_code(str(target_category))
    
    if not target_code:
        return False, None, []
    
    # Get flagged categories from either comprehensive or moderation columns
    # Priority: comprehensive_flagged_categories takes precedence (more detailed stage)
    flagged_categories = []
    
    if 'comprehensive_flagged_categories' in row.index and pd.notna(row['comprehensive_flagged_categories']):
        try:
            flagged = json.loads(row['comprehensive_flagged_categories'])
            if isinstance(flagged, list):
                flagged_categories = [normalize_flagged_category(cat) for cat in flagged if cat]
        except (json.JSONDecodeError, ValueError):
            pass
    elif 'moderation_flagged_categories' in row.index and pd.notna(row['moderation_flagged_categories']):
        # Fallback to moderation_flagged_categories if comprehensive is not available
        try:
            flagged = json.loads(row['moderation_flagged_categories'])
            if isinstance(flagged, list):
                flagged_categories = [normalize_flagged_category(cat) for cat in flagged if cat]
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Check if target category code is in flagged categories
    is_met = target_code in flagged_categories
    
    return is_met, target_code, [c for c in flagged_categories if c]


def calculate_target_category_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics about target category accuracy.
    
    Returns:
        Dictionary with accuracy metrics
    """
    if 'target_category' not in df.columns:
        return {}
    
    stats = {
        'total_lessons': len(df),
        'lessons_with_target': 0,
        'target_correctly_identified': 0,
        'target_missed': 0,
        'false_positives': 0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'target_category_distribution': {},
        'by_target_category': {}
    }
    
    # Analyze each row
    for idx, row in df.iterrows():
        is_met, target_code, flagged_codes = is_target_category_met(row)
        
        if target_code:
            stats['lessons_with_target'] += 1
            
            # Update target category distribution
            if target_code not in stats['target_category_distribution']:
                stats['target_category_distribution'][target_code] = {
                    'count': 0,
                    'correctly_identified': 0,
                    'missed': 0
                }
            
            stats['target_category_distribution'][target_code]['count'] += 1
            
            if is_met:
                stats['target_correctly_identified'] += 1
                stats['target_category_distribution'][target_code]['correctly_identified'] += 1
            else:
                stats['target_missed'] += 1
                stats['target_category_distribution'][target_code]['missed'] += 1
            
            # Count false positives (flagged categories that don't match target)
            # Count each incorrectly flagged category, not just the number of lessons
            false_positive_count = sum(1 for code in flagged_codes if code != target_code)
            stats['false_positives'] += false_positive_count
    
    # Calculate metrics
    if stats['lessons_with_target'] > 0:
        stats['accuracy'] = stats['target_correctly_identified'] / stats['lessons_with_target']
        stats['recall'] = stats['target_correctly_identified'] / stats['lessons_with_target']
    
    if stats['target_correctly_identified'] + stats['false_positives'] > 0:
        stats['precision'] = stats['target_correctly_identified'] / (stats['target_correctly_identified'] + stats['false_positives'])
    
    if stats['precision'] + stats['recall'] > 0:
        stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
    
    # Calculate by-category statistics
    for target_code, cat_stats in stats['target_category_distribution'].items():
        total = cat_stats['count']
        correct = cat_stats['correctly_identified']
        missed = cat_stats['missed']
        
        stats['by_target_category'][target_code] = {
            'total': total,
            'correctly_identified': correct,
            'missed': missed,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    return stats


def get_target_category_name(code: str) -> str:
    """Get human-readable name for category code."""
    category_names = {
        'l': 'Language',
        'u': 'Upsetting/Sensitive',
        'v': 'Violence',
        's': 'Sexual',
        'p': 'Physical',
        't': 'Toxic',
        'r': 'Recent Events',
        'n': 'News',
        'e': 'RSHE'
    }
    return category_names.get(code.lower(), code.upper())

