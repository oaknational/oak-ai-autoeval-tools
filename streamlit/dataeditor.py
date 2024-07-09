""" Functions used to standardize data in the database.

Functions:

- standardize_key_stage: 
Standardizes Key Stage labels.

- standardize_subject: 
Standardizes subject labels.

Usage Examples:

standardize_key_stage('KS2')
>> 'key-stage-2'

standardize_subject('Maths')
>> 'mathematics'

Note:
- If the input is not a string, the functions return it as-is.
"""

# Import the required libraries and modules
import pandas as pd


def standardize_key_stage(ks):
    """Standardizes Key Stage labels."""
    if isinstance(ks, str):
        ks = ks.strip().lower()
    else:
        return ks  # Return as is if not a string

    ks_mappings = {
        "year 6": "key-stage-2",
        "ks1": "key-stage-1",
        "1": "key-stage-1",
        "2": "key-stage-2",
        "3": "key-stage-3",
        "4": "key-stage-4",
        "ks3": "key-stage-3",
        "ks4": "key-stage-4",
        "ks2": "key-stage-2",
        "key stage 1": "key-stage-1",
        "key stage 2": "key-stage-2",
        "key stage 3": "key-stage-3",
        "key stage 4": "key-stage-4",
        "key stage 5": "key-stage-5",
    }
    # Return the mapped value or the original cleaned string
    return ks_mappings.get(ks, ks)


def standardize_subject(subj):
    """Standardizes subject labels."""
    if isinstance(subj, str):
        subj = subj.strip().lower()
    else:
        return subj  # Return as is if not a string

    subject_mappings = {
        "maths": "mathematics",
        "english": "english",
        "science": "science",
        "history": "history",
        "geography": "geography",
        "psed": "personal, social and emotional development",
        "rshe-pshe": "personal, social, health and economic education",
    }
    # Return the mapped value or the original cleaned string
    return subject_mappings.get(subj, subj)
