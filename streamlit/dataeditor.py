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

# Mappings for standardization
KS_MAPPINGS = {
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

SUBJECT_MAPPINGS = {
    "maths": "mathematics",
    "english": "english",
    "science": "science",
    "history": "history",
    "geography": "geography",
    "psed": "personal, social and emotional development",
    "rshe-pshe": "personal, social, health and economic education",
}


def standardize_key_stage(ks):
    """Standardizes Key Stage labels."""
    if isinstance(ks, str):
        ks = ks.strip().lower()
        return KS_MAPPINGS.get(ks, ks)
    return ks  # Return as is if not a string


def standardize_subject(subj):
    """Standardizes subject labels."""
    if isinstance(subj, str):
        subj = subj.strip().lower()
        return SUBJECT_MAPPINGS.get(subj, subj)
    return subj  # Return as is if not a string
