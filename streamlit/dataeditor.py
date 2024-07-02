import pandas as pd


#these functions are used to standardize the data in the database


def standardize_key_stage(ks):
    if isinstance(ks, str):
        ks = ks.strip().lower()
    else:
        return ks  # Return as is if not a string

    ks_mappings = {
        'year 6': 'key-stage-2',
        'ks1': 'key-stage-1',
        '1': 'key-stage-1',
        '2': 'key-stage-2',
        '3': 'key-stage-3',
        '4': 'key-stage-4',
        'ks3': 'key-stage-3',
        'ks4': 'key-stage-4',
        'ks2': 'key-stage-2',
        'key stage 1': 'key-stage-1',
        'key stage 2': 'key-stage-2',
        'key stage 3': 'key-stage-3',
        'key stage 4': 'key-stage-4',
        'key stage 5': 'key-stage-5'
    }
    return ks_mappings.get(ks, ks)  # Return the mapped value or the original cleaned string

def standardize_subject(subj):
    if isinstance(subj, str):
        subj = subj.strip().lower()
    else:
        return subj  # Return as is if not a string

    subject_mappings = {
        'maths': 'mathematics',
        'english': 'english',
        'science': 'science',
        'history': 'history',
        'geography': 'geography',
        'psed': 'personal, social and emotional development',
        'rshe-pshe': 'personal, social, health and economic education'
    }
    return subject_mappings.get(subj, subj)  # Return the mapped value or the original cleaned string
