"""
constants.py

This module contains constants used throughout the application.

Classes:
    ErrorMessages: Error message constants used for exception handling.
    OptionConstants: Option-related constants for menu selections.

Usage:
    from constants import ErrorMessages, OptionConstants

    # Example usage of ErrorMessages
    log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")

    # Example usage of OptionConstants
    display_message(OptionConstants.SELECT_TEACHER)
"""

class ErrorMessages:
    """
    Error message constants used for exception handling.

    Attributes:
        UNEXPECTED_ERROR (str): A generic error message for unexpected errors.
    """
    UNEXPECTED_ERROR = "An unexpected error occurred"


class OptionConstants:
    """
    Option-related constants for menu selections.

    Attributes:
        SELECT_TEACHER (str): Message to prompt selection of a teacher.
    """
    SELECT_TEACHER = "Select a teacher"


class ColumnLabels:
    NUM_LESSONS = "Number of Lessons"
