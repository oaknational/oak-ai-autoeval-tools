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
    
    
class ExamplePrompts:
    PROMPT_OBJECTIVE = (
        f"""
        Evaluate whether the quiz questions from the Starter Quiz and 
        Exit Quiz require specific, explicit knowledge for correct 
        answers, or if they can be answered using general knowledge or 
        educated guesses. Make sure to assess if the wording of 
        questions allows for answers to be guessed without substantial 
        knowledge of the topic. This assessment should determine the 
        effectiveness of the questions in measuring targeted learning 
        outcomes.
        
        Note: A thoughtful analysis of the Starter Quiz and Exit Quiz is 
        required. Submissions that do not demonstrate a detailed 
        examination will be disregarded.
        """
    )
    SCORE = """
        **Label for 1:** Don't Require Explicit Knowledge
        
        **Description for 1:** The questions can largely be answered using 
        general knowledge, guesses, or can be easily inferred from the 
        phrasing itself, indicating they do not effectively measure the 
        students' specific learning of the material.
        
        **Label for 5:** Require Explicit Knowledge
        
        **Description for 5:** The quiz questions require detailed, explicit 
        knowledge of the topic, ensuring that correct answers depend on 
        thorough understanding and precise information. The questions 
        are structured to prevent guessing from the phrasing alone.
    """

    BOOL = """
        **Description for TRUE:** There is an increase in challenge across 
        the learning cycles, confirming progressive learning structure.
        
        **Description for FALSE:** There is no detectable increase in 
        challenge across the learning cycles, indicating a potential 
        issue in the progressive learning structure.
    """
    
    GENERAL_CRITERIA_SCORE = """
        Ensure that ratings not only reflect the depth of knowledge 
        required but also how well the questions are designed to prevent 
        answers being guessed based on their wording.
    """
    
    GENERAL_CRITERIA_BOOL = """
        Review the complexity, depth of content, and cognitive demands 
        of each cycle to determine if there is a progression in 
        challenge.
    """
    
    RATING_INSTRUCTION_SCORE = """
        Rate the quiz questions on a scale from 1 to 5 based on their 
        need for specific, explicit knowledge and their design to 
        prevent guesswork. A score of 5 indicates that the questions 
        require precise and detailed understanding of the lesson's 
        content and are phrased to prevent guessing, while a score of 1 
        means the questions can be answered with general knowledge or 
        simple inference from the question phrasing.
    """
    
    RATING_INSTRUCTION_BOOL = """
        Provide a Boolean TRUE if an increase in challenge is detected 
        across the learning cycles, and FALSE if no such increase is 
        found.
    """


