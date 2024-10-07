"""
constants.py

This module contains constants used throughout the application.

Classes:
    ErrorMessages: Error message constants used for exception handling.
    OptionConstants: Option-related constants for menu selections.
    ColumnLabels: Labels used for column headers in data tables.
    ExamplePrompts: Example prompt texts to help users create new prompts.

Usage:
    from constants import (ErrorMessages, OptionConstants,
        ColumnLabels, ExamplePrompts
    )

    # Example usage of ErrorMessages
    log_message("error", f"{ErrorMessages.UNEXPECTED_ERROR}: {e}")

    # Example usage of OptionConstants
    display_message(OptionConstants.SELECT_TEACHER)

    # Example usage of ColumnLabels
    df.rename(columns={df.columns[0]: ColumnLabels.NUM_LESSONS}, inplace=True)

    # Example usage of ExamplePrompts
    st.text_area("Prompt Objective", ExamplePrompts.PROMPT_OBJECTIVE)
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
    """
    Labels used for column headers in data tables.

    Attributes:
        NUM_LESSONS (str): Label for the number of lessons column.
    """
    NUM_LESSONS = "Number of Lessons"
    
    
class ExamplePrompts:
    """
    Example prompt texts used for evaluation guidance.

    Attributes:
        PROMPT_OBJECTIVE (str): Example prompt objective.
        SCORE (str): Example scoring criteria for a 1-5 Likert rating 
            scale.
        BOOL (str): Example true/false criteria for boolean evaluation.
        GENERAL_CRITERIA_SCORE (str): Example general criteria note for 
            scoring.
        GENERAL_CRITERIA_BOOL (str): Example general criteria note for 
            boolean evaluation.
        RATING_INSTRUCTION_SCORE (str): Example instructions for 
            scoring evaluation.
        RATING_INSTRUCTION_BOOL (str): Example instructions for 
        boolean evaluation.
    """
    PROMPT_OBJECTIVE = """
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


class LessonPlanParameters:
    LESSON_PARAMS = [
    "lesson",
    "title",
    "topic",
    "subject",
    "cycles",
    "cycle_titles",
    "cycle_feedback",
    "cycle_practice",
    "cycle_explanations",
    "cycle_spokenexplanations",
    "cycle_accompanyingslidedetails",
    "cycle_imageprompts",
    "cycle_slidetext",
    "cycle_durationinmins",
    "cycle_checkforunderstandings",
    "cycle_scripts",
    "exitQuiz",
    "keyStage",
    "keywords",
    "starterQuiz",
    "learningCycles",
    "misconceptions",
    "priorKnowledge",
    "learningOutcome",
    "keyLearningPoints",
    "additionalMaterials",
]

    LESSON_PARAMS_TITLES = [
    "Lesson",
    "Title",
    "Topic",
    "Subject",
    "Cycles",
    "Titles",
    "Feedback",
    "Practice Tasks",
    "Explanations",
    "Spoken Explanations",
    "Accompanying Slide Details",
    "Image Prompts",
    "Slide Text",
    "Duration in Minutes",
    "Check for Understandings",
    "Scripts",
    "Exit Quiz",
    "Key Stage",
    "Keywords",
    "Starter Quiz",
    "Learning Cycles",
    "Misconceptions",
    "Prior Knowledge",
    "Learning Outcome",
    "Key Learning Points",
    "Additional Materials",
]

    LESSON_PARAMS_PLAIN_ENG =   [
    "Whole lesson",
    "Title",
    "Topic",
    "Subject",
    "All content from all cycles",
    "All cycle titles",
    "All cycle feedback",
    "All cycle practice",
    "Entire explanations from all cycles",
    "All spoken explanations from all cycles",
    "All accompanying slide details from all cycles",
    "All image prompts from all cycles",
    "All slide text from all cycles",
    "All durations in minutes from all cycles",
    "All check for understandings from all cycles",
    "All scripts from all cycles",
    "Exit Quiz",
    "Key Stage",
    "Keywords",
    "Starter Quiz",
    "Learning cycles",
    "Misconceptions",
    "Prior knowledge",
    "Learning outcomes",
    "Key learning points",
    "Additional materials",
]
