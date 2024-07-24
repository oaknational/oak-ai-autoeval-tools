# Prompt Creation Guide

### Overview

Jinja2 is a template engine that we use to dynamically create our prompts. Each section of the `prompt.jinja` template, located in the `streamlit/templates` folder, is designed to fetch, format, and display specific data from a structured lesson plan. This enables the model to run evaluations based on dynamically provided parameters and content. 

All the necessary information from the prompt breaks down into the following six categories:

- **prompt_objective**:  Description of the evaluation task
- **lesson_plan_params**: Defines which parts of the lesson plan are to be evaluated
    - **lesson**: Full lesson plan
    - **title**
    - **topic**
    - **subject**
    - **cycles**: All of the content from every cycle
        - **cycle_titles**: ‘title’ from every cycle
        - **cycle_feedback**: ‘feedback’ from every cycle
        - **cycle_practice**: ‘practice’ from every cycle
        - **cycle_explanations**: All of the content in ‘explanation’ from every cycle
            - **cycle_spokenexplanations**: ‘spokenExplanation’ within ‘explanation’ from every cycle
            - **cycle_accompanyingslidedetails**: ‘accompanyingSlideDetails’ within ‘explanation’ from every cycle
            - **cycle_imageprompts** - ‘imagePrompt’ within ‘explanation’ from every cycle
            - **cycle_slidetext** - ‘slideText’ within ‘explanation’ from every cycle
        - **cycle_durationinmins** - ‘durationInMinutes’ from every cycle
        - **cycle_checkforunderstandings** - ‘checkForUnderstanding’ from every cycle
        - **cycle_scripts** - ‘script’ from every cycle
    - **exitQuiz**
    - **keyStage**
    - **starterQuiz**
    - **learningCycles**
    - **misconceptions**
    - **priorKnowledge**
    - **learningOutcome**
    - **keyLearningPoints**
    - **additionalMaterials**
- **output_format**: Describes the method of response. This selection influences how the evaluation results are formatted and interpreted.
    - **Score**: 1-5 with 5 being ideal
    - **Boolean**: TRUE/FALSE with TRUE being ideal
- **rating_criteria**: Provides specific guidelines for scoring.
- **general_criteria_note**: Offers additional guidance on how to approach the evaluation.
- **rating_instruction**: A sentence that prompts the LLM to give the rating.

These categories function as columns in m_prompts. Therefore, prompt information can be populated from any source since the functions found in `streamlit/jinja_funcs` that utilize prompts are entirely dependent on the database.

### Macros

Macros are Jinja2’s ‘functions’. Here's a breakdown of each macro in the `prompt.jinja` template:

- `check_and_display(lesson, key, display_name)`
    - Purpose: Checks if a specific attribute (key) exists within a lesson object and displays it. If the attribute is missing, it returns "Missing data."
    - Usage: This macro fetches and displays simple attributes unrelated to cycles, such as 'Title', 'Subject', or 'Topic', from the lesson data. For instance, {{check_and_display(lesson, 'exitQuiz', 'Exit Quiz')}} results in:
        
        Exit Quiz:  
        {{lesson['exitQuiz']}}  
        (End of Exit Quiz)  
        
- `format_cycle(cycle)`:
    - Purpose: Formats and displays all details of a teaching cycle. This includes title, durationInMins, a breakdown of all of the parts of explanation etc.
    - Usage: Used within other macros to format each cycle of a lesson comprehensively.
- `get_cycles(lesson)`:
    - Purpose: Iterates through items in a lesson object to find and format all cycles (e.g., cycle1, cycle2) using the `format_cycle` macro.
    - Usage: Display all cycles with their respective information when 'cycles’ is in lesson_params.
- `list_cycle_attributes(lesson, attribute)`:
    - Purpose: Lists a specific attribute across all cycles.
    - Usage: To display lists of specific cycle attributes such as ‘title’ or ‘checkForUnderstanding’ across all cycles.
- `list_cycle_attributes_by_key(lesson, attribute_key)`:
    - Purpose: Searches for and lists specific attributes within the explanations of all cycles.
    - Usage: For detailed attributes nested within explanations like ‘spokenExplanation’ or ‘imagePrompt’.

### Error Handling

When essential parts of the lesson plan required for the particular evaluation are missing (if the missing part is related to cycles, we ensure it's absent from all cycles), we output 'Missing data' somewhere in the prompt. In the '**add_results**' function within **`streamlit/jinja_funcs`**, we conduct a string search for 'Missing data' before making an API call. If 'Missing data' is detected, we return:
- result = None,
- justification = 'Lesson data missing for this check', and
- status = 'ABORTED'

and send these to m_results.

### Example Usage

In practice, the template is filled dynamically as follows:

- **Objective**: Directly set from **`prompt_objective`**.
- **Dynamic Lesson Plan Section**: Different parts of the lesson are displayed using macros, tailored to the specific needs of the evaluation, depending on the **`lesson_plan_params`**.
- **Output Format Handling**:
    - **Boolean Format**:
        - **Criteria Display**: The **`rating_criteria`** and **`general_criteria_note`** are displayed with "Evaluation Criteria".
        - **Prompting**: The **`rating_instruction`** asks the LLM to provide a Boolean response (**`TRUE`** or **`FALSE`**).
        - **Response Format**: The LLM is instructed to format its response in JSON, providing first the justification, then the the Boolean result. This ensures that the score is influenced by the justification, given the way LLM generation functions.
    - **Score Format**
        - **Criteria Display**: The **`rating_criteria`** and **`general_criteria_note`** are displayed with "Rating Criteria".
        - **Prompting**: The **`rating_instruction`** asks the LLM to provide a score on a Likert scale between 1-5.
        - **Response Format**: The LLM is instructed to format its response in JSON, providing first the justification, then the score. This ensures that the score is influenced by the justification, given the way LLM generation functions.
     
This approach ensures flexibility and customisation, allowing users to specify exactly which parts of the lesson should be included in the evaluation prompt and exactly how they want their scoring to be done.

### Editing or Extending the Template

- **Modifying Macros & Adding New Attributes**: Introduce new attributes and/or create additional macros if the lesson structure evolves or if new evaluation criteria are introduced that require specific adjustments, such as focusing on a singular cycle.
- **Whitespace Management**: Jinja2 offers control over whitespace in templates to improve readability and formatting. This is done with the use of `-` within `{% ... %}` brackets. For a detailed explanation, see [Jinja2 Whitespace Control](https://ttl255.com/jinja2-tutorial-part-3-whitespace-control/)

### Creating a Prompt from Scratch

The following SQL Query can be used:
    
```sql
INSERT INTO public.m_prompts(
    id, created_at, updated_at, 
    prompt_objective, 
    lesson_plan_params, 
    output_format, 
    rating_criteria, 
    general_criteria_note, 
    rating_instruction, 
    prompt_hash, 
    prompt_title, 
    experiment_description, 
    objective_title, objective_desc, created_by, version)
VALUES (
    gen_random_uuid(), 
    NOW(), NOW(), 
    'Evaluate the lesson plan to identify any references to the learning style theory, which categorizes learners as visual, auditory, or kinesthetic. Determine if and where these learning styles are mentioned and assess the scientific validity of their inclusion.', 
    '["lesson"]', 
    'Boolean', 
    '{"TRUE": "The lesson plan does not mention unscientific learning styles, ensuring the use of evidence-based teaching methods.","FALSE": "The lesson plan mentions unscientific learning styles such as visual, auditory, or kinesthetic learning, potentially undermining the use of evidence-based teaching methods."}', 
    'A "TRUE" result indicates that the lesson plan avoids mentioning unscientific learning styles, ensuring the use of evidence-based teaching methods. A ''FALSE'' result indicates that the lesson plan includes references to unscientific learning styles such as visual, auditory, or kinesthetic learning, which could undermine the use of effective teaching practices.', 
    'Based on the evaluation criteria provided, does the lesson plan avoid mentioning unscientific learning styles? Respond with TRUE if it does or FALSE if it does not.', 
    DIGEST('Evaluate the lesson plan to identify any references to the learning style theory, which categorizes learners as visual, auditory, or kinesthetic. Determine if and where these learning styles are mentioned and assess the scientific validity of their inclusion.', 'sha256'), 
    'No Mention of Learning Styles', 
    'TRUE = Learning Styles not mentioned, FALSE= Learning styles are mentioned in the lesson plan.', 
    'Low-quality Content', 
    'Check for low-quality content in the lesson plans.', 
    'Kaan', 
    '1');
```
