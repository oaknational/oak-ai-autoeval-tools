# AutoEval User Documentation: Create Prompt Tests
This page allows you to create new prompt tests that can be selected when running experiments. Users can create new prompts from scratch with guidance or create new versions of existing prompts.

## Create a New Prompt
- **Dropdown Selection**: Use the dropdown menu to select "Create a new prompt".
  
### Enter Prompt Details
- **Prompt Title**: Enter a unique title for your prompt.
- **Prompt Objective**: Describe what you want the Language Learning Model (LLM) to check for. You can refer to the example provided in the expander for guidance.
- **Lesson Plan Parameters**: Select the relevant parts of the lesson plan that you want to evaluate from the provided multiselect options.

### Select Output Format
- **Output Format Selection**: Choose between "Score" (for a Likert scale rating) or "Boolean" (for a TRUE/FALSE evaluation).
  
### Provide Rating Criteria
Once output format is chosen, further details about the evaluation can be provided.
- **Score Format**: If "Score" is selected, provide labels and descriptions for the ideal (5) and worst (1) scores.
- **Boolean Format**: If "Boolean" is selected, provide descriptions for TRUE (ideal) and FALSE outputs.
- **Examples**: Refer to the example criteria provided in the expander for guidance.
- **General Criteria Note**: Provide additional instructions or criteria you want the LLM to focus on.
- **Rating Instruction**: Provide specific instructions for the LLM on how to perform the evaluation.

### Provide Further Prompt Details
- **Prompt Group**: Select the group name that the prompt belongs to. Alternatively, you can select "New Group" and specify a name for a new prompt group and a description specifying the focus of the evaluation.
- **Teacher Selection**: Choose the name of the teacher creating the prompt.

### View and Save Prompt
- **View Your Prompt**: Click the "View Your Prompt" button to see a simplified version of the prompt you have created.
- **Save New Prompt**: Click the "Save New Prompt" button to save the new prompt to the database. Ensure the prompt title is unique to avoid errors.

## Modify an Existing Prompt
- **Dropdown Selection**: Use the dropdown menu to select "Modify an existing prompt".
- **Select Prompt Title**: From the dropdown box that appears, select the title of an existing prompt to modify.

### View Prompt Details
- **Table View**: View key details of the selected prompt, including creation date, title, objective, output format, created by, and version.
- **Full Prompt**: Expand the "View Full Prompt" section to see a detailed view of the prompt.

### Modify Prompt Details
- **Prompt Title**: The title is non-editable and displayed for reference.
- **Prompt Objective**: Update the objective of the prompt.
- **Lesson Plan Parameters**: The lesson plan parameters are non-editable and displayed for reference.
- **Output Format**: Choose the output format (Score or Boolean). If this changes, you will need to specify new rating criteria, a new general criteria note, and new rating instructions.
- **Rating Criteria**: Update the rating criteria based on the selected output format.
- **General Criteria Note**: Update the general criteria note as needed.
- **Rating Instruction**: Update the rating instruction as needed.
- **Prompt Group**: The prompt group is non-editable and displayed for reference.
- **Teacher Selection**: Choose the name of the teacher modifying the prompt.

### View and Save Modified Prompt
- **View Your Prompt**: Click the "View Your Prompt" button to see the updated prompt details.
- **Save Prompt**: Click the "Save Prompt" button to save the modified prompt to the database. The modified prompt is saved as a new prompt with the version number incremented by 1.

## Common Issues and Troubleshooting
- **Duplicate Prompt Title**: Ensure the prompt title is unique to avoid errors when saving a new prompt.
- **Cache Issues**: If encountering issues with outdated or incorrect data, use the "Clear Cache" button to reset.
