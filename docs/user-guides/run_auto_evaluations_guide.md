# AutoEval User Documentation: Run Auto Evaluations
This page allows you to run evaluations on a dataset using selected prompts. Results are stored in the database that can then be reviewed on the Visualise Results page.

### 1. Set Up Environment
- **Clear Cache**: Use the "Clear Cache" button in the sidebar to clear the application cache. This can help resolve any issues related to cached data.

### 2. Select Prompts
- **Prompt Selection**: Use the "Select prompts" multiselect box to choose the prompts you want to run evaluations on. You can select multiple prompts.
- **Prompt Information**: The selected prompts and their descriptions will be displayed in a table for review.

### 3. Select Datasets
- **Dataset Selection**: Use the "Select datasets to run evaluation on" multiselect box to choose the datasets you want to evaluate. The available datasets are displayed with the number of lesson plans in each sample.
- **Dataset Information**: The selected datasets and the number of lessons in each will be displayed in a table for review.

### 4. Set Evaluation Parameters
- **Estimate Run Time**: The application calculates and displays an estimated run time for the evaluations based on the selected prompts and datasets. This helps you set appropriate limits to avoid long run times.
- **Set Limit**: Use the "Set a limit on the number of lesson plans per sample to evaluate" number input to limit the number of lesson plans per sample that will be evaluated. The default limit is 5, but you can adjust it based on your needs.
- **Select Model**: Use the "Select a model" dropdown to choose the LLM model (e.g., GPT-4) for the evaluation.
- **Set Temperature**: Use the "Enter temperature" number input to set the model temperature. This parameter controls the randomness of the model's output.

### 5. Running the Experiment
- **Enter Your Name**: Use the "Who is running the experiment?" dropdown to select your name from the list of available teachers. This information will be stored with the experiment details.
- **Generate Placeholders**: The application generates placeholders for the experiment name and description based on the selected parameters.
- **Experiment Information**: Use the provided form to enter the experiment name and description. The placeholders can be used as a starting point.
- **Run Evaluation**: Click the "Run evaluation" button to start the experiment. A warning will appear advising you not to close the page until the evaluation is complete.

### 6. Viewing Results
- **View Insights**: After the evaluation is complete, click the "View Insights" button to navigate to the Visualise Results page, where you can view the results of the evaluation.

## Example Workflow
1. **Clear Cache**: Click the "Clear Cache" button.
2. **Select Prompts**: Choose prompts from the "Select prompts" multiselect box.
3. **Select Datasets**: Choose datasets from the "Select datasets to run evaluation on" multiselect box.
4. **Set Limit**: Enter a limit for the number of lesson plans per sample.
5. **Select Model**: Choose the desired model (e.g., GPT-4).
6. **Set Temperature**: Set the model temperature.
7. **Enter Your Name**: Select your name from the dropdown.
8. **Enter Experiment Information**: Provide a name and description for the experiment.
9. **Run Evaluation**: Click the "Run evaluation" button.
10. **View Insights**: Click the "View Insights" button after the evaluation completes.

## Common Issues and Troubleshooting
- **No Prompts or Datasets Found**: Ensure the keyword and filters are correct and relevant to the prompts and datasets available.
- **Missing Required Fields**: Ensure all required fields (e.g., dataset title, creator's name, prompts, and datasets) are provided before running the evaluation.
- **Cache Issues**: If you encounter issues with outdated or incorrect data, use the "Clear Cache" button to reset.
