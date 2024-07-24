### AutoEval User Guide

# User Documentation: Upload Content

Welcome to the "Upload Content" screen of our application! This guide will help you understand how to upload and process your content data effectively.

## Overview

This screen allows you to upload a CSV file containing your lesson plan data, process the data to ensure it is in the correct format, and insert it into our database. Follow the steps below for a smooth experience.

## Steps to Upload and Process Data

### 1. Preparing Your CSV File

Ensure your CSV file adheres to the following format:
- **Columns**: Include columns with data that need to be converted to JSON format.
- **JSON Data**: If any column contains JSON data, ensure it is correctly formatted.
- **String Data**: If the data is plain text, it will be converted to a JSON object with the text stored under the key `text`.

### 2. Uploading Your CSV File

1. **Open the File Uploader**: Click on the "Upload your CSV file" button.
2. **Select Your File**: Choose the CSV file from your local machine.

### 3. Viewing and Selecting Data

Once the file is uploaded, the data will be displayed in a table. You can then:
1. **View the Data**: Inspect the uploaded data to ensure it is correctly read.
2. **Select the Column for Conversion**: Choose the column that contains JSON data or data that needs to be converted to JSON.

### 4. Entering Generation Details

Provide a unique identifier or description for your dataset. This helps differentiate your data from other entries in the lesson plans table when creating a dataset.

### 5. Inserting Data into the Database

After selecting the column and entering the generation details:
1. **Assign Unique IDs**: Each entry will be assigned a unique ID.
2. **Insert Data**: Click on the "Insert Data into Database" button to insert the processed data into the lesson plans table.
3. **Confirmation**: A success message will appear once the data is successfully inserted.

## Detailed Instructions

### Example Data

**JSON Data Example**:
```json
{
  "name": "Lesson 1",
  "content": "This is a lesson plan."
}
```

**String Data Example**:
```text
This is a plain text lesson plan.
```

After conversion, the plain text will be stored as:
```json
{
  "text": "This is a plain text lesson plan."
}
```

### How Your Data Will Be Processed

1. **Upload the CSV File**: Use the file uploader to select your CSV file.
2. **Select the Column**: Choose the column with JSON data or data to be converted to JSON.
3. **Generation Details**: Enter a unique identifier or description for your dataset.
4. **Insert into Database**: Data will be inserted into the lesson plans table with each entry having a unique ID and the provided generation details.

### Handling Missing Values

Rows with missing (NaN) values in the selected column will be skipped during the conversion process to ensure data integrity.

### Manual Data Insertion

Alternatively, you can manually insert data into the lesson plans table using SQL.

## Troubleshooting

### Common Issues

- **Incorrect File Format**: Ensure your CSV file follows the specified format.
- **Invalid JSON Data**: Check that JSON data is correctly formatted.
- **Missing Generation Details**: Provide a unique identifier or description for your dataset.

### Getting Help

If you encounter any issues or have questions about using the "Upload Content" screen, please reach out to our support team.

## Conclusion

By following these steps, you can easily upload, process, and insert your lesson plan data into our database. Thank you for using our application!

---

# User Documentation: Build Datasets

Welcome to the "Build Datasets" screen of the AutoEval app! This guide will help you understand how to create and manage datasets by selecting and saving subsets of lesson plans for evaluation.

## Overview

This screen allows you to:
- Provide inputs for dataset title, creator's name, and keyword search for lesson plans.
- Display the retrieved lesson plans.
- Save the selected lesson plans to a new or existing sample.
- Clear the cache.

## Steps to Build and Manage Datasets

### 1. Setting Up Your Dataset

#### Enter Dataset Title

1. **Dataset Title**: In the text input box labeled "Enter a dataset title for the Eval UI (e.g. history_ks2)", provide a descriptive title for your dataset. This title will help you identify your dataset in the evaluation UI.

#### Enter Your Name

2. **Creator's Name**: In the text input box labeled "Enter your name", enter your name. This information will be stored as part of the dataset details.

### 2. Filtering Lesson Plans

#### Enter Keyword for Generation Details

3. **Keyword Search**: In the text input box labeled "Enter keyword for generation details", provide a keyword to filter lesson plans based on their generation details. This will help narrow down the lesson plans to those relevant to your dataset.

4. **Retrieve Lesson Plans**: Click the "Get Lesson Plans" button to search for lesson plans that match the provided keyword. The matching lesson plans will be displayed in a table.

### 3. Viewing and Selecting Lesson Plans

#### Displaying Lesson Plans

- After clicking the "Get Lesson Plans" button, the retrieved lesson plans will be displayed in a table. You can review the lesson plans to ensure they meet your criteria.

### 4. Saving the Dataset

#### Save Sample with Selected Lesson Plans

5. **Save Sample**: Click the "Save Sample with Selected Lesson Plans" button to save the selected lesson plans to a new or existing sample. The following actions will be performed:
    - **Creating a New Sample**: If you have provided a dataset title and your name, a new sample will be created.
    - **Linking Lesson Plans**: The selected lesson plans will be linked to the new sample.

    **Note**: Ensure that all required fields (dataset title and creator's name) are filled before saving.

### 5. Clearing the Cache

#### Clear Cache

6. **Clear Cache**: Use the "Clear Cache" button in the sidebar to clear the application cache. This can help resolve any issues related to cached data.

### Example Workflow

1. **Enter Dataset Title**: Enter "history_ks2" as the dataset title.
2. **Enter Your Name**: Enter "John Doe".
3. **Enter Keyword**: Enter "history" as the keyword.
4. **Retrieve Lesson Plans**: Click the "Get Lesson Plans" button to display lesson plans related to "history".
5. **Save Sample**: Click the "Save Sample with Selected Lesson Plans" button to save the retrieved lesson plans to a new sample.

## Common Issues and Troubleshooting

### No Lesson Plans Found

- **Issue**: No lesson plans are found with the given filters.
- **Solution**: Ensure the keyword is correct and relevant to the generation details of the lesson plans.

### Missing Required Fields

- **Issue**: Trying to save the sample without filling in all required fields.
- **Solution**: Ensure both the dataset title and creator's name are provided before saving.

### Cache Issues

- **Issue**: Encountering issues with outdated or incorrect data.
- **Solution**: Use the "Clear Cache" button to clear the application cache.

## Getting Help

If you encounter any issues or have questions about using the "Build Datasets" screen, please reach out to our support team.

## Conclusion

By following these steps, you can easily create and manage datasets by selecting and saving lesson plans for evaluation. Thank you for using the AutoEval app!

---

# User Documentation: Run Auto Evaluations

Welcome to the "Run Auto Evaluations" screen of the AutoEval app! This guide will help you understand how to run evaluations on a dataset using selected prompts and manage the results effectively.

## Overview

This screen allows you to:
- Run evaluations on a dataset using selected prompts.
- Store the results in the database.
- View the results on the Visualise Results page.

## Steps to Run Auto Evaluations

### 1. Setting Up the Environment

#### Clear Cache

1. **Clear Cache**: Use the "Clear Cache" button in the sidebar to clear the application cache. This can help resolve any issues related to cached data.

### 2. Configuring Evaluation Parameters

#### Test Selection

2. **Select Prompts**:
    - **Prompt Selection**: Use the "Select prompts" multiselect box to choose the prompts you want to run evaluations on. You can select multiple prompts.
    - **Prompt Information**: The selected prompts and their descriptions will be displayed in a table for review.

#### Dataset Selection

3. **Select Datasets**:
    - **Dataset Selection**: Use the "Select datasets to run evaluation on" multiselect box to choose the datasets you want to evaluate. The available datasets are displayed with the number of lesson plans in each sample.
    - **Dataset Information**: The selected datasets and the number of lessons in each will be displayed in a table for review.

### 3. Setting Limits and Models

#### Time Estimates

4. **Estimate Run Time**: The application calculates and displays an estimated run time for the evaluations based on the selected prompts and datasets. This helps you set appropriate limits to avoid long run times.

#### Set Limit on Lesson Plans

5. **Set Limit**: Use the "Set a limit on the number of lesson plans per sample to evaluate" number input to limit the number of lesson plans per sample that will be evaluated. The default limit is 5, but you can adjust it based on your needs.

#### Select Model and Temperature

6. **Select Model**: Use the "Select a model" dropdown to choose the LLM model (e.g., GPT-4) for the evaluation.
7. **Set Temperature**: Use the "Enter temperature" number input to set the model temperature. This parameter controls the randomness of the model's output.

#### Specify Experiment Runner

8. **Enter Your Name**: Use the "Who is running the experiment?" dropdown to select your name from the list of available teachers. This information will be stored with the experiment details.

### 4. Running the Experiment

#### Generate Placeholders

9. **Generate Placeholders**: The application generates placeholders for the experiment name and description based on the selected parameters.

#### Enter Experiment Information

10. **Experiment Information**: Use the provided form to enter the experiment name and description. The placeholders can be used as a starting point.

#### Run Evaluation

11. **Run Evaluation**: Click the "Run evaluation" button to start the experiment. A warning will appear advising you not to close the page until the evaluation is complete.

### 5. Viewing Results

#### View Insights

12. **View Insights**: After the evaluation is complete, click the "View Insights" button to navigate to the Visualise Results page, where you can view the results of the evaluation.

### Example Workflow

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

### No Prompts or Datasets Found

- **Issue**: No prompts or datasets are found with the given filters.
- **Solution**: Ensure the keyword and filters are correct and relevant to the prompts and datasets available.

### Missing Required Fields

- **Issue**: Trying to run the evaluation without filling in all required fields.
- **Solution**: Ensure all required fields (e.g., dataset title, creator's name, prompts, and datasets) are provided before running the evaluation.

### Cache Issues

- **Issue**: Encountering issues with outdated or incorrect data.
- **Solution**: Use the "Clear Cache" button to clear the application cache.

## Getting Help

If you encounter any issues or have questions about using the "Run Auto Evaluations" screen, please reach out to our support team.

## Conclusion

By following these steps, you can easily run evaluations on your datasets using selected prompts and manage the results effectively. Thank you for using the AutoEval app!

---

# User Documentation: Visualise Results

Welcome to the "Visualise Results" screen of the AutoEval app! This guide will help you understand how to use this screen to visualize and analyze the results of your experiments.

## Overview

This screen allows you to:
- Select and filter experiments to visualize results.
- Apply various filters to view specific insights.
- View detailed experiment data and metrics.
- Clear the cache to ensure you are working with the most recent data.

## Steps to Visualize Results

### 1. Setting Up the Environment

#### Clear Cache

1. **Clear Cache**: Use the "Clear Cache" button in the sidebar to clear the application cache. This ensures that you are working with the most up-to-date data.

### 2. Selecting an Experiment

#### Experiment Selection

2. **Select Experiment**:
    - **Dropdown Selection**: Use the "Select Experiment" dropdown to choose an experiment to view. The dropdown includes experiments with their run dates and teacher names.
    - **Note**: The run date is in the format YYYY-MM-DD.

### 3. Viewing Experiment Data

#### Fetch and Display Data

3. **View Data**: Once an experiment is selected, the screen will display data related to that experiment, including key stages, subjects, prompts, and results.

### 4. Applying Filters

#### Filter Options

4. **Filter Experiment Data**: Use the available filters to narrow down the data:
    - **Teacher Filter**: Select one or more teachers to filter the experiments.
    - **Prompt Filter**: Select one or more prompts to filter the experiments.
    - **Sample Filter**: Select one or more samples to filter the experiments.

5. **Key Stage and Subject Filters**: Use the multiselect boxes under "Key Stage and Subject Filters" to select key stages and subjects to further refine the data.

### 5. Selecting Outcome Type and Filtering Results

#### Outcome Selection

6. **Outcome Type**: Use the "Select Outcome Type" dropdown to choose the type of outcome you are interested in (e.g., Score, Boolean).
    - **Filter by Result Outcome**: Based on the outcome type, filter the results further to view specific outcomes.

### 6. Viewing Detailed Insights

#### Detailed Data and Metrics

7. **View Detailed Data**: The filtered data will be displayed in a table. This includes detailed metrics such as the number of lesson plans, evaluator model, and success ratio.
    - **Pie Charts**: View pie charts displaying the distribution of lesson plans by key stage and subject.
    - **Spider Chart**: View a spider chart showing the average success rate by prompt title and sample title.

### 7. Viewing Justification and Lesson Plan Details

#### Justification Lookup

8. **Justification Lookup**: Use the sidebar to enter a Result ID to view the justification for the selected run.
    - **Lesson Plan Details**: View detailed information about the relevant lesson plan parts and their justification.

### Example Workflow

1. **Clear Cache**: Click the "Clear Cache" button in the sidebar.
2. **Select Experiment**: Choose an experiment from the "Select Experiment" dropdown.
3. **Apply Filters**: Use the filters to narrow down the data by teacher, prompt, and sample.
4. **Select Outcome Type**: Choose the outcome type and filter by specific result outcomes.
5. **View Insights**: Review the pie charts and spider charts for detailed insights.
6. **Justification Lookup**: Enter a Result ID in the sidebar to view detailed justification and lesson plan parts.

## Common Issues and Troubleshooting

### No Data Found

- **Issue**: No data is displayed after selecting an experiment or applying filters.
- **Solution**: Ensure that you have selected the correct experiment and filters. Clear the cache if necessary.

### Cache Issues

- **Issue**: Encountering issues with outdated or incorrect data.
- **Solution**: Use the "Clear Cache" button in the sidebar to clear the application cache.

## Getting Help

If you encounter any issues or have questions about using the "Visualise Results" screen, please reach out to our support team.

## Conclusion

By following these steps, you can effectively visualize and analyze the results of your experiments using the "Visualise Results" screen. Thank you for using the AutoEval app!

---

