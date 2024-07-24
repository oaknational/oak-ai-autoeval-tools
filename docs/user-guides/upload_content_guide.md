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
