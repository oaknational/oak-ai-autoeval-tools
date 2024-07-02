import streamlit as st
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv
from jinja_funcs import *
import json
import uuid
import os
import numpy as np

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Upload Content", page_icon="⬆️")
st.markdown("# ⬆️ Upload Content")

def get_db_connection():
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    return conn

def insert_into_database(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO public.lesson_plans (id, json, generation_details)
    VALUES %s;
    """
    
    values = list(data.apply(lambda row: (row['id'], json.dumps(row['json']), row['generation_details']), axis=1))
    
    execute_values(cursor, insert_query, values)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return "Data inserted successfully!"

def convert_to_json(text):
    if pd.isna(text):
        return None
    try:
        json_data = json.loads(text)
    except json.JSONDecodeError:
        json_data = {"text": text}
    except TypeError as e:
        st.error(f"TypeError: {e} - Value: {text}")
        json_data = {"text": str(text)}
    return json_data

st.header('CSV Upload and Process')
st.write("### Instructions for Uploading Data")

st.write("""
1. **CSV File Format**: The CSV file should contain columns with data to be converted to JSON format.
2. **JSON Data**: If you have JSON data, ensure it is correctly formatted in the respective column.
3. **String Data**: If the data is not in JSON format (e.g., plain text), it will be converted to a JSON object with the text stored under the key `text`.
""")

st.write("### Example")

st.write("**JSON Data**:")
st.code('''{
  "name": "Lesson 1",
  "content": "This is a lesson plan."
}''', language='json')

st.write("**String Data**:")
st.code("This is a plain text lesson plan.", language='text')

st.write("After conversion, it will be stored as:")
st.code('''{
  "text": "This is a plain text lesson plan."
}''', language='json')

st.write("### How Your Data Will Be Processed")

st.write("""
1. **Upload the CSV File**: Use the file uploader below to select your CSV file.
2. **Select the Column**: Choose the column that contains JSON data or data to be converted to JSON.
3. **Generation Details**: Provide a unique identifier or description for your dataset. This will allow you to create a dataset from the uploads. 
4. **Insert into Database**: The data will be inserted into the lesson plans table with each entry having a unique ID and the provided generation details.
""")

st.write("**Note**: Rows with missing (NaN) values in the selected column will be skipped during the conversion process.")

st.write('Alternatively you can insert your data into the lesson plans table manually using SQL.')
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.dataframe(data)
    
    column_to_convert = st.selectbox("Select the column which contains content in JSON format or to be converted to JSON", data.columns)
    
    data[column_to_convert] = data[column_to_convert].astype(str)
    data['json'] = data[column_to_convert].apply(convert_to_json)
    
    # Remove rows with None in 'json' column
    data = data[data['json'].notna()]
    
    st.write('Enter generation details for all rows.')
    generation_details = st.text_input("This will help you differentiate your data from other entries in the lesson plans table when creating a dataset.")
    
    data['id'] = [str(uuid.uuid4()) for _ in range(len(data))]
    data['generation_details'] = generation_details
    
    if st.button('Insert Data into Database'):
        insert_into_database(data[['id', 'json', 'generation_details']])
        st.success("Data inserted successfully!")
