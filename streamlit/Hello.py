import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Oak AI AutoEval App 👋")

st.sidebar.success("Select a tab above.")

st.markdown(
    """
        **👈 Select a tab from the sidebar** to see what it can do!
    """)
st.markdown("""
        ***If your expected changes don't appear immediately*** try clearing cache from the three dots on the top right corner of the page /developer options and reload the page!
""")

markdown_text = """


## What You Can Do With It:

### Upload Content
- This page allows you to upload data into the lesson_plans table. You can upload a csv file with a column containing your lesson plans or other educational material. 


### Build Datasets
- This page allows you to search-filter existing lesson plans using key-stage, subject info and the generation_details of the lesson_plans. You need to create a dataset to run evaluation experiments. 


### Create Tests
- You can review existing evaluation prompts and make changes to create new prompts to test various aspects of your material using auto eval. It is also possible to review exactly what will be sent to the LLM after rendering it with jinja.


### Run Auto Evaluation
- You can select evaluation prompts, choose a dataset to run them on, and start an experiment. The app will notify you when the experiment has finished and can direct you to the insights page. It also logs the status of the experiments. 


### Visualize Results
- This interface allows you to explore detailed insights and data regarding various educational experiments conducted. 


**Your feedback is invaluable to us, so don't hesitate to share your thoughts and report any issues you encounter!**
"""

st.markdown(markdown_text)