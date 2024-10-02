import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
import json
import re
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import tempfile  # For handling temporary files
import pandas as pd
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pyvis.network import Network


# Define Node, Relationship, and GraphDocument classes
@dataclass
class Node:
    id: str
    type: str
    properties: Dict

@dataclass
class Relationship:
    source: str  # Node ID
    target: str  # Node ID
    type: str
    properties: Dict

@dataclass
class GraphDocument:
    nodes: List[Node]
    relationships: List[Relationship]


def sanitize_llm_response(response):
    response = re.sub(r"(\}\s*\{)", r"}, {", response)
    response = re.sub(r'(?<=[\}"])(\s*\{)', r', {', response)
    response = response.replace("```json", "").replace("```", "").strip()
    return response

def extract_json_from_response(response):
    if isinstance(response, dict):
        return response
    sanitized_response = sanitize_llm_response(response)
    try:
        return json.loads(sanitized_response)
    except ValueError as e:
        st.error(f"Error parsing LLM response: {e}")
        return None

def parse_llm_response(response, allowed_nodes, allowed_relationships, strict_mode):
    json_content = extract_json_from_response(response)
    if not json_content:
        return None

    data = json_content if isinstance(json_content, dict) else json.loads(json_content)
    nodes = []
    node_ids = {}

    for node in data.get('nodes', []):
        node_id = node.get('id')
        node_type = node.get('type')
        if strict_mode and node_type not in allowed_nodes:
            continue
        nodes.append(node)
        node_ids[node_id] = node_id

    relationships = []
    for rel in data.get('relationships', []):
        source_id = rel.get('source')
        target_id = rel.get('target')
        rel_type = rel.get('type')
        if strict_mode and rel_type not in allowed_relationships:
            continue
        if source_id in node_ids and target_id in node_ids:
            relationships.append(rel)
        else:
            st.warning(f"Invalid relationship: {source_id} -> {target_id}, nodes not found.")

    return GraphDocument(nodes=nodes, relationships=relationships)

# Function to generate LLM response
def generate_response(prompt, llm_model, llm_model_temp, top_p=1, timeout=150):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=timeout)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm_model_temp,
        seed=42,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    message = response.choices[0].message.content
    return message

# Function to create a custom prompt for the LLM
def create_custom_prompt(document_content, allowed_nodes, allowed_relationships, strict_mode):
    allowed_nodes_str = ', '.join(f'"{node}"' for node in allowed_nodes)
    allowed_relationships_str = ', '.join(f'"{rel}"' for rel in allowed_relationships)
    strict_mode_str = 'True' if strict_mode else 'False'
    prompt = f"""
    You are an expert in extracting knowledge graph data from curriculum content. Please extract the key elements in the form of nodes and relationships, ensuring that all relationships refer to the correct node IDs.

    **Instructions:**

    1. Each node must have a unique `id`, `type`, and `properties` (including at least a `label`).
    2. Each relationship must have a `source` (the node's unique `id`), `target` (another node's unique `id`), and a `type` (relationship type).
    3. Ensure that all node IDs are referenced properly in relationships.
    4. Only use the following allowed node types: {allowed_nodes_str}
    5. Only use the following allowed relationship types: {allowed_relationships_str}
    6. If there is content that does not fit the allowed nodes or relationships, you can skip it.
    7. This is for teachers to use to plan their lessons for the semester so you don't need to include every detail, just the most important elements of the curriculum.
    8. Science should be a Subject while Physics, Chemistry, and Biology should be SubjectDiscipline.
    9. Output format should be JSON.

    **Example Output:**


    {{
        "nodes": [
            {{"id": "Subject_1", "type": "Subject", "properties": {{"label": "Science"}}}},
            {{"id": "SubjectDiscipline_1", "type": "SubjectDiscipline", "properties": {{"label": "Physics"}}}}
        ],
        "relationships": [
            {{"source": "Subject_1", "target": "SubjectDiscipline_1", "type": "INCLUDES"}}
        ]
    }}

    **Content to Analyze:**
    {document_content}
    """
    return prompt

def extract_nodes(graph_document):
    return [
        {"id": node.get('id'), "label": node['properties'].get('label', node.get('id')), "category": node.get('type')}
        for node in graph_document.nodes
    ]

def extract_relationships(graph_document):
    return [
        {"source": rel.get('source'), "target": rel.get('target'), "type": rel.get('type')}
        for rel in graph_document.relationships
    ]


# Streamlit app setup
st.title("National Curriculum Knowledge Graph Generator")

# Upload file input
uploaded_file = st.file_uploader("Upload Curriculum Content File", type=["txt"])


if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")

    # Create a temporary file to store uploaded content for TextLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(content.encode("utf-8"))
        temp_file_path = temp_file.name

    # Use the temporary file path with TextLoader
    loader = TextLoader(file_path=temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    documents = text_splitter.split_documents(documents=docs)

    allowed_nodes = [
        'KeyStage', 'Subject', 'Topic', 'Subtopic', 'Concept', 
        'Lesson', 'LearningObjective', 'LearningOutcome', 'Skill',   
        'Prerequisite', 'Equipment', 'Activity', 'YearGroup'
    ]

    allowed_relationships = [
        'INCLUDES', 'TEACHES', 'REQUIRES_PREREQUISITE', 
        'USES_EQUIPMENT',  'PROGRESSES_TO',  'HAS_LESSON', 
        'HAS_LEARNING_OBJECTIVE', 'HAS_LEARNING_OUTCOME', 
    ]

    # Initialize a NetworkX graph and a PyVis network for visualization
    G = nx.Graph()
    net = Network(notebook=True)

    # Button to start extraction
    if st.button("Start Extraction"):
        with st.spinner("Generating Knowledge Graph..."):
            strict_mode = True
            json_output = {"nodes": [], "edges": []}
            progress_bar = st.progress(0)
            total_documents = len(documents)
            st.write(f"**Total Documents to Process: {total_documents}**")

            # Initialize empty DataFrames to store nodes and edges for the table
            node_df = pd.DataFrame(columns=["id", "label", "category"])
            edge_df = pd.DataFrame(columns=["source", "target", "type"])

            # Create placeholders for the table (to update dynamically)
            node_table_placeholder = st.empty()
            edge_table_placeholder = st.empty()

            for idx, document in enumerate(documents):
                # Access document content correctly using document.page_content
                prompt = create_custom_prompt(document.page_content, allowed_nodes, allowed_relationships, strict_mode)
                response = generate_response(prompt, llm_model='gpt-4o-mini', llm_model_temp=0.2)

                # Parse the response
                graph_document = parse_llm_response(response, allowed_nodes, allowed_relationships, strict_mode)
                if graph_document is None:
                    st.warning("Failed to parse LLM response.")
                    continue

                # Extract nodes and relationships
                new_nodes = extract_nodes(graph_document)
                new_edges = extract_relationships(graph_document)

                # Add new nodes and edges to the json_output
                json_output["nodes"].extend(new_nodes)
                json_output["edges"].extend(new_edges)

                # Convert the new nodes and edges to DataFrames
                new_node_df = pd.DataFrame(new_nodes)
                new_edge_df = pd.DataFrame(new_edges)

                # Append the new data to the existing DataFrames
                node_df = pd.concat([node_df, new_node_df], ignore_index=True)
                edge_df = pd.concat([edge_df, new_edge_df], ignore_index=True)

                # Add nodes and edges to the NetworkX graph and PyVis graph
                for node in new_nodes:
                    node_id = node['id']
                    G.add_node(node_id, label=node['label'], category=node.get('category', 'Unknown'))
                    net.add_node(node_id, label=node['label'], title=f"{node.get('category', 'Unknown')}", physics=True)
                    
                for edge in new_edges:
                    G.add_edge(edge['source'], edge['target'], relationship=edge['type'])
                    net.add_edge(edge['source'], edge['target'], title=edge['type'])

                # Update the progress bar
                progress_bar.progress((idx + 1) / total_documents)

                # Dynamically update the tables with the appended data
                node_table_placeholder.dataframe(node_df)
                edge_table_placeholder.dataframe(edge_df)

                

        st.success("Knowledge Graph generation complete!")
# Dynamically update and display the graph
        net.set_options("""
        {
            "nodes": {
                "physics": {
                    "enabled": true
                },
                "color": {
                    "highlight": {
                        "border": "yellow",
                        "background": "yellow"
                    }
                }
            },
            "edges": {
                "color": {
                    "highlight": "black",
                    "inherit": false
                },
                "hoverWidth": 3
            },
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based"
            }
        }
        """)

        # Generate dynamic HTML for the graph
        html_output = net.generate_html()
        st.components.v1.html(html_output, height=600, width=1000)
