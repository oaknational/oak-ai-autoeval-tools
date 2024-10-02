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

def build_graph(nodes, edges, selected_node_types, selected_relationship_types):
    G = nx.Graph()
    net = Network(notebook=True)
    
    # Filter nodes and collect categories
    filtered_nodes = [node for node in nodes if node.get('category', 'Unknown') in selected_node_types]
    node_ids = set(node['id'] for node in filtered_nodes)
    unique_categories = set(node.get('category', 'Unknown') for node in filtered_nodes)
    
    # Color mapping
    color_map = plt.cm.get_cmap('hsv', len(unique_categories) + 1)
    category_colors = {category: mcolors.rgb2hex(color_map(i / (len(unique_categories) + 1))[:3]) for i, category in enumerate(unique_categories)}
    
    # Add nodes
    for node in filtered_nodes:
        node_id = node['id']
        category = node.get('category', 'Unknown')
        color = category_colors.get(category, 'gray')
        title = f"{category}"
        net.add_node(node_id, label=node['label'], color=color, title=title, physics=True)
        G.add_node(node_id, label=node['label'], category=category)
    
    # Add edges
    for edge in edges:
        if edge.get('type', 'Unknown') in selected_relationship_types:
            source = edge['source']
            target = edge['target']
            relationship_type = edge.get('type', 'Unknown')
            if source in node_ids and target in node_ids:
                net.add_edge(source, target, title=relationship_type)
                G.add_edge(source, target, type=relationship_type)
    
    # Set visual options
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
    
    return net

# Streamlit app setup
st.title("Knowledge Extractor")

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

    # Initialize session state variables if they don't exist
    if 'all_nodes' not in st.session_state:
        st.session_state.all_nodes = []
    if 'all_edges' not in st.session_state:
        st.session_state.all_edges = []
    if 'node_types' not in st.session_state:
        st.session_state.node_types = set()
    if 'relationship_types' not in st.session_state:
        st.session_state.relationship_types = set()
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    
    # Button to start extraction
    if st.button("Start Extraction"):
        # Initialize placeholders
        progress_bar = st.progress(0)
        total_documents = len(documents)
        st.write(f"**Total Documents to Process: {total_documents}**")
        graph_placeholder = st.empty()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted Nodes")
            node_table_placeholder = st.empty()
        with col2:
            st.subheader("Extracted Edges")
            edge_table_placeholder = st.empty()
        strict_mode = True
        

        # Initialize DataFrames to store nodes and edges for the table
        node_df = pd.DataFrame(columns=["id", "label", "category"])
        edge_df = pd.DataFrame(columns=["source", "target", "type"])

        # Process each document
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

            # Add new nodes and edges to session state
            st.session_state.all_nodes.extend(new_nodes)
            st.session_state.all_edges.extend(new_edges)

            # Update node types and relationship types
            st.session_state.node_types.update([node.get('category', 'Unknown') for node in new_nodes])
            st.session_state.relationship_types.update([edge.get('type', 'Unknown') for edge in new_edges])

            # Convert the new nodes and edges to DataFrames
            new_node_df = pd.DataFrame(new_nodes)
            new_edge_df = pd.DataFrame(new_edges)

            # Append the new data to the existing DataFrames
            node_df = pd.concat([node_df, new_node_df], ignore_index=True)
            edge_df = pd.concat([edge_df, new_edge_df], ignore_index=True)

            # Update the progress bar
            progress_bar.progress((idx + 1) / total_documents)

            # Build the graph based on filters
            selected_node_types = list(st.session_state.node_types)
            selected_relationship_types = list(st.session_state.relationship_types)

            net = build_graph(st.session_state.all_nodes, st.session_state.all_edges, selected_node_types, selected_relationship_types)

            # Inject custom JavaScript
            custom_js = """
                <script type="text/javascript">
                var nodes = network.body.nodes;
                var edges = network.body.edges;

                network.on('click', function(properties) {
                    var clickedNode = properties.nodes[0];
                    if (clickedNode !== undefined) {
                        var node = network.body.nodes[clickedNode];

                        // Unfix the clicked node to allow it to be moved freely
                        node.setOptions({ fixed: { x: false, y: false }, physics: true });

                        var connectedNodes = network.getConnectedNodes(clickedNode);

                        // Highlight clicked node and its neighbors
                        Object.values(nodes).forEach(function(node) {
                            if (connectedNodes.includes(node.id) || node.id == clickedNode) {
                                node.setOptions({ opacity: 1 });  // Full opacity for neighbors
                            } else {
                                node.setOptions({ opacity: 0.2 });  // Fade non-connected nodes
                            }
                        });

                        Object.values(edges).forEach(function(edge) {
                            if (connectedNodes.includes(edge.to) || connectedNodes.includes(edge.from)) {
                                edge.setOptions({ color: 'black' });
                            } else {
                                edge.setOptions({ color: 'rgba(200,200,200,0.5)' });  // Fade non-connected edges
                            }
                        });
                    } else {
                        // Reset if nothing clicked
                        Object.values(nodes).forEach(function(node) {
                            node.setOptions({ opacity: 1 });
                        });
                        Object.values(edges).forEach(function(edge) {
                            edge.setOptions({ color: 'black' });
                        });
                    }
                });

                // Fix node position after drag
                network.on('dragEnd', function(properties) {
                    properties.nodes.forEach(function(nodeId) {
                        var node = network.body.nodes[nodeId];
                        node.setOptions({ fixed: { x: true, y: true }, physics: false });  // Fix position after dragging
                    });
                });
                </script>
            """

            # Generate dynamic HTML for the graph
            html_output = net.generate_html()
            html_output = html_output.replace("</body>", custom_js + "</body>")
            st.markdown("""
            <style>
            /* Move the specific graph iframe container closer to the sidebar */
            .stApp iframe {
                margin-left: -150px !important;  /* Move the iframe container closer to the sidebar */
                padding-left: 0px !important;    /* Ensure no padding on the left */
            }

            /* Ensure the iframe takes full width of its container */
            .stApp .stComponent {
                width: 100% !important;
                margin-left: -150px !important;  /* Adjust this value to move graph */
                padding-left: 0px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            # Update the graph placeholder
            with graph_placeholder:
                st.components.v1.html(html_output, height=1000, width=1200)

            # Update the data tables
            with col1:
                node_table_placeholder.dataframe(node_df)
            with col2:
                edge_table_placeholder.dataframe(edge_df)

        st.success("Knowledge Graph generation complete!")
        st.session_state.processing_complete = True

    # Sidebar filters (after processing is complete)
    if st.session_state.processing_complete:
        st.sidebar.title("Graph Filters")
        selected_node_types = st.sidebar.multiselect(
            "Select node types to display:",
            options=list(st.session_state.node_types),
            default=list(st.session_state.node_types)
        )

        selected_relationship_types = st.sidebar.multiselect(
            "Select relationship types to display:",
            options=list(st.session_state.relationship_types),
            default=list(st.session_state.relationship_types)
        )

        # Build the graph based on filters
        net = build_graph(st.session_state.all_nodes, st.session_state.all_edges, selected_node_types, selected_relationship_types)

        # Inject custom JavaScript
        custom_js = """
            <script type="text/javascript">
                var nodes = network.body.nodes;
                var edges = network.body.edges;

                network.on('click', function(properties) {
                    var clickedNode = properties.nodes[0];
                    if (clickedNode !== undefined) {
                        var node = network.body.nodes[clickedNode];

                        // Unfix the clicked node to allow it to be moved freely
                        node.setOptions({ fixed: { x: false, y: false }, physics: true });

                        var connectedNodes = network.getConnectedNodes(clickedNode);

                        // Highlight clicked node and its neighbors
                        Object.values(nodes).forEach(function(node) {
                            if (connectedNodes.includes(node.id) || node.id == clickedNode) {
                                node.setOptions({ opacity: 1 });  // Full opacity for neighbors
                            } else {
                                node.setOptions({ opacity: 0.2 });  // Fade non-connected nodes
                            }
                        });

                        Object.values(edges).forEach(function(edge) {
                            if (connectedNodes.includes(edge.to) || connectedNodes.includes(edge.from)) {
                                edge.setOptions({ color: 'black' });
                            } else {
                                edge.setOptions({ color: 'rgba(200,200,200,0.5)' });  // Fade non-connected edges
                            }
                        });
                    } else {
                        // Reset if nothing clicked
                        Object.values(nodes).forEach(function(node) {
                            node.setOptions({ opacity: 1 });
                        });
                        Object.values(edges).forEach(function(edge) {
                            edge.setOptions({ color: 'black' });
                        });
                    }
                });

                // Fix node position after drag
                network.on('dragEnd', function(properties) {
                    properties.nodes.forEach(function(nodeId) {
                        var node = network.body.nodes[nodeId];
                        node.setOptions({ fixed: { x: true, y: true }, physics: false });  // Fix position after dragging
                    });
                });
                </script>
        """

        # Generate dynamic HTML for the graph
        html_output = net.generate_html()
        html_output = html_output.replace("</body>", custom_js + "</body>")
        
        st.markdown("""
            <style>
            /* Move the specific graph iframe container closer to the sidebar */
            .stApp iframe {
                margin-left: -150px !important;  /* Move the iframe container closer to the sidebar */
                padding-left: 0px !important;    /* Ensure no padding on the left */
            }

            /* Ensure the iframe takes full width of its container */
            .stApp .stComponent {
                width: 100% !important;
                margin-left: -150px !important;  /* Adjust this value to move graph */
                padding-left: 0px !important;
            }
            </style>
            """, unsafe_allow_html=True)

        # Update the graph placeholder
        with graph_placeholder:
            st.components.v1.html(html_output, height=1000, width=1200)

        # Update the data tables
        with col1:
            node_df = pd.DataFrame(st.session_state.all_nodes)
            node_table_placeholder.dataframe(node_df)
        with col2:
            edge_df = pd.DataFrame(st.session_state.all_edges)
            edge_table_placeholder.dataframe(edge_df)
