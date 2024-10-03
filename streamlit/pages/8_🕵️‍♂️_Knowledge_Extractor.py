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
import uuid
from datetime import datetime
from utils.db_scripts import execute_single_query




st.markdown("""
            <style>
            /* Move the specific graph iframe container closer to the sidebar */
            .stApp iframe {
                margin-left: -150px !important;  /* Move the iframe container closer to the sidebar */
                padding-left: 0px !important;    /* Ensure no padding on the left */
                margin-bottom: -400px !important;  /* Reduce space below the iframe */
            }

            /* Ensure the iframe takes full width of its container */
            .stApp .stComponent {
                width: 100% !important;
                margin-left: -150px !important;  /* Adjust this value to move graph */
                padding-left: 0px !important;
            }
            </style>
            """, unsafe_allow_html=True)

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

# Define Node, Relationship, and GraphDocument classes
@dataclass
class Node:
    id: str
    type: str
    properties: Dict

@dataclass
class Edge:
    source: str  # Node ID
    target: str  # Node ID
    type: str
    properties: Dict

@dataclass
class GraphDocument:
    nodes: List[Node]
    edges: List[Edge]

def sanitize_llm_response(response):
    # Modify the first regex to be more efficient by using a non-greedy match
    response = re.sub(r"}\s*\{", r"}, {", response)
    
    # Modify the second regex for more efficiency by reducing the pattern complexity
    response = re.sub(r'"\s*\{', r'", {', response)
    
    # Remove any code blocks and strip leading/trailing spaces
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

    edges = []
    for rel in data.get('edges', []):
        source_id = rel.get('source')
        target_id = rel.get('target')
        rel_type = rel.get('type')
        if strict_mode and rel_type not in allowed_relationships:
            continue
        if source_id in node_ids and target_id in node_ids:
            edges.append(rel)
        else:
            st.warning(f"Invalid edge: {source_id} -> {target_id}, nodes not found.")

    return GraphDocument(nodes=nodes, edges=edges)

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
def create_custom_prompt(document_content, allowed_nodes, allowed_relationships, discovered_nodes):
    allowed_nodes_str = ', '.join(f'"{node}"' for node in allowed_nodes)
    allowed_relationships_str = ', '.join(f'"{rel}"' for rel in allowed_relationships)
    discovered_nodes_str = ', '.join(f'"{node}"' for node in discovered_nodes)

    prompt = f"""
    You are an expert in extracting knowledge graph data from curriculum content. Please extract the key elements in the form of nodes and relationships, ensuring that all relationships refer to the correct node IDs.

    **Instructions:**

    1. Each node must have a unique `id`, `type`, and `properties` (including at least a `label`).
    2. Each relationship must have a `source` (the node's unique `id`), `target` (another node's unique `id`), and a `type` (relationship type).
    3. Ensure that all node IDs are referenced properly in relationships.
    4. You will also be given a list of node IDs from previous responses to reference. If there are no nodes provided yet, it means you will start fresh.
    5. Only use the following allowed node types: {allowed_nodes_str}
    6. Only use the following allowed relationship types: {allowed_relationships_str}
    7. If there is content that does not fit the allowed nodes or relationships, you can skip it.
    8. This is for teachers to use to plan their lessons for the semester so you don't need to include every detail, just the most important elements of the curriculum.
    9. Science should be a Subject while Physics, Chemistry, and Biology should be SubjectDiscipline.
    10. Output format should be JSON.

    **Example Output:**

    {{
        "nodes": [
            {{"id": "Subject_1", "type": "Subject", "properties": {{"label": "Science"}}}},
            {{"id": "SubjectDiscipline_1", "type": "SubjectDiscipline", "properties": {{"label": "Physics"}}}}
        ],
        "edges": [
            {{"source": "Subject_1", "target": "SubjectDiscipline_1", "type": "INCLUDES"}}
        ]
    }}

    **Content to Analyze:**
    {document_content}


    **Node IDs from previous responses:**
    {discovered_nodes_str}
    """
    return prompt

def DocumentRelationshipWriter(document_content,allowed_nodes, allowed_relationships):
    allowed_nodes_str = ', '.join(f'"{node}"' for node in allowed_nodes)
    allowed_relationships_str = ', '.join(f'"{rel}"' for rel in allowed_relationships)
    prompt = f"""
    You are tasked with rewriting complex texts by identifying and clarifying the key nodes (important concepts, entities, or terms) and relationships between them. Your output must strictly follow the relationship sentence format, where each sentence explicitly states the connection between two key concepts.

    Your Responsibilities:

    Identify Key Concepts (Nodes):
    Break down the document into its core elements by identifying the important terms, concepts, or entities. These should be central to understanding the document’s content and will serve as the "nodes" in the knowledge graph. Use the following allowed node labels: {allowed_nodes_str}.

    Clarify Relationships (Edges):
    For each important concept, identify how it relates to other key concepts within the text only using one of the {allowed_relationships_str}. Each sentence must describe the relationship between two key concepts using this format:
    [Subject Node] [Relationship Verb] [Object Node].
    
    Ensure Consistency in Relationships:
    Use the following allowed relationship types: {allowed_relationships_str}. If there are multiple possible relationships, select the one that best reflects the context and rewrite it in the relationship sentence format.

    Simplify Without Oversimplifying:
    Your job is to make the relationships easier to understand, but do not strip away any important nuances. However, every relationship should be expressed as a simple, clear sentence in the format described.

    Reorganize if Necessary:
    If the original text is disorganized, feel free to reorder sentences to ensure that each relationship is clearly presented. However, all content must be delivered in the relationship sentence format.

    Key Constraints:

    No descriptive or explanatory text should be included.
    Every sentence must strictly follow the format:
    [Subject Node] [Relationship Verb] [Object Node].
    Example Output:

    Original Text:
    “Subject content Science – Physics
    Key Stage 3 – Year 8 
    Pupils should be taught about:
    Energy
    Calculation of fuel uses and costs in the domestic context
     comparing energy values of different foods (from labels) (kJ)
     comparing power ratings of appliances in watts (W, kW)
     comparing amounts of energy transferred (J, kJ, kW hour)
     domestic fuel bills, fuel use and costs
     fuels and energy resources. 
    Lessons in unit:
    Energy and temperature
    Energy and substance
”
    Rewritten:

    "
    Subject Science includes SubjectDiscipline Physics.
    KeyStage Key Stage 3 includes YearGroup Year 8.
    YearGroup Year 8 includes SubjectDicipline Physics.
    SubjectDiscipline Physics includes Topic Energy.
    Topic Energy includes Subtopic Calculation of fuel usage.
    Subtopic Calculation of fuel usage involves Skill measurement
    Subtopic Calculation of fuel usage requires Knowledge basic arithmetic.
    Subtopic Calculation of fuel usage hasActivity Activity calculating energy values of foods.
    Subtopic Calculation of fuel usage hasActivity Activity comparing power ratings of appliances.
    Subtopic Calculation of fuel usage hasActivity Activity comparing energy transfer amounts.
    Subtopic Calculation of fuel usage hasLearningOutcome LearningOutcome fuels and energy resources.
    Topic Energy includes Lesson Energy and temperature.
    Topic Energy includes Lesson Energy and substance.
    "


    End Goal:
    Your output must consist entirely of relationship sentences in this format, clearly stating the connection between key nodes, to make it easy to extract relationships for a knowledge graph.

    **Content to Re-write:**
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
        for rel in graph_document.edges
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

import json

def add_data_to_knowledge_graph(json_data, source):
    """
    Insert a new record into the knowledge_graph table with JSON data, nodes, edges, and timestamps.

    Args:
        json_data (dict): The JSON data to be inserted, containing nodes and edges.
        source (str): The source of the knowledge graph data (e.g., the uploaded file name).

    Returns:
        bool: True if successful, False otherwise.
    """
    query = """
    INSERT INTO public.knowledge_graph (
        id, created_at, updated_at, nodes, edges, json, source
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    
    # Generate a unique ID and current timestamp for created_at and updated_at
    record_id = str(uuid.uuid4())
    now = datetime.now()

    # Extract nodes and relationships from json_data
    nodes = json_data.get('nodes', [])
    edges = json_data.get('edges', [])
    
    # Convert the nodes and edges dict to JSON strings
    nodes_str = json.dumps(nodes)
    edges_str = json.dumps(edges)
    
    # Convert the whole dict to a JSON string for the json column
    json_data_str = json.dumps(json_data)

    # Call the execute_single_query function to insert the record
    return execute_single_query(
        query,
        (record_id, now, now, nodes_str, edges_str, json_data_str, source)
    )





# Streamlit app setup
st.title("Knowledge Extractor")


# Upload file input
st.header("Upload a Content File to Build a Knowledge Graph")
uploaded_file = st.file_uploader('', type=["txt"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    content = uploaded_file.read().decode("utf-8")

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

    graph_placeholder = st.empty()
    # Create a temporary file to store uploaded content for TextLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(content.encode("utf-8"))
        temp_file_path = temp_file.name

    # Use the temporary file path with TextLoader
    loader = TextLoader(file_path=temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    documents = text_splitter.split_documents(documents=docs)

    allowed_nodes = [
        'KeyStage', 'Subject','SubjectDiscipline', 'Topic', 'Subtopic',
        'Lesson', 'LearningOutcome', 'Skill',   'Knowledge',
          'Activity', 'YearGroup'
    ]

    allowed_relationships = [
        'includes', 'involves',
         'hasLearningOutcome',  'hasActivity', 
    ]

 
    
    
    # Button to start extraction
    if st.button("Start Extraction"):
        # Initialize placeholders
        progress_bar = st.progress(0)
        total_documents = len(documents)
        st.write(f"**Total Documents to Process: {total_documents}**")
        

        relationship_chunk_prompt_place_holder = st.empty()
        relationship_chunk_place_holder = st.empty()
        prompt_place_holder = st.empty()
        response_place_holder = st.empty()

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



            relationship_chunk_prompt= DocumentRelationshipWriter(document.page_content,allowed_nodes, allowed_relationships)

            # relationship_chunk_prompt_place_holder.markdown(relationship_chunk_prompt)
            relationship_chunk = generate_response(relationship_chunk_prompt, llm_model='gpt-4o-mini', llm_model_temp=0)
            relationship_chunk_place_holder.markdown(relationship_chunk)
            # Access document content correctly using document.page_content
            prompt = create_custom_prompt(relationship_chunk, allowed_nodes, allowed_relationships, st.session_state.all_nodes)
            # prompt_place_holder.markdown(prompt)
            response = generate_response(prompt, llm_model='gpt-4o', llm_model_temp=0.2)
            # response_place_holder.markdown(response)

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

            # Generate dynamic HTML for the graph
            html_output = net.generate_html()
            html_output = html_output.replace("</body>", custom_js + "</body>")
            
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

        if st.button("Save Knowledge Graph to Database"):
            # Create a JSON representation of the nodes and edges
            knowledge_graph_data = {
                "nodes": st.session_state.all_nodes,
                "edges": st.session_state.all_edges
            }

            # Convert to JSON string (if needed for storage purposes)
            knowledge_graph_json = json.dumps(knowledge_graph_data, indent=2)

            # Display the generated JSON (for visual confirmation)
            st.subheader("Generated Knowledge Graph JSON")
            st.code(knowledge_graph_json, language="json")

            # Save the data to the knowledge_graph table in the PostgreSQL database
            success = add_data_to_knowledge_graph(knowledge_graph_data, file_name)

            if success:
                st.success("Knowledge graph saved successfully!")
            else:
                st.error("Failed to save the knowledge graph.")

else: 
    
    
    st.write("Please upload a file to start the extraction process.")


        