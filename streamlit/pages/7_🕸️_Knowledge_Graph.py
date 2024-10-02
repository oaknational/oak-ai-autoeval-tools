from pyvis.network import Network
import networkx as nx
import json
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current script's directory
base_dir = os.path.dirname(script_dir)

kg_json_file_path = os.path.join(base_dir, 'data', 'knowledge_graph.json')

# Check if the file exists
if not os.path.exists(kg_json_file_path):
    st.error(f"File not found: {kg_json_file_path}")
else:
    with open(kg_json_file_path, 'r') as file:
        data = json.load(file)


# Extract unique node types and relationship types from the data
node_types = set([node.get('category', 'Unknown') for node in data['nodes']])
relationship_types = set([edge['type'] for edge in data['edges']])

# Streamlit UI for selecting node types and relationships to display
st.sidebar.title("Graph Filters")

selected_node_types = st.sidebar.multiselect(
    "Select node types to display:", list(node_types), default=list(node_types)
)

selected_relationship_types = st.sidebar.multiselect(
    "Select relationship types to display:", list(relationship_types), default=list(relationship_types)
)

# Create a graph
G = nx.Graph()

# Create a set to track added nodes
added_node_ids = set()

# Filter and add nodes based on the selected node types
selected_node_ids = set()
for node in data['nodes']:
    node_id = node['id']
    # Check if the node has already been added (based on its ID)
    if node_id not in added_node_ids and node.get('category', 'Unknown') in selected_node_types:
        G.add_node(node_id, label=node['label'], category=node.get('category', 'Unknown'))
        selected_node_ids.add(node_id)  # Store the IDs of the selected nodes
        added_node_ids.add(node_id)  # Track the node as added

# Filter and add edges based on the selected relationship types and selected nodes
for edge in data['edges']:
    if edge['type'] in selected_relationship_types and edge['source'] in selected_node_ids and edge['target'] in selected_node_ids:
        G.add_edge(edge['source'], edge['target'], relationship=edge.get('relationship', ''), type=edge['type'])

# Identify unique categories in the filtered graph
unique_categories = set([node[1].get('category', 'Unknown') for node in G.nodes(data=True)])

# Dynamically generate colors for each unique category
color_map = plt.cm.get_cmap('hsv', len(unique_categories))  # You can choose any colormap here
category_colors = {category: mcolors.rgb2hex(color_map(i)[:3]) for i, category in enumerate(unique_categories)}

# Create a pyvis network
net = Network(notebook=True)  # Initialize net before adding nodes and edges

# Add nodes with dynamic colors based on their category and enable dragging with physics enabled initially
for node, node_attrs in G.nodes(data=True):
    category = node_attrs.get('category', 'Unknown')
    color = category_colors.get(category, 'gray')  # Default color is gray if no category is found
    title = f"{category}"  # Tooltip text
    # Add node with physics enabled for dragging, and allow the position to be fixed after drag
    net.add_node(node, label=node_attrs.get('label'), color=color, title=title, physics=True)

# Add edges and set titles for edge relationships
for source, target, edge_attrs in G.edges(data=True):
    relationship_type = edge_attrs.get('type', 'No relationship defined')  # Fallback if type is missing
    net.add_edge(source, target, title=relationship_type)

# Set visual options (only JSON, no JavaScript here)
net.set_options("""
{
    "nodes": {
        "physics": true,
        "fixed": {
            "x": false,
            "y": false
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

# Inject custom JavaScript separately into the HTML
html_output = net.generate_html()
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

# Append the custom JavaScript to the HTML output
html_output = html_output.replace("</body>", custom_js + "</body>")

# Save the final HTML with the custom JS
with open("graph.html", "w") as f:
    f.write(html_output)

# Inject custom CSS to align the graph container closer to the sidebar
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

# Display the graph in Streamlit
st.components.v1.html(open("graph.html").read(), height=1000, width=1000)