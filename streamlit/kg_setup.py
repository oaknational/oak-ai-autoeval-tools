from neo4j import GraphDatabase
import streamlit as st

# Define your Neo4j Aura credentials
NEO4J_URI = "neo4j+s://e25758e3.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "nWLYMrh0yrKzcdnLK4_XMzBkxlH_qo857_s6LBB9WAE"

# Initialize the Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Page Config
st.set_page_config("Biology KS4", page_icon=":tropical_fish:")

st.write("## Biology KS4 Curriculum Assistant")

def create_knowledge_graph():
    csv_url = "https://drive.google.com/uc?export=download&id=1twDDSBQBqF8D70Yo-EAZbLHDl_cNjKTw"

    # Define the Cypher query to load the data, create nodes, and assign relationships based on the CSV
    query = f"""
    LOAD CSV WITH HEADERS FROM '{csv_url}' AS row
    // Merge Year and Unit nodes with their properties
    MERGE (k:KeyStage {{keyStageTitle: row.keyStageTitle, phaseTitle: row.phaseTitle}})
    MERGE (y:Year {{yearTitle: row.yearTitle}})
    MERGE (s:Subject {{subjectTitle: row.subjectTitle}})
    MERGE (u:Unit {{
        unitTitle: row.unitTitle, 
        unitOrder: toInteger(row.unitOrder), 
        isLegacy: toBoolean(row.isLegacy)
    }})
    // Create a relationship between Year and Unit based on the row data
    MERGE (y)-[:HAS_UNIT]->(u)
    MERGE (s)-[:HAS_UNIT]->(u)
    MERGE (k)-[:HAS_YEAR]->(y)
    """
    
    # Execute the query within a session
    with driver.session() as session:
        session.run(query)
        st.write("Year and Unit nodes with HAS_UNIT relationships created successfully from CSV.")
        

def assign_uuids_to_all_nodes():
    uuid_query = """
    MATCH (n)
    WHERE n.id IS NULL
    SET n.id = apoc.create.uuid()
    """
    with driver.session() as session:
        session.run(uuid_query)
        st.write("Assigned UUIDs to all nodes without a UUID.")


def print_nodes():
    # Query to retrieve and display all nodes
    nodes_query = "MATCH (n) RETURN n"
    with driver.session() as session:
        nodes_result = session.run(nodes_query)
        
        st.write("### Nodes")
        for record in nodes_result:
            node = record["n"]
            properties = dict(node)
            st.write(f"Node: {node.id}, Labels: {node.labels}, Properties: {properties}")


def print_relationships():
    # Query to retrieve and display all relationships with their connected nodes
    relationships_query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    with driver.session() as session:
        relationships_result = session.run(relationships_query)
        
        st.write("### Relationships")
        for record in relationships_result:
            node1 = record["n"]
            node2 = record["m"]
            relationship = record["r"]
            
            # Extract properties from nodes
            node1_title = node1.get("unitTitle") or node1.get("yearTitle") or node1.get("keyStageTitle") or node1.get("subjectTitle") or "Unknown"
            node2_title = node2.get("unitTitle") or node2.get("yearTitle") or node2.get("keyStageTitle") or node2.get("subjectTitle") or "Unknown"
            
            # Print the relationship and connected nodes
            st.write(f"{node1_title} -[:{relationship.type}]-> {node2_title}")


# Run the function to create the knowledge graph
if __name__ == "__main__":
    try:
        create_knowledge_graph()
        assign_uuids_to_all_nodes()
        print_nodes()
        print_relationships()
    finally:
        driver.close()




