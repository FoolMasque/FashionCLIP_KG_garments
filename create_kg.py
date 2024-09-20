import rdflib
import networkx as nx
import matplotlib.pyplot as plt
import re


def get_graph_data():
    # Loading the Colour Knowledge Graph
    g = rdflib.Graph()
    g.parse("ontology.ttl", format="ttl")

    # Create a NetworkX graph to represent the knowledge graph
    G = nx.Graph()

    # Parsing category files
    category_file = "list_category_cloth.txt"
    with open(category_file, "r") as f:
        lines = f.readlines()
        category_count = int(lines[0].strip())  # of categories obtained
        categories = [line.split() for line in lines[2:]]  # Read from the third line

    # Parsing Property Files
    attribute_file = "list_attr_cloth1.txt"
    with open(attribute_file, "r") as f:
        lines = f.readlines()
        attribute_count = int(lines[0].strip())  # Get the number of attributes
        attributes = [re.split(r'\s{2,}', line.strip()) for line in lines[2:]]  # Read from the third line

    # Adding apparel categories to the knowledge graph
    for category_name, category_type in categories:
        G.add_node(category_name, type="Clothing Category", category_type=category_type)

    # Adding clothing attributes to the knowledge graph
    for attribute_name, attribute_type in attributes:
        G.add_node(attribute_name, type="Clothing Attribute", attribute_type=attribute_type)

    # Define more comprehensive connection rules
    for category_name, category_type in categories:
        for attribute_name, attribute_type in attributes:
            # Upper Body Clothing Rules
            if category_type == "1":  # Upper Body Clothing
                if attribute_type == "1":  # material
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # farbic
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # shape
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # part
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # style
                    G.add_edge(category_name, attribute_name, relation="has_style")

            # Lower Body Clothing Rules
            elif category_type == "2":  # Lower Body Clothing
                if attribute_type == "1":
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":
                    G.add_edge(category_name, attribute_name, relation="has_style")

            # Full Body Clothing Rule
            elif category_type == "3":
                if attribute_type == "1":
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":
                    G.add_edge(category_name, attribute_name, relation="has_style")

    # Integration of colour knowledge graph (colour nodes from colour ontology and linking to garment categories, attributes)
    namespace_manager = g.namespace_manager

    def remove_prefix(uri):
        uri_str = str(uri)
        for prefix, namespace in namespace_manager.namespaces():
            if uri_str.startswith(str(namespace)):
                return uri_str[len(str(namespace)):]
        return uri_str

    for subj, pred, obj in g:
        subj_filtered = remove_prefix(subj)
        pred_filtered = remove_prefix(pred)
        obj_filtered = remove_prefix(obj)

        # Handling the hasColourName and isOfGroup relationships
        if pred_filtered == "hasColourName":
            G.add_node(obj_filtered, label="ColourName")
            G.add_edge(subj_filtered, obj_filtered, relation="hasColourName")
        elif pred_filtered == "isOfGroup":
            G.add_node(obj_filtered, label="ColourGroup")
            G.add_edge(subj_filtered, obj_filtered, relation="isOfGroup")

        # Handling of colour-defined attributes such as brightness, saturation, etc.
        elif pred_filtered in ["brightness", "hue", "saturation", "temperature", "hex_string"]:
            G.add_node(obj_filtered, label="Attribute")
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

        # Handling relationships between colour groups
        elif pred_filtered in ["complementaryOf", "analogousWith", "triadicWith", "tetradicWith"]:
            G.add_node(subj_filtered, label="ColourGroup")
            G.add_node(obj_filtered, label="ColourGroup")
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

        # Dealing with other relationships not expressly designated
        else:
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

    # print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    # Get a list of all entities (nodes)
    all_entities = list(G.nodes())
    # given entity
    # Construct one-hot vectors for each entity
    node_features = {entity: [1 if i == entity_index else 0 for i in range(len(all_entities))] for
                     entity_index, entity in enumerate(all_entities)}

    return G, all_entities, node_features
    # Calling visualisation functions
    visualize_graph(G)


# Visual Knowledge Graph
def visualize_graph(G):
    # Suppose G is your network graph, with nodes and relations already loaded
    pos = nx.spring_layout(G, k=0.3, iterations=50)  # Layout settings

    plt.figure(figsize=(20, 20))  # Setting a larger image size

    # Drawing nodes
    node_labels = {node: node for node, data in G.nodes(data=True)}  # Only node names are displayed
    node_sizes = [len(list(G.neighbors(node))) * 50 for node in G.nodes()]  # Adjust node size according to node degree
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Draw edges and enable arrows
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color='black')  # arrows=True start using FancyArrowPatches

    # Drawing labels on edges (showing the type of relationship between edges)
    edge_labels = nx.get_edge_attributes(G, 'relation')  # 获取边的关系标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

    # plot
    plt.title("Colour Ontology Knowledge Graph")
    plt.show()


if __name__ == '__main__':
    get_graph_data()
