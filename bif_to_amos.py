import os
import time
from pgmpy.readwrite import BIFReader
import igraph as ig
from src.steps.MES.cpdag import CPDAG
from src.steps.MES.count import count


def process_bif_file(file_path):
    """
    Reads a .bif file using pgmpy and converts the Bayesian network structure
    to an igraph graph, computes connected component sizes, and plots the graph.

    Parameters:
        file_path (str): Path to the .bif file.

    Returns:
        dict: A dictionary containing the network name, number of nodes,
              component sizes, count value, and count runtime.
    """
    # Read the .bif file and convert to igraph
    reader = BIFReader(file_path)
    model = reader.get_model()
    node_mapping = list(model.nodes())
    edges = list(model.edges())

    # Map names to indices and create graph
    name_to_index = {name: idx for idx, name in enumerate(node_mapping)}
    indexed_edges = [(name_to_index[u], name_to_index[v]) for u, v in edges]
    graph = ig.Graph(directed=True)
    graph.add_vertices(len(node_mapping))
    graph.add_edges(indexed_edges)

    # Process the graph to extract connected component sizes
    uccgs = CPDAG(graph)[0].subgraph(CPDAG(graph)[0].vs.select(_degree_gt=0))
    component_sizes = uccgs.connected_components().sizes()

    # Measure runtime of count
    start_time = time.time()
    count = 0
    for uccg in uccgs:
        count += count(uccg)
    count_runtime = time.time() - start_time

    # Plot the graph
    output_image = f"{os.path.splitext(os.path.basename(file_path))[0]}_UCCG.png"
    ig.plot(uccgs, target=output_image)

    return {
        "network_name": os.path.splitext(os.path.basename(file_path))[0],
        "num_nodes": graph.vcount(),
        "component_sizes": component_sizes,
        "count_value": count,
        "count_runtime": count_runtime,
    }


# Define the folder containing .bif files
data_folder = "./data/networks"
output_results = []

# Process each .bif file in the folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".bif"):
        file_path = os.path.join(data_folder, file_name)
        result = process_bif_file(file_path)
        output_results.append(result)

# Display results nicely
for result in output_results:
    print(f"Network: {result['network_name']}")
    print(f"  Number of Nodes: {result['num_nodes']}")
    print(f"  Component Sizes: {result['component_sizes']}")
    print(f"  Count Value: {result['count_value']}")
    print(f"  Count Runtime: {result['count_runtime']:.4f} seconds")
    print("-" * 40)
