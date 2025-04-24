import os
import time
from pgmpy.readwrite import BIFReader
import igraph as ig
from src.steps.MES.sample import MES
from src.steps.MES.cpdag import CPDAG
from src.steps.MES.count import count
from tabulate import tabulate


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

    uccgs = CPDAG(graph)[0].subgraph(CPDAG(graph)[0].vs.select(_degree_gt=0))
    component_sizes = uccgs.connected_components().sizes()

    start_time = time.time()
    equivalent_dag = MES(graph)
    count_runtime = time.time() - start_time

    # Plot the graph
    output_image = f"{os.path.splitext(os.path.basename(file_path))[0]}_UCCG.png"
    ig.plot(uccgs, target=output_image)

    return {
        "network_name": os.path.splitext(os.path.basename(file_path))[0],
        "num_nodes": graph.vcount(),
        "component_sizes": component_sizes,
        "sample_runtime": count_runtime,
    }


data_folder = "./data/networks"
output_results = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".bif"):
        file_path = os.path.join(data_folder, file_name)
        result = process_bif_file(file_path)
        output_results.append(result)

output_results.sort(key=lambda x: x["num_nodes"])

table_data = [
    [
        result["network_name"],
        result["num_nodes"],
        ", ".join(map(str, result["component_sizes"])),
        f"{result['sample_runtime']:.4f} s",
    ]
    for result in output_results
]

headers = [
    "Network Name",
    "Num Nodes (|V|)",
    "UCCG Component Sizes",
    "Sampling Runtime",
]

print(tabulate(table_data, headers=headers, tablefmt="grid"))
