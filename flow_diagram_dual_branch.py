# Using graphviz for a clean, dedicated flowchart
from graphviz import Digraph

# Define the nodes and edges clearly
nodes = [
    "Multi-omics Data",
    "Autoencoder\n(Dimensionality Reduction)",
    "Latent Embeddings",
    "Attentional Module\n(Transformer)",
    "Attentional Embeddings",
    "Similarity Graph Construction",
    "Knowledge-based Graph Construction\n(e.g., Gene-Gene, miRNA-Gene)",
    "Hybrid Graph Integration",
    "Graph Convolutional Neural Network (GCN)",
    "Patient Clustering",
    "Survival Analysis",
    "Clinical Insights"
]

edges = [
    ("Multi-omics Data", "Autoencoder\n(Dimensionality Reduction)"),
    ("Autoencoder\n(Dimensionality Reduction)", "Latent Embeddings"),
    ("Latent Embeddings", "Attentional Module\n(Transformer)"),
    ("Attentional Module\n(Transformer)", "Attentional Embeddings"),
    ("Attentional Embeddings", "Similarity Graph Construction"),
    ("Knowledge Databases\n(e.g., BioGrid, miRDB)", "Prior Knowledge Graph"),
    ("Similarity Graph Construction", "Combined Graph"),
    ("Prior Knowledge Graph", "Combined Graph"),
    ("Combined Graph", "Graph Convolutional Neural Network (GCN)"),
    ("Graph Convolutional Neural Network (GCN)", "Patient Clustering"),
    ("Patient Clustering", "Survival Analysis")
]

from graphviz import Digraph

# Define and render the flowchart
chart = Digraph('ArchitectureFlow', format='png')

# Adding nodes
for node in ["Multi-omics Data", "Autoencoder\n(Dimensionality Reduction)", "Latent Embeddings", "Attentional Module\n(Transformer)", "Attentional Embeddings", "Similarity Graph Construction", "Knowledge Databases", "Prior Knowledge Graph", "Combined Graph", "Graph Convolutional Neural Network (GCN)", "Patient Risk Clusters", "Survival Analysis", "Clinical Insights"]:
    chart.node(node, shape='rectangle', style='rounded, filled', fillcolor="lightgrey")

# Additional nodes explicitly defined
chart.node("Knowledge Databases", shape="folder", style="filled", fillcolor="orange")
chart.node("Prior Knowledge Graph", shape="box", style="filled", fillcolor="lightyellow")
chart.node("Combined Graph", shape="ellipse", style="filled", fillcolor="lightgreen")
chart.node("Graph Convolutional Neural Network (GCN)", shape="box", style="filled", fillcolor="lightcyan")

# Adding edges explicitly for better clarity
chart.edge("Multi-omics Data", "Autoencoder\n(Dimensionality Reduction)")
chart.edge("Autoencoder\n(Dimensionality Reduction)", "Latent Embeddings")
chart.edge("Latent Embeddings", "Attentional Module\n(Transformer)")
chart.edge("Attentional Module\n(Transformer)", "Attentional Embeddings")
chart.edge("Attentional Embeddings", "Similarity Graph Construction")
chart.edge("Knowledge Databases", "Prior Knowledge Graph")
chart.edge("Similarity Graph Construction", "Combined Graph")
chart.edge("Prior Knowledge Graph", "Combined Graph")
chart.edge("Knowledge Databases", "Prior Knowledge Graph")
chart.edge("Combined Graph", "Graph Convolutional Neural Network (GCN)")
chart.edge("Graph Convolutional Neural Network (GCN)", "Patient Risk Clustering")
chart.edge("Patient Risk Clustering", "Survival Analysis")

# Rendering graph
chart.attr(rankdir='TB')  # Top to Bottom layout
chart.attr(label='Proposed Dual-Branch Multi-omics Integration Flowchart', fontsize='20')

chart.render("dual_branch_flowchart", format="png", cleanup=True)
chart.view()