# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:52:39 2024

@author: jveraz
"""

import json
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Cargar el archivo JSON
with open("grafo/graph.json", "r") as file:
    graph_data = json.load(file)

# Convertir el JSON a un grafo de NetworkX
nx_graph = json_graph.node_link_graph(graph_data)
    
# Procesar el componente más grande del grafo
#Gcc = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
#nx_graph = nx_graph.subgraph(Gcc[0])

# Layout del grafo
pos = nx.kamada_kawai_layout(nx_graph)

# Figura
plt.figure(figsize=(10, 10))

# Calcular la centralidad de intermediación (betweenness centrality)
centrality = nx.betweenness_centrality(nx_graph)

# Ordenar los nodos de mayor a menor centralidad
sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

# Crear una lista con los nodos ordenados
nodes_sorted_by_centrality = [node for node, centrality_value in sorted_centrality]

# Escalar el tamaño de los nodos basado en la centralidad
node_sizes = [50 + 2500 * centrality[node] for node in nx_graph.nodes()]

# Dibujar aristas
nx.draw_networkx_edges(nx_graph, pos, edge_color="k", width=0.75)

# Dibujar nodos
nx.draw_networkx_nodes(nx_graph, pos, node_size=node_sizes, node_color="gold")

# Dibujar etiquetas
nx.draw_networkx_labels(nx_graph, pos, labels = {n: n for n in nx_graph if n in nodes_sorted_by_centrality[:20]}, font_size=7, font_weight="bold")

# Guardar la imagen
plt.savefig("grafo/graph.png", format="PNG",  dpi=1000)
plt.show()

# aristas
aristas = [item[2]['content'] for item in list(nx_graph.edges(data=True))]

# contar
C = Counter(aristas).most_common()

# relaciones vs número
R = list(zip(*C))[0]
N = list(zip(*C))[1]

# pandas
D = {'tipo de relación':R, 'número':N}
DF = pd.DataFrame.from_dict(D)
DF.to_excel("grafo/relaciones.xlsx")  

# nodos
nodos = [item[1]['content'] for item in list(nx_graph.nodes(data=True)) if 'content' in item[1]]

# contar
C = Counter(aristas).most_common()

# entidades vs número
E = list(zip(*C))[0]
N = list(zip(*C))[1]

# pandas
D = {'tipo de entidad':E, 'número':N}
DF = pd.DataFrame.from_dict(D)
DF.to_excel("grafo/entidades.xlsx")  