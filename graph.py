# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:34:42 2024

@author: jveraz
"""

import os
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
import networkx as nx
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document as DocxDocument  # Importar para leer archivos .docx

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key from the .env file
api_key = os.getenv("OPENAI_API_KEY")

# Función para leer todos los archivos .txt, .pdf y .docx de la carpeta "archivos"
def read_files_from_folder(folder_path):
    content = ""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content += file.read() + "\n"
        elif filename.endswith(".pdf"):
            pdf_reader = PdfReader(file_path)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        elif filename.endswith(".docx"):
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
    return content

# Leer todos los archivos de la carpeta "archivos"
folder_path = "archivos"
lines = read_files_from_folder(folder_path)

# Dividir el texto en fragmentos de 1000 caracteres (ajusta este tamaño si es necesario)
chunk_size = 10000
text_chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

# Crear grafo de NetworkX
nx_graph = nx.Graph()

# Procesar cada fragmento individualmente
for i, chunk in enumerate(text_chunks):
    print(f"Procesando fragmento {i + 1} de {len(text_chunks)}...")

    # Crear un documento con el fragmento de texto
    document = Document(page_content=chunk)
  
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # Configuración del modelo
    llm_transformer = LLMGraphTransformer(llm=llm)

    graph_documents = llm_transformer.convert_to_graph_documents([document])
    
    # Agregar nodos y aristas del fragmento al grafo
    for doc in graph_documents:
        for node in doc.nodes:
            node_id = dict(node)["id"]
            content = dict(node)["type"]
            nx_graph.add_node(node_id, content=content)
        
        for edge in doc.relationships:
            edge_dict = vars(edge) if not isinstance(edge, dict) else edge
            source = str(dict(edge_dict["source"])["id"])
            target = str(dict(edge_dict["target"])["id"])
            tipo = edge_dict["type"]
            nx_graph.add_edge(source, target, content=tipo)
            
# Guardar el grafo en JSON
data = nx.node_link_data(nx_graph)
with open("graph.json", "w") as file:
    json.dump(data, file)
    
# Procesar el componente más grande del grafo
Gcc = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
nx_graph = nx_graph.subgraph(Gcc[0])

# Layout del grafo
pos = nx.kamada_kawai_layout(nx_graph)

# Figura
plt.figure(figsize=(10, 10))

# Dibujar aristas
nx.draw_networkx_edges(nx_graph, pos, edge_color="k", width=0.75)

# Dibujar nodos
nx.draw_networkx_nodes(nx_graph, pos, node_size=500, node_color="gold")

# Dibujar etiquetas
nx.draw_networkx_labels(nx_graph, pos, font_size=7, font_weight="bold")

# Guardar la imagen
plt.savefig("graph.png", format="PNG")
plt.show()
