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
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document as DocxDocument 
from rapidfuzz import fuzz, process

############
## OPENAI ##
############

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key from the .env file
api_key = os.getenv("OPENAI_API_KEY")

##############
## ARCHIVOS ##
##############

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

###################
## NORMALIZACIÓN ##
###################

# Diccionario de equivalencias para normalización

NAME_EQUIVALENCES = {
    "ancelmo rojas": "anselmo rojas",
    "anselmo rojas": "anselmo rojas",
    "don anselmo rojas": "anselmo rojas",
    "don alsemo rojas": "anselmo rojas",
    "alsemo rojas": "anselmo rojas",
    "anselmo": "anselmo rojas",
    "collas" : "qollas",
    "mamita del rosario": "virgen del rosario",
    "festividad de la virgen del rosario": "virgen del rosario",
    "carmen": "virgen del carmen"
}

def normalize_name(name):
    """
    Normaliza un nombre utilizando un diccionario de equivalencias, eliminando títulos y errores ortográficos.
    Args:
        name (str): El nombre a normalizar.
    Returns:
        str: El nombre normalizado.
    """
    # Convertir a minúsculas y quitar espacios
    name = name.lower().strip()

    # Eliminar títulos comunes
    for title in ["don", "mr.", "mrs.", "sr.", "dr."]:
        name = name.replace(title, "").strip()

    # Si está en el diccionario de equivalencias, usar el valor mapeado
    if name in NAME_EQUIVALENCES:
        return NAME_EQUIVALENCES[name]

    # Buscar coincidencias similares usando RapidFuzz
    known_names = list(NAME_EQUIVALENCES.values())
    normalized = process.extractOne(name, known_names, scorer=fuzz.ratio)
    if normalized and normalized[1] >= 80:  # Umbral de similitud
        return normalized[0]

    # Si no hay coincidencias, devolver el nombre procesado
    return name

########################
## DIVISIÓN EN CHUNKS ##
########################

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
            node_id = normalize_name(dict(node)["id"])
            content =normalize_name( dict(node)["type"])
            nx_graph.add_node(node_id, content=content)
        
        for edge in doc.relationships:
            edge_dict = vars(edge) if not isinstance(edge, dict) else edge
            source = normalize_name(str(dict(edge_dict["source"])["id"]))
            target = normalize_name(str(dict(edge_dict["target"])["id"]))
            tipo = edge_dict["type"]
            nx_graph.add_edge(source, target, content=tipo)
            
# Guardar el grafo en JSON
data = nx.node_link_data(nx_graph)
with open("grafo/graph.json", "w") as file:
    json.dump(data, file)


