from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from scripts.ct_gen_prompt import get_prompt
from scripts.constants import *
import json
import os

def run():
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if os.path.exists(CT_GEN_OUTPUT_FILE_PATH):
        os.remove(CT_GEN_OUTPUT_FILE_PATH)
    with open(US_CONTENT_PATH, "r", encoding="utf-8") as json_us:
        us = json.load(json_us)
    
    reader = SimpleDirectoryReader(US_CONTEXT_FOLDER)
    documents = reader.load_data()
    Settings.llm = Groq(model="llama-3.2-11b-vision-preview")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    for chunk_size in CHUNK_SIZES:
        print("Creating test cases with chunk size = " + str(chunk_size))
        create_test_cases(
            documents, 
            chunk_size,
            us['id'],
            us['title'], 
            us['description'],
            us['acceptance_criteria']
        )



def create_test_cases(
        documents,
        chunk_size,
        us_id, 
        us_title,
        us_description, 
        acceptance_criteria
    ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    parser = LangchainNodeParser(text_splitter)
    # Parse the documents using the parser
    nodes = []
    for doc in documents:
        nodes.extend(parser.split_text(doc.text))

    # Create VectorStoreIndex using the parsed nodes
    vector_index = VectorStoreIndex.from_documents(
        [Document(text=node) for node in nodes]
    )
    query_engine = vector_index.as_query_engine()
    response_vector = query_engine.query(get_prompt(us_id, us_title, us_description, acceptance_criteria))
    with open(CT_GEN_OUTPUT_FILE_PATH, 'a') as file:
            file.write("### Chunk size = " + str(chunk_size) + "\n")
            file.write(str(response_vector) + "\n")

if __name__ == "__main__":
    run()