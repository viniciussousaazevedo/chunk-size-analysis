from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from questions_creator import create_questions
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from constants import *
import time
import os

def run():
    if os.path.exists(QA_OUTPUT_FILE):
        os.remove(QA_OUTPUT_FILE)

    reader = SimpleDirectoryReader(QA_CONTEXT_FOLDER)
    documents = reader.load_data()
    eval_llm = Groq(model="llama3-70b-8192")
    eval_questions = create_questions(eval_llm, documents)

    Settings.llm = eval_llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    # Faithfulness Evaluator - It is useful for measuring if the response was hallucinated and measures if the response from a query engine matches any source nodes.
    faithfulness = FaithfulnessEvaluator()
    # Relevancy Evaluator - It is useful for measuring if the query was actually answered by the response and measures if the response + source nodes match the query.
    relevancy = RelevancyEvaluator()

    for n in range(3):
        print("Starting iteration number " + str(n+1))
        with open(QA_OUTPUT_FILE, 'a') as file:
            file.write("## Run number " + str(n+1))
        for chunk_size in [128, 256, 512, 1024]:
            print("\tStarting with chunk size = " + str(chunk_size))
            avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(
                documents,
                chunk_size,
                eval_questions,
                faithfulness,
                relevancy
                )
            with open(QA_OUTPUT_FILE, 'a') as file:
                file.write(f"""
    ### Chunk size {chunk_size}
    - Average Response time: {avg_response_time:.2f}s
    - Average Faithfulness: {avg_faithfulness*100:.2f}%
    - Average Relevancy: {avg_relevancy*100:.2f}%
                """)
            print("\tFinishing with chunk size = " + str(chunk_size))
        with open(QA_OUTPUT_FILE, 'a') as file:
            file.write("\n")
        print("Finishing iteration number " + str(n+1))


# This function will perform:
#   1. VectorIndex Creation.
#   2. Building the Query Engine.
#   3. Metrics Calculation.
def evaluate_response_time_and_accuracy(documents, chunk_size, eval_questions, faithfulness, relevancy):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # Initialize LLM
    qa_llm = Groq(model="llama-3.2-11b-vision-preview")

    # Create a LangchainNodeParser using the given chunk_size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    parser = LangchainNodeParser(text_splitter)

    # Parse the documents using the parser
    nodes = []
    for doc in documents:
        nodes.extend(parser.split_text(doc.text))

    # Create VectorStoreIndex using the parsed nodes
    vector_index = VectorStoreIndex.from_documents(
        [Document(text=node) for node in nodes], llm=qa_llm
    )
    query_engine = vector_index.as_query_engine()

    for question in eval_questions:
        start_time = time.time()
        print("\t\tCurrent question: " + question)
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time

        faithfulness_result = faithfulness.evaluate_response(
            response=response_vector
        ).passing

        relevancy_result = relevancy.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    num_questions = len(eval_questions)
    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy

if __name__ == "__main__":
    run()