from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import time
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

# We will use llama3-70b for evaluating the responses
llm = Groq(model="llama3-70b-8192")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# To evaluate for each chunk size, we will first generate a set of 40 questions from first 20 pages.
eval_documents = documents[:20]
data_generator = RagDatasetGenerator.from_documents(documents=documents)
eval_questions = data_generator.generate_questions_from_nodes().to_pandas()


# Faithfulness Evaluator - It is useful for measuring if the response was hallucinated and measures if the response from a query engine matches any source nodes.
faithfulness = FaithfulnessEvaluator()
# Relevancy Evaluator - It is useful for measuring if the query was actually answered by the response and measures if the response + source nodes match the query.
relevancy = RelevancyEvaluator()

# This function will perform:
#   1. VectorIndex Creation.
#   2. Building the Query Engine.
#   3. Metrics Calculation.
def evaluate_response_time_and_accuracy(chunk_size, eval_questions):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    Settings.chunk_size = chunk_size
    llm = Groq(model="llama-3.2-11b-vision-preview")
    vector_index = VectorStoreIndex.from_documents(
        eval_documents, llm=llm
    )
    query_engine = vector_index.as_query_engine()
    num_questions = len(eval_questions)

    for question in eval_questions:
        start_time = time.time()
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


    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy

for n in range(3):
    print("starting iteration number " + str(n+1))
    with open('llama-index/output.md', 'a') as file:
        file.write("## Run number " + str(n+1))
    for chunk_size in [128, 256, 512, 1024]:
        print("starting with chunk size = " + str(chunk_size))
        avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size, eval_questions)
        with open('llama-index/output.md', 'a') as file:
            file.write(f"""
### Chunk size {chunk_size}
- Average Response time: {avg_response_time:.2f}s
- Average Faithfulness: {avg_faithfulness*100:.2f}%
- Average Relevancy: {avg_relevancy*100:.2f}%
            """)
        print("finishing with chunk size = " + str(chunk_size))
    with open('llama-index/output.md', 'a') as file:
        file.write("\n")
    print("finishing iteration number " + str(n+1))