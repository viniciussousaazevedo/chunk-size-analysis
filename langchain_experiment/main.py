from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from lc_constants import *

# Load and process the file
def load_and_prepare_docs(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector store
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Load documents and prepare vector store
docs = load_and_prepare_docs(CONTENT_FILE)
vector_store = create_vector_store(docs)

# Initialize chat model
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant. Use the following context to answer questions."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# Define retrieval-augmented generation function
def rag_query(query):
    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    full_query = f"Context:\n{context}\n\nQuestion: {query}"
    return chat.invoke({"text": full_query}).content

# Example query
query = "What is this document about?"
print(rag_query(query))