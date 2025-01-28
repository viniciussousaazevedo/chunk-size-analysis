from constants import *
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

def load_documents(content_folder):
    pdf_loader = PyPDFLoader(file_path=content_folder)
    documents = pdf_loader.load()
    return documents


def create_questions(llm, documents, num_questions=20):
    # Combine all document content into a single string
    full_text = "\n".join([doc.page_content for doc in documents])
    
    prompt_template = f"""
Based on the following text, create {num_questions} distinct questions
that cover its key points:\n\n{full_text}\n\n. Do not enumarate your questions. Questions:
"""
    prompt=ChatPromptTemplate.from_messages([
        ("system",""),
        ("human",prompt_template)
    ])
    chain = prompt | llm
    response = chain.invoke({"text":prompt_template})
    questions = response.content.split("\n")[2:]
    return [q.strip() for q in questions if q.strip()]

if __name__ == "__main__":
    documents = load_documents(CONTENT_FILE)
    llm=ChatGroq(model="llama3-70b-8192", temperature=0)
    generated_questions = create_questions(llm, documents)
    
    # Print the generated questions
    for i, question in enumerate(generated_questions, 1):
        print(f"Q{i}: {question}")
