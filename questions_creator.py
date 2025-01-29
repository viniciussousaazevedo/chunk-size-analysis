from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from constants import *

def create_questions(llm, documents, num_questions=10):
    Settings.llm = llm
    # Combine all documents into one string
    full_text = "\n".join([doc.text for doc in documents])
    
    prompt = (
        f"Based on the following text, create {num_questions} distinct questions "
        f"that cover its key points:\n\n{full_text}\n\n. Do not enumarate your questions. Questions:"
    )
    print("Creating questions for evaluation...")
    response = llm.complete(prompt=prompt)
    questions = response.text.split("\n")[2:]
    print("Done!\n")
    return [q.strip() for q in questions if q.strip()]

if __name__ == "__main__":
    reader = SimpleDirectoryReader(CONTENT_FOLDER)
    documents = reader.load_data()
    llm = Groq(model="llama3-70b-8192")
    generated_questions = create_questions(llm, documents)

    for i in range(len(generated_questions)):
        print(f"Q{i+1}: {generated_questions[i]}")