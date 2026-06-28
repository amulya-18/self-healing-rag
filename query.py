"""
query.py
Loads the FAISS index, retrieves relevant chunks for a question,
and asks Groq's LLaMA model to answer using only that context.
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from your .env file

INDEX_FOLDER = "faiss_index"


def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True,  # safe here since we created this index ourselves
    )
    return vectorstore


def retrieve_chunks(vectorstore, question, k=3):
    """Return the k most relevant chunks for the question."""
    results = vectorstore.similarity_search(question, k=k)
    return results


def generate_answer(question, chunks):
    """Send the question + retrieved chunks to Groq's LLaMA model."""
    context_text = "\n\n".join([chunk.page_content for chunk in chunks])

    prompt = f"""Answer the question using ONLY the context below.
Do not infer or assume any information that is not explicitly stated in the context.
If you are listing items (like projects, skills, or achievements), only include items that are clearly and explicitly named as such — do not count a detail or bullet point as a separate new item unless it is clearly a distinct, named entry.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    vectorstore = load_index()
    print("Index loaded. Ask a question (or type 'exit' to quit).\n")

    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            break

        chunks = retrieve_chunks(vectorstore, question, k=5)

        print("\n--- Retrieved chunks ---")
        for i, chunk in enumerate(chunks):
            print(f"[{i+1}] {chunk.page_content[:150]}...")

        answer = generate_answer(question, chunks)
        print(f"\n--- Answer ---\n{answer}\n")
