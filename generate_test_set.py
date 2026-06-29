"""
generate_test_set.py
Automatically generates a test set of question/answer pairs from
whatever document is currently in the FAISS index - works for any
document, not just one specific file.
"""

import json
import random
from langchain_groq import ChatGroq
from self_healing import load_index

OUTPUT_FILE = "test_questions.json"
NUM_QUESTIONS = 5


def get_sample_chunks(vectorstore, num_chunks=NUM_QUESTIONS):
    """Pull a sample of chunks directly from the FAISS index's stored documents."""
    # docstore holds all the chunks that were embedded during ingestion
    all_docs = list(vectorstore.docstore._dict.values())

    if len(all_docs) <= num_chunks:
        return all_docs
    return random.sample(all_docs, num_chunks)


def generate_qa_pair(chunk_text):
    """Ask the LLM to write one question + answer based on this chunk alone."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

    prompt = f"""Based ONLY on the text below, write exactly one clear question
that can be answered using this text, and its correct answer.
The question should be answerable in one short sentence or phrase.

Text:
{chunk_text}

Respond in exactly this format:
QUESTION: <the question>
ANSWER: <the correct answer>
"""

    response = llm.invoke(prompt).content

    question, answer = "", ""
    for line in response.splitlines():
        if line.upper().startswith("QUESTION:"):
            question = line.split(":", 1)[1].strip()
        if line.upper().startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()

    return question, answer


def generate_test_set():
    vectorstore = load_index()
    chunks = get_sample_chunks(vectorstore)

    test_set = []
    for i, chunk in enumerate(chunks):
        print(f"Generating question {i + 1}/{len(chunks)}...")
        question, answer = generate_qa_pair(chunk.page_content)
        if question and answer:
            test_set.append({"question": question, "expected_answer": answer})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)

    print(f"\nGenerated {len(test_set)} test questions -> saved to {OUTPUT_FILE}")
    return test_set


if __name__ == "__main__":
    generate_test_set()
