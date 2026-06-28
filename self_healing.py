"""
self_healing.py
A RAG pipeline that checks its own answers for faithfulness to the
retrieved context, and automatically retries if the check fails.
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

INDEX_FOLDER = "faiss_index"
MAX_RETRIES = 2


def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)


def retrieve_chunks(vectorstore, question, k=5):
    return vectorstore.similarity_search(question, k=k)


def generate_answer(question, chunks, strict=False, previous_failure=None):
    """Generate an answer. If strict=True, use a tighter prompt (used on retry).
    previous_failure, if given, is (previous_answer, reason) from a failed attempt,
    which gets fed back in so the model doesn't repeat the same mistake."""
    context_text = "\n\n".join([c.page_content for c in chunks])

    base_rules = """Answer the question using ONLY the context below.
Do not infer or assume information that is not explicitly stated.
If listing items, only include items explicitly and clearly named as such.
If the answer is not in the context, say "I don't know based on the provided context."
"""

    strict_rules = """Be extremely conservative. Only state facts that are
word-for-word supported by the context. Do not speculate, do not say "however"
or "but if we consider" - give one final, clean answer only.
"""

    feedback_block = ""
    if previous_failure:
        prev_answer, reason = previous_failure
        feedback_block = f"""
Your previous attempt was REJECTED for this reason: {reason}
Your previous (wrong) answer was: {prev_answer}
Do not repeat this exact mistake. Re-read the context carefully and fix it.
"""

    rules = base_rules + (strict_rules if strict else "") + feedback_block

    prompt = f"""{rules}

Context:
{context_text}

Question: {question}

Answer:"""

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke(prompt)
    return response.content


def check_faithfulness(question, answer, chunks):
    """
    Use the LLM as a judge: does the answer only contain claims
    supported by the context? Returns (score, explanation).
    Score is 0-1, where 1 = fully faithful.
    """
    context_text = "\n\n".join([c.page_content for c in chunks])

    judge_prompt = f"""You are a strict fact-checker. Given a CONTEXT and an ANSWER,
determine if every claim in the ANSWER is explicitly supported by the CONTEXT.

Important: if the ANSWER includes any item, entity, or claim that is not explicitly
and clearly named in the CONTEXT (including vague references like "an unnamed project"
or "a third item, not explicitly named"), treat this as a SIGNIFICANT violation and
score 0.5 or lower, even if most of the rest of the answer is accurate.

CONTEXT:
{context_text}

ANSWER:
{answer}

Respond in exactly this format:
SCORE: <a number between 0 and 1, where 1 means fully supported and 0 means not supported at all>
REASON: <one short sentence explaining the score>
"""

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke(judge_prompt).content

    # Parse the score out of the response
    score = 0.0
    reason = response
    for line in response.splitlines():
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                score = 0.0
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score, reason


def self_healing_query(vectorstore, question, faithfulness_threshold=0.9):
    """
    Full self-healing loop:
    1. Retrieve chunks and generate an answer.
    2. Check faithfulness.
    3. If below threshold, retry with a stricter prompt (up to MAX_RETRIES).
    4. Return the best answer found, with its score.
    """
    k = 5
    attempt = 0
    best_answer = None
    best_score = -1
    best_reason = ""
    previous_failure = None

    while attempt <= MAX_RETRIES:
        chunks = retrieve_chunks(vectorstore, question, k=k)
        strict = attempt > 0  # use stricter prompt on retries
        answer = generate_answer(question, chunks, strict=strict, previous_failure=previous_failure)
        score, reason = check_faithfulness(question, answer, chunks)

        print(f"\n[Attempt {attempt + 1}] Faithfulness score: {score} | Reason: {reason}")

        if score > best_score:
            best_answer, best_score, best_reason = answer, score, reason

        if score >= faithfulness_threshold:
            break  # good enough, stop retrying

        # Self-healing action: widen retrieval, get stricter, and tell the model what it got wrong
        previous_failure = (answer, reason)
        k += 2
        attempt += 1

    return best_answer, best_score, best_reason


if __name__ == "__main__":
    vectorstore = load_index()
    print("Self-healing RAG ready. Ask a question (or type 'exit' to quit).\n")

    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            break

        answer, score, reason = self_healing_query(vectorstore, question)

        print(f"\n--- Final Answer (faithfulness: {score}) ---")
        print(answer)
        print()
