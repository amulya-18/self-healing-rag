# Self-Healing RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that checks its own answers for
faithfulness to the source document, and automatically retries with corrective
feedback when it detects a hallucination or unsupported claim.

## Why "self-healing"?

Most RAG demos stop at "retrieve chunks → generate an answer." This project adds
a verification and correction loop:

1. Retrieve relevant chunks from a FAISS vector index
2. Generate an answer using only those chunks
3. Run a second LLM call as a **faithfulness judge** — does every claim in the
   answer actually appear in the retrieved context?
4. If the judge flags a problem, the failure reason is fed back into a new
   generation attempt (not just "try again," but "here's specifically what you
   got wrong, don't repeat it")
5. Retry up to a fixed number of times, widening retrieval each time, and return
   the best-scoring answer

## Real example caught during development

**Question:** "How many projects are there in the document?"

**Attempt 1** (initial generation):
> There are 3 projects explicitly mentioned in the document:
> 1. AI Code Sandbox
> 2. MedAssist – AI-Driven Emergency Assistant
> 3. (The third project is not explicitly named, but it is mentioned as
> "Projects" with a description of a full-stack AI-powered code editor...)

**Faithfulness judge:** Score 0.5 — *"The answer includes a third project that
is not explicitly named in the CONTEXT, violating the strict fact-checking
criteria."*

The system detected that the LLM had double-counted a detail belonging to
"AI Code Sandbox" as a separate, unnamed third project — a hallucinated item
not actually present in the source document.

**Attempt 2** (retry, with the rejection reason fed back into the prompt):
> There are 2 projects explicitly mentioned in the document:
> 1. AI Code Sandbox
> 2. MedAssist – AI-Driven Emergency Assistant

**Faithfulness judge:** Score 1.0 — *"The answer accurately lists the two
projects explicitly mentioned in the context."*

This demonstrates the core value of the self-healing loop: the same LLM, given
specific feedback about its own error, corrected itself without any human
intervention.

## Architecture

```
docs/ (source PDFs/text)
    -> ingest.py: chunk + embed (local, free) -> faiss_index/
                                                       |
question --------------------------------------------> |
                                                       v
                                          retrieve top-k chunks
                                                       v
                                          generate_answer() [Groq LLaMA]
                                                       v
                                          check_faithfulness() [LLM judge]
                                                       v
                                  score >= threshold? --no--> retry with feedback
                                          |
                                         yes
                                          v
                                   return final answer
```

## Tech stack

- **LangChain / LangGraph** — orchestration
- **FAISS** — local vector index for similarity search
- **sentence-transformers (all-MiniLM-L6-v2)** — local, free embeddings
- **Groq (LLaMA 3.1 8B Instant)** — free-tier LLM for both generation and judging
- **Python** — implementation language

## Files

- `ingest.py` — loads documents from `docs/`, chunks them, builds and saves a
  FAISS index
- `query.py` — basic retrieve-and-generate script (no self-healing)
- `self_healing.py` — full pipeline with faithfulness checking and automatic
  feedback-driven retries
- `generate_test_set.py` — automatically generates a test set of question/answer
  pairs from whatever document is currently ingested (works for any document,
  not just one specific file)
- `evaluate.py` — runs the test set through the self-healing pipeline, records
  faithfulness scores and retry counts, and saves results to `eval_results.csv`

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install langchain langchain-community langchain-groq faiss-cpu sentence-transformers pypdf python-dotenv

# Add a .env file with:
# GROQ_API_KEY=your_key_here

# Add documents to docs/, then:
python ingest.py
python self_healing.py

# To evaluate against an auto-generated test set for your document:
python generate_test_set.py
python evaluate.py
```

## Evaluation results

Running `generate_test_set.py` auto-generates questions directly from whatever
document is currently indexed, so the eval set is never hardcoded to one
specific file. In one test run (5 auto-generated factual questions), the
pipeline scored a perfect 1.0 average faithfulness with zero retries needed.

This makes sense: auto-generated questions tend to be direct factual lookups
("What is the candidate's CGPA?") that the LLM answers correctly on the first
try. The self-healing/retry mechanism shows its real value on harder,
**aggregation-style questions** (e.g. "how many projects are mentioned?") that
require correctly combining information across multiple chunks — exactly the
kind of question that exposed the original hallucination documented above.
A natural extension would be generating harder, multi-chunk aggregation
questions specifically to stress-test the self-healing loop further.

## Lessons learned

- **Retrieval and generation failures look similar but need different fixes.**
  An early bug looked like a generation hallucination but was actually caused
  by `k` (chunks retrieved) being too low — the relevant chunk simply wasn't
  retrieved at all.
- **LLM judges need calibration too.** An early version of the faithfulness
  judge scored a clearly flawed answer (with a fabricated "unnamed third
  project") at 0.8 — above the original 0.7 pass threshold. The judge prompt
  was tightened to explicitly penalize vague/unnamed inferred claims, and the
  threshold was raised to 0.9.
- **"Retry" alone doesn't fix repeated mistakes.** Simply re-running generation
  with a stricter prompt caused the model to repeat the exact same error three
  times in a row. The fix was passing the judge's specific rejection reason
  back into the next generation attempt, which broke the loop immediately.

## Next steps

- Build a visual dashboard (charts/table) on top of `eval_results.csv`
- Generate harder, multi-chunk aggregation questions to better stress-test
  the self-healing loop
- Add retrieval precision/recall metrics, not just generation faithfulness
- Deploy as a small web app (FastAPI backend + simple frontend)
