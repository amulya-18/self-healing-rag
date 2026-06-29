"""
evaluate.py
Runs a test set of questions through the self-healing RAG pipeline,
records faithfulness scores and retry counts, and prints/saves a report.
"""

import json
import csv
from self_healing import load_index, self_healing_query

TEST_SET_FILE = "test_questions.json"
RESULTS_CSV = "eval_results.csv"


def load_test_set():
    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    vectorstore = load_index()
    test_set = load_test_set()

    results = []

    for i, item in enumerate(test_set):
        question = item["question"]
        expected = item.get("expected_answer", "")

        print(f"\n=== Question {i + 1}/{len(test_set)} ===")
        print(f"Q: {question}")

        answer, score, reason = self_healing_query(vectorstore, question)

        print(f"Final answer: {answer}")
        print(f"Faithfulness score: {score}")

        results.append({
            "question": question,
            "expected_answer": expected,
            "final_answer": answer,
            "faithfulness_score": score,
            "judge_reason": reason,
        })

    return results


def save_results_csv(results):
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected_answer", "final_answer", "faithfulness_score", "judge_reason"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to {RESULTS_CSV}")


def print_summary(results):
    scores = [r["faithfulness_score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    low_scores = [r for r in results if r["faithfulness_score"] < 0.9]

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total questions tested: {len(results)}")
    print(f"Average faithfulness score: {avg_score:.2f}")
    print(f"Questions below 0.9 threshold: {len(low_scores)}")

    if low_scores:
        print("\nQuestions that needed attention:")
        for r in low_scores:
            print(f"  - \"{r['question']}\" -> score {r['faithfulness_score']} ({r['judge_reason']})")


if __name__ == "__main__":
    results = run_evaluation()
    save_results_csv(results)
    print_summary(results)
