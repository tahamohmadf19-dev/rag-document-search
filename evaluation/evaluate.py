"""Retrieval and abstention evaluation on a small synthetic bilingual corpus.

The corpus and questions are invented for this test and are not real documents. Run:
    python evaluation/evaluate.py

Read the results as a regression check, not a quality claim: the set is tiny, and the
abstention threshold has not been validated on separate data.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core import answer_question, ingest_text, retrieve  # noqa: E402
from vector_db import QdrantStorage  # noqa: E402

CORPUS = {
    "handbook.txt": (
        "Employees receive twenty one days of annual leave each calendar year. Unused leave expires at the "
        "end of March. Requests for leave longer than five days need written approval from a direct manager. "
        "Remote work is allowed for up to two days each week after the probation period ends. "
        "The probation period lasts three months for all new staff."
    ),
    "billing.txt": (
        "Invoices are issued on the first business day of each month. Payment is due within thirty days of "
        "the invoice date. A late fee of two percent applies to overdue balances. Refund requests must be "
        "submitted within fourteen days of the original charge."
    ),
    "security.txt": (
        "Passwords must contain at least fourteen characters. Multi factor authentication is required for all "
        "administrator accounts. Security incidents must be reported within twenty four hours to the security "
        "team. Laptop disks must use full disk encryption."
    ),
    "library_ar.txt": (
        "تفتح المكتبة أبوابها من الساعة الثامنة صباحًا حتى السادسة مساءً. يمكن للطالب استعارة خمسة كتب في المرة الواحدة. "
        "مدة الإعارة أسبوعان قابلة للتجديد مرة واحدة. غرامة التأخير نصف دينار عن كل يوم."
    ),
}

ANSWERABLE = [
    ("How many days of annual leave do employees receive?", "handbook.txt", "twenty one"),
    ("When does unused leave expire?", "handbook.txt", "March"),
    ("How long does the probation period last?", "handbook.txt", "three months"),
    ("How many days a week is remote work allowed?", "handbook.txt", "two days"),
    ("When are invoices issued?", "billing.txt", "first business day"),
    ("What is the late fee for overdue balances?", "billing.txt", "two percent"),
    ("How long do I have to submit a refund request?", "billing.txt", "fourteen days"),
    ("How many characters must a password contain?", "security.txt", "fourteen"),
    ("Who needs multi factor authentication?", "security.txt", "administrator"),
    ("When must security incidents be reported?", "security.txt", "twenty four hours"),
    ("كم كتابًا يمكن للطالب استعارته؟", "library_ar.txt", "خمسة"),
    ("ما هي مدة الإعارة؟", "library_ar.txt", "أسبوعان"),
    ("متى تفتح المكتبة أبوابها؟", "library_ar.txt", "الثامنة"),
    ("كم غرامة التأخير؟", "library_ar.txt", "نصف دينار"),
]

UNANSWERABLE = [
    "What is the capital of Australia?",
    "Which programming language is the fastest for games?",
    "How do volcanoes form?",
    "من هو مؤلف رواية البؤساء؟",
    "ما هو طول نهر النيل؟",
]


def run() -> dict:
    store = QdrantStorage(location=":memory:", collection="evaluation")
    for source, text in CORPUS.items():
        ingest_text(text, source, store)

    hits_at_1 = hits_at_3 = 0
    reciprocal = 0.0
    answer_contains = abstained_wrongly = 0
    misses = []
    for question, source, expected in ANSWERABLE:
        found = retrieve(question, store, top_k=3)
        rank = next((i for i, s in enumerate(found["sources"]) if s == source), None)
        if rank == 0:
            hits_at_1 += 1
        if rank is not None:
            hits_at_3 += 1
            reciprocal += 1 / (rank + 1)
        answer = answer_question(question, store)
        if answer.abstained:
            abstained_wrongly += 1
        if expected in answer.text and not answer.abstained:
            answer_contains += 1
        else:
            misses.append(question)

    correct_abstentions = 0
    false_answers = []
    for question in UNANSWERABLE:
        answer = answer_question(question, store)
        if answer.abstained:
            correct_abstentions += 1
        else:
            false_answers.append(question)

    store.close()
    count = len(ANSWERABLE)
    return {
        "embedding": "local hashing of character n-grams (lexical, not semantic)",
        "answerable_questions": count,
        "unanswerable_questions": len(UNANSWERABLE),
        "hit_at_1": round(hits_at_1 / count, 3),
        "hit_at_3": round(hits_at_3 / count, 3),
        "mrr_at_3": round(reciprocal / count, 3),
        "answer_contains_expected": round(answer_contains / count, 3),
        "wrongly_abstained": abstained_wrongly,
        "correct_abstentions": correct_abstentions,
        "false_answers_on_unanswerable": false_answers,
        "questions_answered_incorrectly": misses,
        "caveat": "Synthetic corpus of 4 short documents. The threshold has not been validated on separate data.",
    }


if __name__ == "__main__":
    results = run()
    output = json.dumps(results, ensure_ascii=False, indent=2)
    Path(__file__).with_name("results.json").write_text(output + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(output)
