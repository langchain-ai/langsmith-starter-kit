#!/usr/bin/env python3
"""Generate the Finance QA synthetic question bank (data/traces/synthetic_question_bank.csv).

Reads the KB from data/traces/ground_truth_kb.csv and produces three question categories:
  - in_scope:       variations of existing KB questions (answerable by the agent)
  - irrelevant_match: banking topics that match on generic terms but aren't in the KB
  - out_of_scope:   completely unrelated topics (no KB match expected)

Run from the starter-kit root:

    python src/finance_qa/scripts/generate_synthetic_question_bank.py
    python src/finance_qa/scripts/generate_synthetic_question_bank.py --num-questions 50 --seed 42
"""
import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langsmith import tracing_context

load_dotenv()

_DATA_DIR = Path(__file__).parent.parent / "data" / "traces"
_KB_PATH = _DATA_DIR / "ground_truth_kb.csv"
_OUTPUT = _DATA_DIR / "synthetic_question_bank.csv"

_llm = init_chat_model("openai:gpt-4.1-mini", temperature=0.7)


# Banking topics that semantically match KB terms but aren't covered by the KB
IRRELEVANT_MATCH_TOPICS = [
    {"category": "savings_accounts", "examples": [
        "What's the interest rate on my savings account?",
        "How do I open a high-yield savings account?",
        "Can I transfer money from savings to checking?",
        "What's the minimum balance for a savings account?",
        "Do you offer money market savings accounts?",
    ]},
    {"category": "debit_cards", "examples": [
        "How do I activate my new debit card?",
        "What's the daily withdrawal limit on my debit card?",
        "Can I use my debit card internationally?",
        "How do I report a lost debit card?",
        "What are the ATM fees for my debit card?",
    ]},
    {"category": "wire_transfers", "examples": [
        "How long does a wire transfer take?",
        "What are the fees for domestic wire transfers?",
        "Can I cancel a wire transfer after it's sent?",
        "What information do I need to receive a wire transfer?",
        "Are there daily limits on wire transfers?",
    ]},
    {"category": "checks_deposits", "examples": [
        "How do I deposit a check using the mobile app?",
        "How long does it take for a check to clear?",
        "Can I order new checks through online banking?",
        "What's the hold policy for large checks?",
        "How do I stop payment on a check?",
    ]},
    {"category": "atm_services", "examples": [
        "Where's the nearest ATM location?",
        "Can I deposit cash at any ATM?",
        "What's my daily ATM withdrawal limit?",
        "How do I avoid ATM fees?",
    ]},
    {"category": "account_alerts", "examples": [
        "How do I set up low balance alerts?",
        "Can I get text notifications for transactions?",
        "How do I change my alert preferences?",
        "What types of account alerts are available?",
    ]},
    {"category": "mobile_app", "examples": [
        "How do I reset my mobile banking password?",
        "Is the mobile app available for Android?",
        "Can I pay bills through the mobile app?",
        "How do I enable biometric login on the app?",
    ]},
    {"category": "branch_services", "examples": [
        "What are your branch hours?",
        "Do I need an appointment to visit a branch?",
        "Where's the closest branch to me?",
        "What services are available at the branch?",
    ]},
]

# Topics completely outside the credit card support KB
OUT_OF_SCOPE_TOPICS = [
    {"category": "cryptocurrency", "examples": [
        "What's your policy on Bitcoin mining rewards for cardholders?",
        "Can I stake Ethereum using funds from my credit card?",
        "Do you support Web3 wallet integration with my account?",
        "How do I set up DeFi protocol payments with my card?",
    ]},
    {"category": "investment_products", "examples": [
        "What's the expense ratio on your index fund options?",
        "Can I set up a Roth IRA with automatic contributions?",
        "How do I rebalance my 401k portfolio through your platform?",
        "What are the margin rates for options trading on your brokerage?",
    ]},
    {"category": "mortgage_loans", "examples": [
        "What's the APR for a 30-year fixed jumbo mortgage?",
        "Can I get pre-approved for a construction loan?",
        "Do you offer reverse mortgages for seniors?",
        "What are the closing costs for an FHA refinance?",
    ]},
    {"category": "business_banking", "examples": [
        "What's the interest rate on a $500K SBA loan?",
        "Can I get a merchant account for my e-commerce business?",
        "Do you offer factoring services for accounts receivable?",
        "Can I integrate QuickBooks with my business checking account?",
    ]},
    {"category": "international_banking", "examples": [
        "How do I open a multicurrency IBAN account?",
        "What are the SWIFT transfer fees to Asia-Pacific countries?",
        "Do you support SEPA transfers for European payments?",
        "What documentation is needed for cross-border remittances over $10K?",
    ]},
    {"category": "insurance_products", "examples": [
        "What's the death benefit on your whole life insurance policies?",
        "Can I bundle homeowners and auto insurance through your bank?",
        "Do you offer long-term care insurance for policyholders?",
        "What are the premium rates for disability income insurance?",
    ]},
    {"category": "tax_services", "examples": [
        "Can you help me file my taxes through your tax preparation service?",
        "What tax documents do I need for capital gains reporting?",
        "Do you offer tax advisory services for high net worth individuals?",
        "What are the tax implications of my IRA withdrawal?",
    ]},
    {"category": "student_loans", "examples": [
        "What are the interest rates on federal student loans?",
        "Can I refinance my student loans with you?",
        "What are the income-driven repayment plan options?",
        "How do I apply for student loan forbearance?",
    ]},
    {"category": "auto_loans", "examples": [
        "What's your current APR for new car loans?",
        "Can I get pre-approved for an auto loan?",
        "What's the maximum loan term for used cars?",
        "How do I refinance my existing car loan?",
    ]},
    {"category": "estate_planning", "examples": [
        "Do you offer trust services for estate planning?",
        "How do I set up a living trust with your wealth management team?",
        "What are the fees for estate administration services?",
        "Do you provide financial power of attorney services?",
    ]},
]


def _load_kb_questions() -> List[Dict[str, str]]:
    with open(_KB_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _generate(prompt: str) -> str:
    with tracing_context(project_name="starter-finance-qa-data"):
        return _llm.invoke([{"role": "user", "content": prompt}]).content.strip()


def _vary_in_scope(original: str) -> str:
    return _generate(f"""Reword this banking/credit card question. Vary the style — sometimes short and direct, sometimes 2-3 sentences with scenario context.

Original: {original}

Requirements:
- Same core question, completely different wording
- Natural and conversational
- Do not add new information

One reworded question:""")


def _vary_other(base: str) -> str:
    return _generate(f"""Reword this banking question with different phrasing:

Original: {base}

Requirements:
- Same information, different words
- Natural, 1-2 sentences

Reworded question:""")


def generate_question_bank(
    num_questions: int,
    in_scope_ratio: float,
    irrelevant_ratio: float,
    out_of_scope_ratio: float,
) -> List[Dict[str, str]]:
    kb = _load_kb_questions()
    n_in = int(num_questions * in_scope_ratio)
    n_irr = int(num_questions * irrelevant_ratio)
    n_out = num_questions - n_in - n_irr

    rows = []

    print(f"Generating {n_in} in-scope questions...")
    pool = random.choices(kb, k=n_in)
    for i, row in enumerate(pool, 1):
        rows.append({
            "id": f"in_scope_{i:04d}",
            "question": _vary_in_scope(row["question"]),
            "original_question": row["question"],
            "relevance": "in_scope",
            "category": "in_scope",
        })
        if i % 10 == 0 or i == n_in:
            print(f"  {i}/{n_in}")

    print(f"Generating {n_irr} irrelevant-match questions...")
    for i in range(1, n_irr + 1):
        topic = random.choice(IRRELEVANT_MATCH_TOPICS)
        base = random.choice(topic["examples"])
        rows.append({
            "id": f"irrelevant_{i:04d}",
            "question": _vary_other(base),
            "original_question": base,
            "relevance": "irrelevant_match",
            "category": topic["category"],
        })
        if i % 10 == 0 or i == n_irr:
            print(f"  {i}/{n_irr}")

    print(f"Generating {n_out} out-of-scope questions...")
    for i in range(1, n_out + 1):
        topic = random.choice(OUT_OF_SCOPE_TOPICS)
        base = random.choice(topic["examples"])
        rows.append({
            "id": f"out_of_scope_{i:04d}",
            "question": _vary_other(base),
            "original_question": base,
            "relevance": "out_of_scope",
            "category": topic["category"],
        })
        if i % 10 == 0 or i == n_out:
            print(f"  {i}/{n_out}")

    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"q_{i+1:04d}"
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate Finance QA synthetic question bank")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--in-scope-ratio", type=float, default=0.4)
    parser.add_argument("--irrelevant-ratio", type=float, default=0.3)
    parser.add_argument("--out-of-scope-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    total = args.in_scope_ratio + args.irrelevant_ratio + args.out_of_scope_ratio
    if abs(total - 1.0) > 0.01:
        parser.error(f"Ratios must sum to 1.0, got {total:.2f}")

    if args.seed is not None:
        random.seed(args.seed)

    rows = generate_question_bank(
        args.num_questions,
        args.in_scope_ratio,
        args.irrelevant_ratio,
        args.out_of_scope_ratio,
    )

    fieldnames = ["id", "question", "relevance", "category", "original_question"]
    with open(_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {k: sum(1 for r in rows if r["relevance"] == k) for k in ("in_scope", "irrelevant_match", "out_of_scope")}
    print(f"\nWrote {len(rows)} questions → {_OUTPUT}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v/len(rows):.0%})")


if __name__ == "__main__":
    main()
