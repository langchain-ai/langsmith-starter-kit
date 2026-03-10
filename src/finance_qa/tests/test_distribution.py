"""Tests for _sample_with_distribution in setup/traces.py."""
import math
import random
import pytest
from src.finance_qa.setup.traces import _sample_with_distribution


def _make_bank(counts: dict) -> list:
    """Build a fake question bank with the given per-value row counts."""
    rows = []
    for value, n in counts.items():
        for i in range(n):
            rows.append({"question": f"{value}_{i}", "relevance": value, "category": value})
    return rows


# --- basic proportions ---

def test_exact_proportions():
    bank = _make_bank({"in_scope": 40, "irrelevant_match": 30, "out_of_scope": 30})
    dist = {"in_scope": 0.4, "irrelevant_match": 0.3, "out_of_scope": 0.3}
    result = _sample_with_distribution(bank, 10, dist)

    assert len(result) == 10
    counts = {k: sum(1 for r in result if r["relevance"] == k) for k in dist}
    # floor(10*0.4)=4, floor(10*0.3)=3, floor(10*0.3)=3 → sum=10, no topup needed
    assert counts["in_scope"] == 4
    assert counts["irrelevant_match"] == 3
    assert counts["out_of_scope"] == 3


def test_rounding_topup():
    """Ratios that don't divide evenly are topped up to exactly n."""
    bank = _make_bank({"a": 50, "b": 50, "c": 50})
    dist = {"a": 1/3, "b": 1/3, "c": 1/3}
    # Confirm the test precondition: floor sum is actually short
    assert sum(math.floor(10 * r) for r in dist.values()) < 10
    result = _sample_with_distribution(bank, 10, dist)
    assert len(result) == 10


def test_no_duplicates():
    bank = _make_bank({"in_scope": 20, "irrelevant_match": 20, "out_of_scope": 20})
    dist = {"in_scope": 0.4, "irrelevant_match": 0.3, "out_of_scope": 0.3}
    result = _sample_with_distribution(bank, 15, dist)
    assert len({id(r) for r in result}) == len(result)


def test_output_is_shuffled():
    """Categories should be interleaved after shuffle, not grouped."""
    bank = _make_bank({"in_scope": 100, "irrelevant_match": 100, "out_of_scope": 100})
    dist = {"in_scope": 0.4, "irrelevant_match": 0.3, "out_of_scope": 0.3}
    categories = [r["relevance"] for r in _sample_with_distribution(bank, 30, dist)]
    # An unshuffled result would have all same-category rows contiguous.
    # P(first 6 all the same category after shuffle) ≈ 3 * (12/30 * 11/29 * ... * 7/25) ≈ 0.005
    assert len(set(categories[:6])) > 1


# --- arbitrary column ---

def test_arbitrary_column():
    """distribution can stratify by any row field, not just 'relevance'."""
    bank = [
        {"question": f"q{i}", "relevance": "in_scope", "category": cat}
        for i, cat in enumerate(["payments"] * 20 + ["fraud"] * 20 + ["rewards"] * 20)
    ]
    dist = {"payments": 0.5, "fraud": 0.3, "rewards": 0.2}
    result = _sample_with_distribution(bank, 10, dist, column="category")

    assert len(result) == 10
    counts = {k: sum(1 for r in result if r["category"] == k) for k in dist}
    # floor(10*0.5)=5, floor(10*0.3)=3, floor(10*0.2)=2 → sum=10, no topup
    assert counts["payments"] == 5
    assert counts["fraud"] == 3
    assert counts["rewards"] == 2


# --- edge cases ---

def test_small_pool_clamps():
    """A category with fewer rows than requested is clamped; topup fills the gap."""
    bank = _make_bank({"in_scope": 2, "irrelevant_match": 50, "out_of_scope": 50})
    dist = {"in_scope": 0.4, "irrelevant_match": 0.3, "out_of_scope": 0.3}
    result = _sample_with_distribution(bank, 20, dist)
    assert len(result) == 20
    # Can't exceed what's in the pool; topup draws from other categories
    assert sum(1 for r in result if r["relevance"] == "in_scope") <= 2


def test_unknown_category_ignored():
    """A distribution key with no matching rows contributes 0; topup fills the gap."""
    bank = _make_bank({"in_scope": 50})
    dist = {"in_scope": 0.5, "nonexistent": 0.5}
    result = _sample_with_distribution(bank, 10, dist)
    assert len(result) == 10
