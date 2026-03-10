"""
Knowledge Base Retrieval Tools

Simple TF-IDF-based retrieval from synthetic dataset.
"""
import csv
from pathlib import Path
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.tools import tool

# Cache for loaded data and vectorizer
_KB_DATA = None
_VECTORIZER = None
_TFIDF_MATRIX = None


def _load_kb_data() -> List[Dict]:
    """Load knowledge base data from CSV (cached)."""
    global _KB_DATA
    if _KB_DATA is None:
        kb_path = Path(__file__).parent.parent / "data" / "traces" / "ground_truth_kb.csv"
        with open(kb_path, encoding="utf-8") as f:
            _KB_DATA = list(csv.DictReader(f))
        print(f"Loaded {len(_KB_DATA)} KB entries")
    return _KB_DATA


def _initialize_vectorizer():
    """Initialize TF-IDF vectorizer and compute document vectors (cached)."""
    global _VECTORIZER, _TFIDF_MATRIX

    if _VECTORIZER is None:
        data = _load_kb_data()

        documents = []
        for row in data:
            text = f"{row['question']} {row['retrieved_chunks']}"
            documents.append(text)

        _VECTORIZER = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1
        )
        _TFIDF_MATRIX = _VECTORIZER.fit_transform(documents)

    return _VECTORIZER, _TFIDF_MATRIX


def search_knowledge_base(query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict]:
    """Search knowledge base using TF-IDF similarity."""
    data = _load_kb_data()
    vectorizer, tfidf_matrix = _initialize_vectorizer()

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score >= min_similarity:
            results.append({
                'question': data[idx]['question'],
                'retrieved_chunks': data[idx]['retrieved_chunks'],
                'answer': data[idx]['answer'],
                'similarity_score': float(score)
            })

    return results


def get_article_by_topic(topic: str) -> Optional[Dict]:
    """Get a specific article/entry by topic match."""
    data = _load_kb_data()

    for row in data:
        if row['question'].lower() == topic.lower():
            return {
                'question': row['question'],
                'retrieved_chunks': row['retrieved_chunks'],
                'answer': row['answer']
            }

    for row in data:
        if topic.lower() in row['question'].lower():
            return {
                'question': row['question'],
                'retrieved_chunks': row['retrieved_chunks'],
                'answer': row['answer']
            }

    return None


def list_available_topics(category: Optional[str] = None) -> List[str]:
    """List all available topics in the knowledge base."""
    data = _load_kb_data()
    topics = [row['question'] for row in data]

    if category:
        topics = [t for t in topics if category.lower() in t.lower()]

    return sorted(set(topics))


@tool
def search_kb_tool(query: str, num_results: int = 3) -> str:
    """
    Search the knowledge base for relevant information.

    Use this tool to find answers to customer questions about banking services,
    credit cards, payments, disputes, fraud protection, and other banking topics.

    Args:
        query: The customer's question or search query
        num_results: Number of results to return (default: 3, max: 10)

    Returns:
        Formatted search results with relevant information
    """
    num_results = min(num_results, 10)
    results = search_knowledge_base(query, top_k=num_results, min_similarity=0.05)

    if not results:
        return f"No relevant information found for: {query}\nTry rephrasing your query or searching for related topics."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n--- Result {i} (relevance: {result['similarity_score']:.2f}) ---")
        output.append(f"Topic: {result['question']}")
        output.append(f"\nAnswer: {result['answer']}")

        chunks = result['retrieved_chunks'].split('\n\n')
        if chunks:
            output.append(f"\nDetailed Procedures:")
            for chunk in chunks[:3]:
                if chunk.strip():
                    output.append(f"  • {chunk.strip()[:300]}...")

    return "\n".join(output)


@tool
def get_topic_details(topic: str) -> str:
    """
    Get detailed information about a specific topic.

    Use this tool when you need complete procedural details about a specific
    banking topic or process.

    Args:
        topic: The specific topic to retrieve

    Returns:
        Complete information about the topic
    """
    article = get_article_by_topic(topic)

    if not article:
        results = search_knowledge_base(topic, top_k=3)
        if results:
            similar_topics = [r['question'] for r in results]
            return f"Topic '{topic}' not found.\n\nDid you mean one of these?\n" + "\n".join(f"  - {t}" for t in similar_topics)
        return f"Topic '{topic}' not found in knowledge base."

    output = []
    output.append(f"Topic: {article['question']}")
    output.append(f"\n{'='*60}")
    output.append(f"Summary: {article['answer']}")
    output.append(f"\n{'='*60}")
    output.append(f"Detailed Information:\n\n{article['retrieved_chunks']}")

    return "\n".join(output)


@tool
def list_topics(category: Optional[str] = None) -> str:
    """
    List available topics in the knowledge base.

    Use this tool to discover what topics are available to search.

    Args:
        category: Optional category filter (e.g., "payment", "dispute", "fraud")

    Returns:
        List of available topics
    """
    topics = list_available_topics(category)

    if not topics:
        return f"No topics found{' for category: ' + category if category else ''}."

    output = [f"Available topics{' in category: ' + category if category else ''}:\n"]
    for topic in topics:
        output.append(f"  • {topic}")

    output.append(f"\nTotal: {len(topics)} topics")

    return "\n".join(output)
