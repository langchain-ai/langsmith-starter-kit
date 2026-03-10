"""Finance QA prompts — builds and uploads evaluation prompts to LangSmith."""
from pydantic import BaseModel, Field
from langchain_core.prompts.structured import StructuredPrompt

from utils.prompts import load_prompt, build_schema


class Helpfulness(BaseModel):
    helpfulness: bool = Field(description="Is the response helpful and accurately addresses the customer's question?")
    comment: str = Field(description="Reasoning for the helpfulness score")


class RagCitationQuality(BaseModel):
    rag_citation_quality: float = Field(description="Citation quality score from 0.0 to 1.0")
    comment: str = Field(description="Reasoning for the citation quality score")


class AnswerCorrectness(BaseModel):
    answer_correctness: float = Field(description="Answer correctness score from 0.0 to 1.0")
    comment: str = Field(description="Reasoning for the answer correctness score")


def load_helpfulness_prompt():
    system = """You are an expert evaluator assessing the helpfulness of customer service chatbot responses.

A helpful response:
- Directly and completely addresses the customer's question
- Is accurate and consistent with the reference answer
- Is clear, concise, and actionable

Score True if the response is genuinely helpful and fully addresses the customer's need; False if it is incomplete, inaccurate, or unhelpful."""

    human = """Please evaluate the following response:

<question>
{input}
</question>

<response>
{output}
</response>

<reference>
{reference}
</reference>"""
    prompt = StructuredPrompt(
        messages=[("system", system), ("human", human)],
        schema_=build_schema(Helpfulness, "helpfulness"),
    )
    return load_prompt("finance-qa-helpfulness-eval", prompt)


def load_rag_citation_prompt():
    system = """You are an expert evaluator assessing citation quality in RAG-based customer service responses.

A well-cited response:
- References relevant KB sources that support the claims made
- Citations are accurate and grounded in the retrieved content
- Key claims are backed by cited sources

Score 1.0 if citations are accurate and comprehensive; 0.0 if citations are absent, fabricated, or inaccurate."""

    human = """Please evaluate the following response:

<question>
{input}
</question>

<response>
{output}
</response>

<expected_sources>
{reference}
</expected_sources>

Provide a rag_citation_quality score from 0.0 to 1.0."""
    prompt = StructuredPrompt(
        messages=[("system", system), ("human", human)],
        schema_=build_schema(RagCitationQuality, "rag_citation_quality"),
    )
    return load_prompt("finance-qa-rag-citation-eval", prompt)


def load_answer_correctness_prompt():
    system = """You are an expert evaluator assessing the factual correctness of customer service chatbot responses.

Score 1.0 if the response is fully correct and consistent with the reference answer; 0.0 if it is completely wrong or missing key information."""

    human = """Please evaluate the following response:

<question>
{input}
</question>

<response>
{output}
</response>

<reference>
{reference}
</reference>

Provide an answer_correctness score from 0.0 to 1.0."""
    prompt = StructuredPrompt(
        messages=[("system", system), ("human", human)],
        schema_=build_schema(AnswerCorrectness, "answer_correctness"),
    )
    return load_prompt("finance-qa-answer-correctness-eval", prompt)


def load_all_prompts() -> dict:
    print("Loading all prompts...")
    results = {
        "helpfulness_eval": load_helpfulness_prompt(),
        "rag_citation_eval": load_rag_citation_prompt(),
        "answer_correctness_eval": load_answer_correctness_prompt(),
    }
    for key, url in results.items():
        print(f"    - {key}: {url if url else 'unchanged'}")
    return results


if __name__ == "__main__":
    load_all_prompts()
