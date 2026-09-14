"""Prompt templates.

The judge prompts demand a bare number and nothing else. That constraint is what
makes a reply either parseable or an explicit failure, instead of prose that a
lenient parser might score generously.
"""

from __future__ import annotations

from typing import Final

RESPONSE_SYSTEM_MESSAGE: Final = """\
You answer questions using only the context provided.

Rules:
- Use only the supplied context. Do not draw on outside knowledge.
- If the context does not contain the answer, say so plainly.
- Be specific and concise.
"""

ANSWER_TEMPLATE: Final = """\
{system_message}

Context:
{context}

Question: {question}

Answer:"""

QUERY_EXPANSION_TEMPLATE: Final = """\
Rewrite the question below to improve document retrieval.
Add relevant domain terms and likely synonyms. Preserve the original intent.
Return only the rewritten question.

Question: {question}

Rewritten question:"""

GROUNDEDNESS_TEMPLATE: Final = """\
Score how well the response is supported by the context.

1.0 = every claim is directly supported by the context
0.0 = the response contains claims absent from the context

Context:
{context}

Response:
{response}

Reply with a single number between 0 and 1 and nothing else."""

PRECISION_TEMPLATE: Final = """\
Score how directly the response answers the question.

1.0 = fully and directly answers the question as asked
0.0 = does not address the question

Question:
{question}

Response:
{response}

Reply with a single number between 0 and 1 and nothing else."""

RESPONSE_REFINEMENT_TEMPLATE: Final = """\
The response below contains claims the context does not support.
Rewrite it so every claim is supported. Delete anything that is not.
Return only the revised response.

Context:
{context}

Response:
{response}

Revised response:"""

QUERY_REFINEMENT_TEMPLATE: Final = """\
The retrieved context did not support a good answer to the question.
Rewrite the search query to surface more relevant material.
Return only the revised query.

Question: {question}
Current query: {query}

Revised query:"""

EXHAUSTED_RESPONSE: Final = (
    "I could not answer this confidently from the available documents. "
    "Please rephrase the question or consult the source material directly."
)
