"""
╔══════════════════════════════════════════════════════════╗
║  STEP 4: Generation — The "G" in RAG                     ║
║                                                          ║
║  Retrieved Context + User Question ──► LLM ──► Answer    ║
║                                                          ║
║  We take the chunks retrieved in Step 3 and pass them    ║
║  to Claude as context. The LLM generates an answer       ║
║  GROUNDED in the retrieved information.                  ║
║                                                          ║
║  Without RAG: LLM answers from training data (may        ║
║               hallucinate or be outdated)                 ║
║  With RAG:    LLM answers from YOUR data (grounded,      ║
║               accurate, up-to-date)                      ║
╚══════════════════════════════════════════════════════════╝
"""

import anthropic


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Only use information from the provided context to answer.
2. If the context doesn't contain the answer, say "I don't have enough information in the knowledge base to answer that."
3. Keep answers concise and clear.
4. Cite which part of the context your answer comes from."""


def generate_answer(question: str, context: str, api_key: str) -> dict:
    """
    Send the question + retrieved context to Claude and get an answer.

    This is the simplest possible generation step:
    - System prompt tells Claude to use only the context
    - User message contains the context + question
    - Claude generates a grounded answer
    """
    client = anthropic.Anthropic(api_key=api_key)

    user_message = f"""Here is the relevant context from the knowledge base:

---CONTEXT START---
{context}
---CONTEXT END---

Question: {question}

Please answer based on the context above."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    return {
        "answer": answer,
        "model": "claude-sonnet-4-20250514",
        "context_used": context,
        "prompt": user_message,
    }
