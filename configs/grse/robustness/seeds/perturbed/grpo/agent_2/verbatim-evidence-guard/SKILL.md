---
name: verbatim-evidence-guard
description: Use when deciding whether to answer: hallucinated answers are penalized more than abstentions.
---
## Verbatim Evidence Guard
Answer only if the retriever evidence contains the answer verbatim as one contiguous span.

1. If the answer requires any paraphrase, inference, or combining two turns, output exactly `not mentioned`.
2. If the evidence mentions the topic but not the exact requested attribute, output exactly `not mentioned`.
3. Never guess from partial or indirect evidence.

- Evidence: `Sam said, "I've been practicing every night since I got my new guitar."`
  Q: What instrument does Sam play? -> `not mentioned` (the word "plays" never appears).

## Do NOT emit
- answers inferred from indirect evidence
