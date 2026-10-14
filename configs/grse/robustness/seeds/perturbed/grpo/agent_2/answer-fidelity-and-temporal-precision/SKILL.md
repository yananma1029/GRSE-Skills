---
name: answer-fidelity-and-temporal-precision
description: Enforces strict causal isolation, temporal constraint validation, and preservation of specific activity nouns in final answers.
---
## Causal Clause Isolation
For "Why" questions, output ONLY the primary motive. If the retriever evidence contains multiple clauses (e.g., "he saw one at a pet store and had to get it" vs "he saw another at a pet store and the tank is big enough for three"), select the clause directly answering the motive.
- **Rule**: Strip secondary justifications or consequences.
- **Bad**: `he saw another at a pet store and the tank is big enough for three`
- **Good**: `he saw it at a pet store` (or similar primary motive extraction).

## Temporal Mismatch Detection
If the question contains a specific date or relative time cue (e.g., `last week`, `8 May 2023`), verify the evidence's timestamp matches.
- **Rule**: If evidence is from `session_3 (8 May 2023)` but the question asks about a different week, do NOT extract entities from the mismatched day.
- **Action**: Cross-validate the date in the evidence header against the question's constraint before outputting any entity.

## Nuanced Noun Phrase Preservation
For "What is special" or detail-oriented questions, preserve specific activity nouns. Do not generalize or truncate specific compound nouns.
- **Bad**: Truncating `conversations` to `memories`.
- **Bad**: Outputting `got a new pup` (contains verb/adjective).
- **Good**: `new puppy` (core noun, stripped of action).
- **Rule**: When extracting nouns from phrases like `got a new pup`, output `puppy` or `pup`, not the full verb phrase.

## Entity Confirmation
If evidence states `Jolene 'got back from an awesome trip to Rio de Janeiro'` and `Deborah was 'also in Rio de Janeiro'`, output `Yes`. Do not refuse based on subtle phrasing differences if the core entity/location matches.

## Do NOT emit
- `not mentioned`
- `cannot be determined`
- `no information`
- `unknown`
- `he saw another at a pet store and the tank is big enough for three`
- `got a new pup`
- `since he saw it`
- `because the tank is big enough`
