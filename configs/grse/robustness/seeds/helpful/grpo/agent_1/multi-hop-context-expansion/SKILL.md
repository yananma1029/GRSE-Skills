---
name: multi-hop-context-expansion
description: Forces the retriever to expand grep results to adjacent turns to capture Q&A pairs, session details, and antecedents required for complete answers.
---
## Retrieval Expansion Rules

1. **Q&A Pair Expansion**: When a grep hit contains a direct question (e.g., "How long...", "What is the name..."), the answer is almost always in the **next turn**. 
   - *Action*: Do not stop at the question turn. Retrieve the subsequent turn (e.g., if `[D20:20]` asks "How long?", retrieve `[D20:21]`).
   - *Example*: If `[D20:20]` is "How long have you been doing yoga?", retrieve `[D20:21]` which contains "3 years".   - *Note*: If the answer turn contains bracketed metadata (e.g., `[image: a dog in a park]`), you MUST include the full metadata tag in your final evidence output. Do not truncate to fit word limits.

2. **Session Context Expansion**: When a broad keyword (e.g., "play games", "husband") hits a turn, the specific details are often in adjacent turns within the same session.
   - *Action*: After finding an initial hit (e.g., `[D19:11]` "play games"), immediately grep for the session ID (e.g., `D19:`) or broader activity terms ("together", "outdoors", "detective") to capture follow-ups.
   - *Example*: If `[D19:11]` mentions "play games", retrieve `[D19:13]` ("play detective games together") and `[D19:15]` ("spend time outdoors").
6. **Process Detail Expansion**: When a turn summarizes a process (e.g., "I made a cake", "We went to the beach"), the specific steps or items are in the subsequent turns.
   - *Action*: Grep for list markers or action verbs in the next 2-3 turns of the same session.
   - *Example*: If `[D5:10]` is "I cooked dinner", retrieve `[D5:11]`-`[D5:12]` to find specific dishes (e.g., "spaghetti", "salad").

3. **Antecedent Retrieval**: When a turn contains a reaction or short cue (e.g., "smile", "nice", "that was great"), the object or subject is in the **previous turn**.
   - *Action*: Retrieve the turn immediately preceding the reaction turn.
   - *Example*: If `[D8:23]` is "smile to face", retrieve `[D8:22]` to identify the object (e.g., "photo").

4. **Named Entity Targeting**: For questions about specific names or titles (e.g., "What is the project called?"), grep for the specific identifier words ("called", "named") or the entity itself in follow-up turns, not just the introductory turn.
   - *Action*: If `[D9:1]` is an introduction, grep for `called` or `name` in the same session to find `[D9:3]`.

5. **Temporal Abstraction**: If a specific date (e.g., "next year", "2024") yields no results, grep for semantic equivalents: "future", "planning", "aiming", "soon".

## Tool Usage

- Always use `<grep>PATTERN</grep>` to search.
- If the first grep returns only a question, run a second grep for the answer content or the session ID.
- **Do NOT** output the grep command itself as the final answer. Extract the specific turn lines.

## Do NOT emit

- `NO_EVIDENCE` (if relevant session turns exist but were not fully retrieved)
- The raw grep command string (e.g., `grep "surf"`)
- `not mentioned` (if adjacent turns contain the answer)
- Truncated excerpts (provide the full turn if the answer is in the second or third sentence)
- Truncated or omitted bracketed metadata (e.g., `[image: ...]`, `[video: ...]`, `[audio: ...]`). These often contain the specific answer (e.g., object identification, date). NEVER drop them to save space.
- `Answer:` prefix (output raw evidence lines)
