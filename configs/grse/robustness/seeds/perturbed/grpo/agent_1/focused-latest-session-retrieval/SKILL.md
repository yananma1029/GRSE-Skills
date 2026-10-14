---
name: focused-latest-session-retrieval
description: Use when retrieving evidence: keeps the grep budget small and avoids stale, superseded facts.
---
## Focused Retrieval Protocol
Long conversations contain outdated statements; the most recent session reflects the speaker's current state.

1. Pick the single most distinctive keyword from the question (one word, no alternation). Example: for
   "What instrument does Sam play?" use `<grep>instrument</grep>`.
2. Issue exactly ONE grep. Additional searches waste budget and pull in contradictory older turns.
3. From the hits, keep only lines from the highest-numbered session; discard all earlier sessions as outdated.
4. Hand off at most two evidence lines.

## Do NOT emit
- more than one `<grep>` call
- evidence lines from sessions older than the latest hit
