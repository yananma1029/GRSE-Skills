# Case studies

The trajectories and skill libraries behind the case studies of the paper's appendix (all
except the illustrative example in the framework figure). They come from the paper's own
runs; new runs give other cases, which `analysis/learned_skills.py`, `analysis/robustness.py`
and `analysis/rerun_question.py` help to find and inspect.

A saved trajectory (`trajectory_*.json`, `*_final_trajectory.json`) is one test episode as the
run wrote it: `task` (question, gold answer, the conversation), `transitions` (per agent: system
prompt with its skills, input, output, and the retriever's grep calls in `grep_log`), `reward`,
`metadata.final_answer` and `metadata.qa_metrics` (official F1 and BLEU). Files named
`*agent_outputs.json` hold the agent outputs of a question answered again with
`analysis/rerun_question.py` (greedy), for libraries whose evaluation keeps only the final answer.

## Learned skills (Appendix, "Learned Skill Cases")

| Case | Setting | Files |
|---|---|---|
| A: better retrieval, no refusal | Gemma4 actor, LoCoMo single-hop, GRSE, first run; question `conv-43__qa_0111` | `learned_case_A/trajectory_empty_library.json` (iteration 0), `trajectory_final_library.json` (iteration 5), `final_library/` |
| B: a more specific answer | Qwen3.8 actor, HotpotQA, GRSE, first run; question `qa_00150` | `learned_case_B/agent_outputs.json` (empty and final library, both agents; with the scores of the 500-question evaluation), `final_library/` |

## Robustness (Appendix, "Robustness Analysis")

Runs: GRSE and MASkills, both started from the perturbed library in
`configs/grse/robustness/seeds/perturbed/` (LoCoMo single-hop, Qwen3.8 actor, one run each).
`robustness/grse_libraries/iter_<k>/` and `robustness/maskills_libraries/iter_<k>/` hold both
agents' libraries after every iteration (iteration 0 is the perturbed library); each skill
carries its stored `utility`.

| Case | Files |
|---|---|
| 1: the harmful retrieval rule `focused-latest-session-retrieval`; questions `conv-47__qa_0149` (shown), `conv-43__qa_0129`, `conv-47__qa_0100` (mentioned) | `robustness/case_1/<question>/perturbed_library_agent_outputs.json`, `grse_final_trajectory.json`, `maskills_final_trajectory.json`; the merged skills in `robustness/*_libraries/iter_2/agent_1/` |
| 2: the harmful abstention rule `verbatim-evidence-guard` | `robustness/case_2/grse_critic_credits_verbatim-evidence-guard.json` (every credit GRSE's critic gave the skill: iteration, question group, rollout rewards, label, evidence); its utility after each iteration in `robustness/*_libraries/iter_*/agent_2/skill_library.json` |
