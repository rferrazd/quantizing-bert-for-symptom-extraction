---
name: handoff-prompt
description: Generate a self-contained handoff prompt to start a new Claude Code session. Captures the overarching goal, what was accomplished this session, the main files involved, open questions, and concrete next steps — everything a fresh session needs to pick up without re-reading the history.
---

# handoff-prompt

Use this skill when the user asks to "write a handoff prompt", "create a handoff for the next session", or "summarize what to tell the next Claude". The output is a ready-to-paste prompt, not a summary for the current user — it is addressed to the next Claude instance.

## Goal

Produce a prompt that is **self-contained**: a fresh Claude session reading it should understand the project, know exactly where things stand, and be able to start the next task without asking clarifying questions.

## Structure

Write the handoff prompt in this order:

### 1. Project overview (2–4 sentences)
What is this project? What problem is it solving? What is the current version / milestone?
Keep it factual, no history lesson.

### 2. Overarching goal of the session just completed
One sentence: what was the session trying to accomplish overall?

### 3. What was accomplished
Bullet list. Each bullet: file or component changed + what changed and why. Be specific — include file paths, function names, constants, and actual numbers where relevant. Do not pad with vague descriptions.

### 4. Key decisions made
Short bullets for non-obvious choices that the next session must respect (e.g. "train HDA K=40, not K=None — user explicitly chose this"). Include the reason if known.

### 5. Main files to know
A table or short list: file path → one-line description of its role. Only files that are directly relevant to next steps.

### 6. Current state / what exists on disk
Concrete facts: what files exist, what counts are confirmed, what tests pass. This is the "ground truth" the next session can rely on without re-running anything.

### 7. Next steps (ordered)
Numbered list. Each step should be concrete enough to act on immediately. If a step has sub-tasks, list them. Flag any open questions or decisions the next session will need to make.

### 8. Warnings / gotchas
Anything that would waste time if discovered late: naming conventions, non-obvious constraints, things that look wrong but are intentional.

## Tone and format

- The prompt is addressed to Claude, not to the user. Write "You are continuing work on..." not "We were working on..."
- Use markdown. Headings, bullets, tables, code blocks where helpful.
- Be precise and dense. This is a technical briefing, not a narrative.
- Do NOT summarize the conversation history. Summarize the **state of the codebase and the task**.
- Length: as long as needed to be self-contained, but no padding. Typical length: 400–800 words.

## Workflow

1. Ask the user: "What are the next steps for the new session?" if they have not already stated them.
2. Review the session context: what files were touched, what decisions were made, what the final state is.
3. Draft the handoff prompt following the structure above.
4. Show it to the user for review before finalizing.

## Anti-patterns to reject

- Vague bullets like "updated dataset generator" — always specify what changed and why.
- Missing file paths — the next session needs exact paths, not descriptions.
- Omitting key decisions — if a non-obvious choice was made (e.g. a specific K value, a seed, a label scheme), it must appear in "Key decisions".
- Writing it as a summary for the current user instead of a briefing for the next Claude.
- Including conversation history or back-and-forth — only the final state matters.
