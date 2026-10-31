---
name: new-skill
description: How to write a new project skill for imigraph (a SKILL.md under .claude/skills/) that coding agents load automatically - when a skill is worth writing, its frontmatter, and what belongs in it. Use when a task says "create a skill" or when a workflow had to be rediscovered twice.
---

# Writing a skill for this repository

A skill is a directory `.claude/skills/<name>/SKILL.md`. Claude Code lists every skill's
`description` to the model and loads the body when it matches the task, so the description is
what makes a skill fire.

## When a skill is worth it

- A workflow that is not derivable from the code and was needed more than once (how to run
  things in the container, how to verify a page).
- Project conventions with a *why* (tolerances, seeds, resolution of comparisons).
- Checklists for recurring extension points (new rule, new reference system, new check).

Not for: content that already lives in `CLAUDE.md`, the README or `docs/` (link to it instead);
one-off task notes (those go into `plans/`); API documentation (docstrings do that).

## Format

```markdown
---
name: <kebab-case, same as the directory>
description: <one or two sentences: what it covers AND when to use it; this is the trigger>
---

# Title

Short orientation (what to read first). Then the procedure: numbered steps, commands in fenced
blocks, file paths in backticks, the reason behind each non-obvious rule. End with how to verify.
```

- Keep it under about 120 lines; split otherwise.
- Prefer commands that work in the container (see the `docker-dev` skill).
- Name files and functions that exist; check them before writing (`Glob`, `Grep`).
- If a skill needs supporting files (templates, scripts), put them next to `SKILL.md` and refer
  to them by relative path.

## After writing

1. Add the skill to the list in `README.md` ("Skills for agents").
2. Run `pre-commit run --all-files` in the container (trailing whitespace, end of file).
3. Try it: ask an agent to do the task the skill covers and see whether it reaches for it.
