# Project-scoped skills

Each subdirectory here defines a skill available only when Claude Code is
running in this project. Skills live alongside personal user skills in
`~/.claude/skills/` and plugin skills.

## Layout

```
.claude/skills/
└── my-skill/
    ├── SKILL.md        # frontmatter + instructions for Claude
    └── ...             # any helper scripts, templates, prompts
```

## SKILL.md template

```markdown
---
description: One-line description of when to use this skill.
---

# My Skill

Detailed instructions for Claude here. Markdown is fine.
```

## Invoking

In Claude Code: `/my-skill` (the folder name becomes the slash command).

Project-scoped skills take precedence over user-scoped skills with the
same name.
