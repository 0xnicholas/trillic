# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — single-context repo (no `CONTEXT-MAP.md`).
- **`docs/decisions.md`** — this repo's decision record (revision-style; the ADR equivalent). **`docs/adr/` is NOT used in this repo** — decisions land as numbered sections in `decisions.md` with dated revision notes.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates/updates them lazily when terms or decisions actually get resolved — new decisions as new numbered sections in `docs/decisions.md`, new terms in `CONTEXT.md`.

## File structure

Single-context repo:

```
/
├── CONTEXT.md            ← ubiquitous language glossary
├── docs/
│   ├── decisions.md      ← decision record (revision-style, ADR equivalent)
│   └── roadmap.md        ← phase structure, gates, stop-loss
└── ...
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag decision conflicts

If your output contradicts an entry in `docs/decisions.md`, surface it explicitly rather than silently overriding:

> _Contradicts decisions.md §4 (roadmap & training gate) — but worth reopening because…_
