# Toolchain state & update log

Checked weekly (or whenever 7 real days have passed since the date below).
Instruments (lm-eval, HRET) are upgraded **only before a scored run** — upgrading
mid-series invalidates that round (lm-eval 0.4.11 explicitly warns results across
task versions are not comparable).

**Last checked: 2026-07-26**

| package | installed | latest stable | notes |
|---|---|---|---|
| lm-eval | **0.4.12** | 0.4.12 | upgraded 2026-07-26 from 0.4.9 — breaking CLI (`lm-eval run`), TaskManager.load() now flat dict |
| haerae-evaluation-toolkit | 0.1.0 | 0.1.0 | current; installed `--no-deps` (its vllm pin does not build on macOS) |
| mlx | 0.32.0 (venvs) / 0.31.2 (system) | 0.32.0 | ml-explore/mlx#3836 (the >2³¹ strided-copy corruption behind our broken 8-bit) CLOSED 2026-07-12 |
| mlx-lm | 0.31.3 | 0.31.3 | current; we serve from forks, not the release |
| litellm | 1.93.0 | 1.93.0 | current |
| codex-cli | 0.144.1 | 0.146.0-alpha | alpha only — staying on stable |

## Backups (two generations kept)

| what | backup | restore |
|---|---|---|
| lm-eval venv @0.4.9 | `.venv-lmeval.bak-0.4.9` | `rm -rf .venv-lmeval && cp -a .venv-lmeval.bak-0.4.9 .venv-lmeval` |

## codex invocation (must-have flags)

`codex exec` silently inherits `~/.codex/config.toml`, project `.rules`, and prior
session summaries — which made it answer three briefs with unrelated documents.
Always call it as:

```bash
codex exec --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check \
           -s read-only --output-schema <schema.json> "<prompt>"
```

Output schemas must set `additionalProperties: false` on **every** object (OpenAI
strict-schema requirement) or the request 400s. Note: `-s read-only` has no network,
so codex cannot do web research — use a web-capable agent for literature work.

**Known limitation (2026-07-26).** The isolation flags fixed *which prompt it answers*,
but four attempts produced no usable analysis: three returned unrelated documents
(pre-flags), and the fourth — correctly isolated and schema-constrained — read the
files (197 KB of tool output) yet emitted zero findings on a codebase where two other
review rounds each found 8–18 real defects. Do not rely on codex for deep adversarial
code review; use it, if at all, only for a cheap second opinion whose absence of
findings means nothing.
