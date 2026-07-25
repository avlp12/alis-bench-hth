# alis-bench-hth

**Head-to-head benchmark harness for locally-served LLMs (Apple Silicon / MLX).**
Built to answer one question honestly: *given two models each at their own best,
which is actually better — and what did each pay to get there?*

Companion to [alis-dwq](https://github.com/avlp12/alis-dwq) (the quantization
pipeline). This repo is the **measurement** half, and it accumulates cases over
time: every matchup keeps its preregistration, run manifest, and results, so
later matchups can be compared against earlier ones.

---

## Design stance

**best-vs-best, with the bill attached.** Each model runs at its own optimal
quantization and peak settings — matching knobs across different architectures
is not fairness, it is a different (and usually unanswerable) question.
Objectivity comes from three things instead:

1. **Invariant task, scoring and judge** — the only axis that must never vary per model.
2. **Full disclosure** — peak settings, runtime commit, and *tokens actually spent*
   are frozen into a manifest before results exist, and printed beside every verdict.
   A win bought with 3× the tokens is reported as exactly that.
3. **Fail-closed integrity** — a verdict is emitted only if every required cell is
   present and stamped with the current run id. Missing or stale evidence blocks
   the report rather than quietly shrinking it.

### Three tracks — only one of them is "best-vs-best"
| track | what it isolates |
|---|---|
| high-precision base (8-bit/bf16) | which *base model* is better |
| matched-resource quant | which *quantization* is better at equal budget |
| unrestricted best-submitted | which *shipped system* is better — the headline, with cost |

Cross-model KL is never ranked: `KL(own-8bit ‖ build)` is a per-model cost, not a
common scale.

## Judging open generation — and who is allowed to judge

Most of the suite needs **no judge at all**: MCQA (KMMLU, HAE-RAE, CLIcK), rule-verified
IFEval, answer-extracted math (HRM8K, GSM8K, MATH), and executed code are all scored
mechanically. Only **open-generation quality** needs one, so a missing judge API blocks
one section — never the whole run.

Judge options, strongest first:

| option | why | publishable |
|---|---|---|
| ≥2 neutral judge APIs, both A/B orders | inter-judge agreement is measurable | yes |
| **blinded pack → human native speaker** (`blind_pack.py`) | for *Korean naturalness* a native reader beats any LLM judge; this is the calibration the protocol already demands | yes (n≥20) |
| 1 API judge + a second model | agreement measurable but thin | marginal |
| the harness author's own model | see below | **no** |

**Why the harness author must not be the sole judge.** Not being a contender is not
enough. Whoever built one side's port, quantization and model card has an
*authorship* conflict, and having read that model's outputs during development, is not
blind to its style. A single judge also yields no agreement statistic — you cannot tell
a confident judge from a correct one. `blind_pack.py` therefore hides model identity,
randomizes the side per item, shuffles item order, and writes the mapping to a
**separate key file**, so a reviewer file alone cannot reveal which side is which.
`score` marks the result `publishable: false` unless the reviewer is human and n≥20.

```bash
python3 harness/blind_pack.py make --prompts logickor.jsonl   # -> blind_pack.md + blind_key.json
# reviewer fills each `판정:` line with 가 / 나 / 무승부 (never opening the key)
python3 harness/blind_pack.py score --reviewer "홍길동"        # -> pairwise.json, same shape aggregate.py reads
```

## Anti-footguns baked in
Each of these is a bug this harness already hit, and now refuses to repeat:
- **metric[filter] schema** — a multi-filter task (`gsm8k_cot`) silently scored the
  wrong filter until keys included the filter; there is no "first float" fallback.
- **group-block merge** — `mmlu_pro`/`bbh`/`minerva_math` aggregates can live under
  `groups`, not `results`; reading one block silently dropped whole benchmarks.
- **paired stats or nothing** — per-item paired bootstrap + 2-sided p + Holm across
  the task family; marginal means alone never decide a winner, and zero paired
  coverage is a loud failure, not a silent "inconclusive".
- **judge hygiene** — ≥2 independent judge families, both A/B orders, and ties /
  parse-failures / order-effects counted separately. Win-rate is over all n, never
  "vs decided" (which can print 100% from 10 wins and 32 disagreements).
- **`--apply_chat_template`** — required by `local-chat-completions`, and with
  `tokenized_requests=False` it makes the *server* apply each model's own template.
- **contamination labelling** — vanilla KMMLU is carried with an explicit
  `_contaminated` flag rather than presented as the decontaminated suite.
- **tokenizer-neutral efficiency** — chars/s *and* tok/s, warm-up first, spread kept.

## Layout
```
harness/    reusable kit (serve, probe, eval, judge, KL, integrity, aggregate)
cases/      one directory per matchup: prereg + manifest + results + writeup
docs/       protocol.html (shareable spec)
```

## Quick start
```bash
cp harness/config.env.example harness/config.env   # set builds, endpoints, judges
harness/serve.sh motif          # box A
harness/serve.sh solar          # box B
harness/preflight.sh            # refuses to start on a bad setup
harness/run_all.sh              # fail-closed; blocks aggregation on missing cells
```

## Cases
| date | matchup | status |
|---|---|---|
| 2026-07 | [Motif-3-Beta vs Solar Open 2](cases/2026-07-motif3-vs-solar2/) | in progress |

## License
MIT for the harness. Benchmark datasets and model weights keep their own licenses —
several Korean suites and both models here are non-commercial research only.
