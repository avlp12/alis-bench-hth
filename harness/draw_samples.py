#!/usr/bin/env python3
"""draw_samples.py — draw the SEEDED RANDOM item sample for each sampled task.

lm-eval's --limit takes the FIRST N docs per leaf task, not a random sample: a
"sampled" task under --limit is really a head-of-dataset slice, biased by
whatever the dataset happens to be ordered by. 0.4.12's --samples flag takes an
explicit {task: [doc indices]} mapping instead, so the sample can be drawn
seeded-random here, frozen (sha256) in the preregistration BEFORE any scored
run, and passed identically to both models. Item identity itself becomes part
of the frozen plan — choosing items after seeing scores is then structurally
impossible.

Group tasks are expanded to their leaves (that is what --samples keys on); a
total budget for a group is split evenly across leaves, remainder to the first
leaves in name order (deterministic).

  .venv-lmeval/bin/python draw_samples.py \
      --spec "mmlu_pro=154,minerva_math500=150,ifeval=200" \
      --seed 20260727 --out samples.json
"""
import argparse, hashlib, json, random

from lm_eval.tasks import TaskManager


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif hasattr(o, "has_test_docs"):
        yield o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True,
                   help="task=TOTAL_N[,task=TOTAL_N...] — group totals split across leaves")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    tm = TaskManager()
    out, report = {}, []
    for part in a.spec.split(","):
        top, n_total = part.split("=")
        n_total = int(n_total)
        ls = sorted(leaves(tm.load([top]) if hasattr(tm, "load") else tm.load_task_or_group([top])),
                    key=lambda t: t.config.task)
        base, extra = divmod(n_total, len(ls))
        for i, task in enumerate(ls):
            docs = list(task.test_docs()) if task.has_test_docs() else list(task.validation_docs())
            want = base + (1 if i < extra else 0)
            n = min(want, len(docs))
            # per-leaf child seed: same draw regardless of which other tasks are in the spec
            rng = random.Random(f"{a.seed}:{task.config.task}")
            out[task.config.task] = sorted(rng.sample(range(len(docs)), n))
            report.append((task.config.task, n, len(docs)))

    js = json.dumps(out, sort_keys=True, separators=(",", ":"))
    open(a.out, "w").write(js)
    sha = hashlib.sha256(js.encode()).hexdigest()
    open(a.out + ".sha256", "w").write(f"{sha}  {a.out}\n")
    print(f"[samples] seed={a.seed}  ->  {a.out}")
    for name, n, pop in report:
        print(f"  {name:44s} {n:>4d} / {pop}")
    print(f"[samples] total={sum(r[1] for r in report)}  sha256={sha[:16]}…  "
          f"(freeze this in the preregistration)")


if __name__ == "__main__":
    main()
