#!/usr/bin/env python3
"""Run provenance + fail-closed integrity for the head-to-head.

Every run gets an immutable RUN_ID = hash(model paths + config + kit code + time).
Each required artifact is stamped with that RUN_ID; aggregation refuses to crown a
winner unless every required cell is present AND stamped with the CURRENT RUN_ID
(so fresh-Motif-vs-stale-Solar can't be silently reported).

  init                       -> create run_manifest.json in RESULTS_DIR, print RUN_ID
  stamp <path>               -> stamp an artifact (dir -> <dir>/.runid, file -> <file>.runid)
  record <step> <exit_code>  -> record a step's exit status
  check                      -> verify completeness+freshness; exit 1 if incomplete
                                (unless ALLOW_PARTIAL=1); prints COMPLETE|PARTIAL|INCOMPLETE
"""
import hashlib, json, os, sys, time
from pathlib import Path

R = Path(os.environ["RESULTS_DIR"])
MAN = R / "run_manifest.json"

REQUIRED = [  # (step-key, path relative to RESULTS_DIR)
    ("lm_eval.motif", "lm_eval/motif"), ("lm_eval.solar", "lm_eval/solar"),
    ("hret.motif", "hret/motif"),       ("hret.solar", "hret/solar"),
    ("judge", "judge/pairwise.json"),
    ("kl.motif", "kl/motif_kl.txt"),    ("kl.solar", "kl/solar_kl.txt"),
    ("thru.motif", "throughput/motif.json"), ("thru.solar", "throughput/solar.json"),
]

def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()

def _code_sha():
    here = Path(__file__).parent
    h = hashlib.sha256()
    for f in sorted(here.glob("*.py")) + sorted(here.glob("*.sh")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]

def _git_provenance(path):
    """(commit, dirty) for a runtime checkout — a dirty tree means NO commit
    reproduces the runtime, which disqualifies the run from being 'reproducible'."""
    import subprocess
    p = Path(path or "")
    if not p.exists():
        return {"path": str(path), "commit": None, "dirty": None, "note": "path missing"}
    def _g(*a):
        try:
            return subprocess.run(["git", "-C", str(p), *a], capture_output=True,
                                  text=True, timeout=15).stdout.strip()
        except Exception:
            return ""
    commit = _g("rev-parse", "HEAD") or None
    status = _g("status", "--porcelain")
    return {"path": str(p), "commit": commit, "dirty": bool(status),
            "dirty_files": len([l for l in status.splitlines() if l.strip()])}

def _disclosure():
    """The cost/settings ledger. best-vs-best only stays honest if the price each
    model paid for its peak is on the record — otherwise 'whoever spent more'
    silently wins. Captured at init so it cannot be edited after seeing results."""
    e = os.environ.get
    return {
        "thinking": e("THINKING"),
        "max_tokens": e("MAX_TOKENS"),
        "seed": e("SEED"),
        "sampling": {
            "motif": {"temp": e("MOTIF_TEMP"), "top_p": e("MOTIF_TOP_P")},
            "solar": {"temp": e("SOLAR_TEMP"), "top_p": e("SOLAR_TOP_P")},
        },
        "builds": {"motif": e("MOTIF_MODEL"), "solar": e("SOLAR_MODEL"),
                   "motif_ref": e("MOTIF_REF"), "solar_ref": e("SOLAR_REF")},
        "runtime": {"motif_fork": _git_provenance(e("MLXLM_FORK")),
                    "solar_fork": _git_provenance(e("SOLAR_FORK"))},
        "hret_penalize": e("HRET_PENALIZE", "off"),
    }

def _stamp_path(p: Path) -> Path:
    """Where the .runid for artifact p lives: dir -> p/.runid, file -> p.runid."""
    if p.is_dir():
        return p / ".runid"
    return Path(str(p) + ".runid")

def _run_id():
    return json.loads(MAN.read_text())["run_id"]

def init():
    R.mkdir(parents=True, exist_ok=True)
    cfg = Path(__file__).parent / "config.env"
    config_sha = _sha(cfg.read_bytes())[:16] if cfg.exists() else "nocfg"
    # DETERMINISTIC on (models, endpoints, config, kit code, suite mode) — NOT on
    # wall-clock. Re-running an identical setup keeps the same run_id, so a KL or
    # throughput cell computed in an earlier session stays valid instead of going
    # falsely stale; changing any model/config/code invalidates every cell.
    run_id = _sha(os.environ.get("MOTIF_MODEL", ""), os.environ.get("SOLAR_MODEL", ""),
                  os.environ.get("MOTIF_URL", ""), os.environ.get("SOLAR_URL", ""),
                  os.environ.get("SUITE_MODE", "full"), config_sha, _code_sha())[:12]
    prev = json.loads(MAN.read_text()).get("status", {}) if MAN.exists() else {}
    prev_id = json.loads(MAN.read_text()).get("run_id") if MAN.exists() else None
    MAN.write_text(json.dumps({
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suite_mode": os.environ.get("SUITE_MODE", "full"),
        "models": {"motif": os.environ.get("MOTIF_MODEL"), "solar": os.environ.get("SOLAR_MODEL")},
        "endpoints": {"motif": os.environ.get("MOTIF_URL"), "solar": os.environ.get("SOLAR_URL")},
        "config_sha": config_sha, "code_sha": _code_sha(),
        "disclosure": _disclosure(),          # peak settings + runtime provenance
        "required": [k for k, _ in REQUIRED],
        # keep prior step statuses when the run_id is unchanged (same setup),
        # so out-of-band cells (KL, throughput) survive a re-run of run_all.sh
        "status": prev if prev_id == run_id else {},
    }, indent=2, ensure_ascii=False))
    print(run_id)

def stamp(target):
    p = R / target if not os.path.isabs(target) else Path(target)
    if not p.exists():
        return  # nothing produced -> leave unstamped so check() flags it
    sp = _stamp_path(p)
    sp.write_text(_run_id())

def record(step, code):
    man = json.loads(MAN.read_text())
    man["status"][step] = int(code)
    MAN.write_text(json.dumps(man, indent=2, ensure_ascii=False))

def check():
    man = json.loads(MAN.read_text())
    run_id = man["run_id"]
    missing, stale, badstep = [], [], []
    for key, rel in REQUIRED:
        p = R / rel
        if not p.exists():
            missing.append(rel); continue
        sp = _stamp_path(p)
        rid = sp.read_text().strip() if sp.exists() else None
        if rid != run_id:
            stale.append(f"{rel} (runid={rid})")
        if man["status"].get(key, 0) not in (0, None):
            badstep.append(f"{key}={man['status'][key]}")
    ok = not (missing or stale)
    n_bad = len(set(missing) | {s.split(' ')[0] for s in stale})
    verdict = "COMPLETE" if ok else ("INCOMPLETE" if n_bad == len(REQUIRED) else "PARTIAL")
    man["integrity"] = {"verdict": verdict, "missing": missing, "stale": stale, "failed_steps": badstep}
    MAN.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(f"[integrity] {verdict}  run_id={run_id}  ({len(REQUIRED)-n_bad}/{len(REQUIRED)} cells fresh)")
    for label, items in (("missing", missing), ("STALE(wrong run_id)", stale), ("failed steps", badstep)):
        if items:
            print(f"  {label}: " + ", ".join(items))
    if not ok and os.environ.get("ALLOW_PARTIAL") != "1":
        print("  -> fail-closed: NOT publishable. Fix the cells above, or set ALLOW_PARTIAL=1 to force a PARTIAL report.")
        sys.exit(1)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "init": init()
    elif cmd == "stamp": stamp(sys.argv[2])
    elif cmd == "record": record(sys.argv[2], sys.argv[3])
    elif cmd == "check": check()
    else: sys.exit(f"unknown cmd {cmd}")
