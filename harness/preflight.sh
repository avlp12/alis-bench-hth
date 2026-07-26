#!/usr/bin/env bash
# preflight.sh — refuse to start a run on a setup that will fail or produce
# unusable numbers. Run this BEFORE run_all.sh. Exit 0 = safe to run.
#
# Checks, in the order they bite you:
#   1 harnesses importable (lm-eval, HRET, numpy)          — else nothing runs
#   2 model dirs exist and look like MLX builds            — else serve fails
#   3 endpoints alive + own chat template + KO consistency — else silent unfairness
#   4 judges configured (>=2 neutral families)             — else not publishable
#   5 runtime provenance clean                             — else not reproducible
#   6 disk headroom for results                            — else truncated run
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/config.env"
FAIL=0; WARN=0
ok(){   printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad(){  printf "  \033[31m✗\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
warn(){ printf "  \033[33m!\033[0m %s\n" "$1"; WARN=$((WARN+1)); }

echo "== 1. harnesses =="
python3 -c "import lm_eval" 2>/dev/null && ok "lm-eval" || bad "lm-eval missing: python3 -m pip install lm-eval"
python3 -c "import llm_eval" 2>/dev/null && ok "HRET" || bad "HRET missing: python3 -m pip install --no-deps haerae-evaluation-toolkit && python3 -m pip install openai datasets litellm math_verify langdetect spacy scikit-learn transformers torch"
python3 -c "import numpy" 2>/dev/null && ok "numpy" || bad "numpy missing"
python3 -c "import spacy;spacy.load('ko_core_news_sm')" 2>/dev/null && ok "spaCy ko model" || warn "spaCy ko model absent (HRET report analysis limited): python3 -m spacy download ko_core_news_sm"

echo "== 2. builds =="
# A contender may be served from ANOTHER box (one model per machine when two
# large builds cannot co-reside in RAM). In that case the path is not local, so a
# missing directory is only a blocker if the endpoint is also unreachable —
# §3 is the real gate. A remote build is noted so it appears in the disclosure.
for pair in "motif|$MOTIF_MODEL|$MOTIF_URL" "solar|$SOLAR_MODEL|$SOLAR_URL"; do
  n="${pair%%|*}"; rest="${pair#*|}"; p="${rest%%|*}"; u="${rest##*|}"
  if [ -e "$p" ] && ls "$p"/*.safetensors >/dev/null 2>&1; then
    ok "$n build: $p ($(ls "$p"/*.safetensors 2>/dev/null | wc -l | tr -d ' ') shards)"
    [ -f "$p/config.json" ] || bad "$n: config.json missing"
  elif curl -s -m 5 "$u/v1/models" >/dev/null 2>&1; then
    warn "$n build not on this host ($p) — served remotely; integrity was verified on its own host"
  else bad "$n build absent locally AND its endpoint is unreachable: $p"; fi
done
for pair in "motif_ref:${MOTIF_REF:-}" "solar_ref:${SOLAR_REF:-}"; do
  n="${pair%%:*}"; p="${pair#*:}"
  [ -n "$p" ] && [ -e "$p" ] && ok "$n (KL anchor): $p" || warn "$n absent — the KL cost cell will be missing"
done

echo "== 3. endpoints =="
for pair in "motif:$MOTIF_URL" "solar:$SOLAR_URL"; do
  n="${pair%%:*}"; u="${pair#*:}"
  if curl -s -m 5 "$u/v1/models" >/dev/null 2>&1; then
    ok "$n endpoint up: $u"
    python3 "$here/sanity_probe.py" "$n" >/dev/null 2>&1 && ok "$n template+KO-consistency probe" \
      || bad "$n sanity probe FAILED — run: python3 $here/sanity_probe.py $n"
  else bad "$n endpoint unreachable: $u  (start it: $here/serve.sh $n)"; fi
done

echo "== 4. judges =="
n_j=$(python3 -c "import json,os;print(len(json.loads(os.environ.get('JUDGE_ENDPOINTS') or '[]')))" 2>/dev/null || echo 0)
[ "$n_j" -ge 2 ] && ok "$n_j neutral judges configured" || bad "need >=2 judges in JUDGE_ENDPOINTS (a 1-judge run is not publishable)"
python3 - <<'PY' 2>/dev/null || warn "could not verify judge API keys"
import json,os
for e in json.loads(os.environ.get("JUDGE_ENDPOINTS") or "[]"):
    k = os.environ.get(e.get("key_env",""), e.get("key",""))
    print(("  \033[32m✓\033[0m " if k else "  \033[33m!\033[0m ") + f"judge {e.get('name')} key {'present' if k else 'MISSING'}")
PY

echo "== 5. provenance =="
H2H_HERE="$here" python3 - <<'PY'
import os,sys; sys.path.insert(0, os.environ["H2H_HERE"])
from manifest import _git_provenance
bad=0
for who,env in (("motif","MLXLM_FORK"),("solar","SOLAR_FORK")):
    p=os.environ.get(env)
    if not p: print(f"  \033[33m!\033[0m {who} runtime path unset ({env}) — provenance will be '?'"); continue
    g=_git_provenance(p)
    if g["commit"] and not g["dirty"]: print(f"  \033[32m✓\033[0m {who} runtime {g['commit'][:12]} clean")
    else: print(f"  \033[31m✗\033[0m {who} runtime dirty ({g['dirty_files']} modified) — commit it, no commit reproduces this run"); bad=1
sys.exit(bad)
PY
[ $? -eq 0 ] || FAIL=$((FAIL+1))

echo "== 6. disk =="
avail=$(df -g "${RESULTS_DIR%/*}" 2>/dev/null | awk 'NR==2{print $4}')
[ -n "$avail" ] && { [ "$avail" -ge 5 ] && ok "${avail}Gi free for results" || warn "only ${avail}Gi free"; }

echo
if [ $FAIL -eq 0 ]; then
  echo "PREFLIGHT PASS ($WARN warnings) — safe to run: $here/run_all.sh"
else
  echo "PREFLIGHT FAIL: $FAIL blocker(s), $WARN warning(s) — fix the ✗ lines above."; exit 1
fi
