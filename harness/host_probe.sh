#!/usr/bin/env bash
# host_probe.sh — record WHERE each contender was served, as facts.
#
# best-vs-best publishes efficiency numbers, and an efficiency number is only
# meaningful next to the machine that produced it. This writes hosts.json so the
# report states the boxes rather than a hand-waved label like "different hardware"
# (which, when actually checked, was wrong: the two boxes are the same model).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; source "$here/config.env"
spec='printf "%s|%s|%.0f|%s|%s" "$(sysctl -n hw.model)" "$(sysctl -n hw.ncpu)" \
  "$(echo "$(sysctl -n hw.memsize)/1073741824" | bc -l)" "$(sw_vers -productVersion)" "$(hostname -s)"'
probe() {  # $1=url -> "model|cores|gb|os|hostname" or ""
  local h; h="$(echo "$1" | sed -E 's#^https?://##; s#:.*##')"
  case "$h" in
    127.0.0.1|localhost) eval "$spec" ;;
    *) ssh -o ConnectTimeout=8 -o BatchMode=yes "$h" "$spec" 2>/dev/null | tail -1 ;;
  esac
}
M="$(probe "$MOTIF_URL")"; S="$(probe "$SOLAR_URL")"
python3 - "$MOTIF_URL" "$SOLAR_URL" "$M" "$S" "$RESULTS_DIR" <<'PY'
import json, sys
mu, su, m, s, rd = sys.argv[1:6]
def d(raw, url):
    f = raw.split("|") if raw else []
    loopback = any(x in url for x in ("127.0.0.1", "localhost"))
    return {"url": url, "reached_over": "loopback" if loopback else "LAN",
            "machine": f[0] if f else "UNKNOWN", "cores": f[1] if len(f) > 1 else "?",
            "ram_gb": f[2] if len(f) > 2 else "?", "os": f[3] if len(f) > 3 else "?",
            "hostname": f[4] if len(f) > 4 else "?"}
M, S = d(m, mu), d(s, su)
same_box = M["hostname"] == S["hostname"] and M["hostname"] != "?"
same_spec = all(M[k] == S[k] for k in ("machine", "cores", "ram_gb", "os")) and M["machine"] != "UNKNOWN"
if same_box:
    note = "Both contenders served on the SAME box — efficiency numbers are directly comparable."
elif same_spec:
    note = ("Served on two SEPARATE but IDENTICALLY SPECIFIED boxes. Throughput is comparable "
            "to within: (a) the network hop on the LAN-reached side, which loopback does not pay; "
            "(b) whatever else each box was doing — the lm-eval driver itself runs co-resident "
            "with the loopback-served model. This is NOT a same-box back-to-back measurement, "
            "and no efficiency claim here is a controlled one.")
else:
    note = ("Served on boxes of DIFFERENT specification — throughput is NOT comparable across "
            "models and must not be read as an efficiency result.")
json.dump({"motif": M, "solar": S, "same_box": same_box, "same_spec": same_spec,
           "comparability": note}, open(rd + "/hosts.json", "w"), ensure_ascii=False, indent=2)
print(note)
PY
