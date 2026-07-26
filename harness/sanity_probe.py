#!/usr/bin/env python3
"""Pre-flight fairness check for one endpoint (run before any eval).

Verifies, per model:
  1. the endpoint answers at all (chat-completions round-trip);
  2. the server applies the model's OWN chat template (we send raw messages,
     so a coherent reply means the template fired);
  3. language consistency — a Korean prompt gets a Korean answer, not English
     code-switching (the #1 silent unfairness for KO-prompted reasoning models);
  4. stop tokens behave (no runaway / no leaked special tokens).

Usage:  python3 sanity_probe.py motif   |   python3 sanity_probe.py solar
Exit 0 = looks fair to proceed; non-zero = fix before benchmarking.
"""
import json, os, sys, urllib.request

def chat(url, model, messages, max_tokens=256, temperature=0.0):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature,
                       "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)

def answer_of(resp):
    """(content, reasoning) — tolerant of reasoning models. Two traps hit live:
      * mlx_lm.server RESOLVES the request's `model` field: if it does not equal
        the served path it tries to fetch that repo from HF (404) or loads a
        second copy and OOMs. Surfaced here as a clear, actionable error.
      * a reasoning model can burn the whole budget inside its thinking block,
        leaving `content` empty while `reasoning_content` is full — scoring
        `content` alone would silently produce 0 on EVERY item."""
    if isinstance(resp, dict) and resp.get("error"):
        raise SystemExit(f"[probe] server error: {str(resp['error'])[:280]}\n"
                         "  -> the request's `model` MUST equal the served model path exactly.")
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    return ((msg.get("content") or "").strip(),
            (msg.get("reasoning_content") or msg.get("reasoning") or "").strip())

def hangul_ratio(s):
    h = sum(0xAC00 <= ord(c) <= 0xD7A3 for c in s)
    letters = sum(c.isalpha() for c in s) or 1
    return h / letters

def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "motif").lower()
    url = os.environ["MOTIF_URL" if name == "motif" else "SOLAR_URL"]
    model = os.environ["MOTIF_MODEL" if name == "motif" else "SOLAR_MODEL"]
    print(f"[probe] {name}  {url}  model={model}")
    fails = []

    # 1+2: basic round-trip, coherent reply => template applied.
    # Budget must be generous: a reasoning model needs room to finish thinking
    # BEFORE it emits any visible answer.
    budget = int(os.environ.get("PROBE_TOKENS", "1024"))
    r = chat(url, model, [{"role": "user", "content": "1+1은 얼마인가요? 숫자만 답하세요."}], budget)
    a, think = answer_of(r)
    print(f"[probe] arithmetic -> content={a[:60]!r} reasoning_len={len(think)}")
    if not a and think:
        fails.append(f"empty content with {len(think)} chars of reasoning at max_tokens={budget} "
                     "-> RAISE MAX_TOKENS; scoring `content` would give 0 on every item")
    elif "2" not in a:
        fails.append("basic arithmetic wrong (template or model load issue)")

    # 3: language consistency on a Korean reasoning prompt
    r = chat(url, model, [{"role": "user",
                           "content": "한국의 사회 고령화가 국민연금에 미치는 영향을 세 문장으로 설명해줘."}],
             max(budget, 1024))
    a, think = answer_of(r)
    if not a:
        fails.append("empty content on the KO prompt (raise max_tokens / check stop tokens)")
    hr = hangul_ratio(a)
    print(f"[probe] KO-consistency hangul_ratio={hr:.2f}  sample={a[:80]!r}")
    if hr < 0.45:
        fails.append(f"Korean prompt answered mostly non-Korean (hangul_ratio={hr:.2f}) "
                     "-> enable HRET language_penalize and record it")

    # 4: stop-token / leakage check.
    # Think markers matter most: if a reasoning block leaks into `content`, every
    # answer-extraction filter scores the *thinking* instead of the answer, and it
    # does so asymmetrically (only the model whose server fails to split it).
    for tok in ["<eos>", "</s>", "<|im_end|>", "<|endoftext|>"]:
        if tok in a:
            fails.append(f"special token {tok!r} leaked into output (stop-token config)")
    for tok in ["</think>", "<think>", "<|think:end|>", "<|think:start|>"]:
        if tok in a:
            fails.append(f"think marker {tok!r} is inside `content` — the reasoning block is "
                         "NOT being split out, so answer extraction will score the thinking")
    # 5: report the thinking split so the round's disclosed cost is auditable.
    #    (Empty reasoning is NOT a failure: a model may legitimately answer without
    #     an internal block. It must be DISCLOSED, not silently averaged away.)
    print(f"[probe] thinking: reasoning_chars={len(think)} answer_chars={len(a)} "
          f"-> {'reasons internally' if len(think) > 40 else 'reasons in the visible answer'}")

    usage = r.get("usage", {})
    print(f"[probe] usage={usage}")
    if fails:
        print("[probe] FAIL:"); [print("   -", f) for f in fails]; sys.exit(1)
    print("[probe] OK — endpoint fair to benchmark"); sys.exit(0)

if __name__ == "__main__":
    main()
