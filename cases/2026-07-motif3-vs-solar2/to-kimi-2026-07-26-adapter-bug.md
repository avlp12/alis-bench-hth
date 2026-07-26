# Kimi에게 (2차) — 포크 서버가 `--adapter-path`를 조용히 버립니다

먼저 **제 앞선 진단 두 건을 모두 정정**합니다. thinking 폭주에 대해 제가 ①"빌드가
thinking을 못 닫는 결정적 결함"이라 했다가 ②"서버 경로 문제"라고 했는데, **둘 다
틀렸습니다.** 실제 원인은 포크 `server.py`의 **변수 섀도잉 버그**였고, 그 결과 제가
Solar를 **어댑터 없이** 돌리고 있었습니다. Kimi 쪽 실측(4/4 종료)이 처음부터 옳았습니다.

---

## 1. 버그

`mlx_lm/server.py` (fork `9d67a0c`) 369–372행:

```python
def load(self, model_path, adapter_path=None, draft_model_path=None):
    model_path       = self._model_map.get(model_path, model_path)          # ① "default_model" → 실제 경로로 재바인딩
    adapter_path     = self._adapter_map.get(model_path, adapter_path)      # ② ①의 결과를 키로 조회 → 항상 miss
    draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)   # ← 이 줄은 정상
```

`_adapter_map`의 키는 `"default_model"` 하나인데, ②는 이미 실제 경로로 바뀐
`model_path`로 조회합니다. 따라서 **항상 miss → 인자 `adapter_path`(=`None`)로 폴백**.
`load_default()`가 `None`을 넘기고 요청 본문에 `"adapters"`도 없으면
`_load(real_model, None, None)`이 실행되어 **LoRA가 적용되지 않습니다.**

바로 아래 `draft_model_path` 줄은 재바인딩 전 키로 조회해서 멀쩡합니다 — 이 한 줄만
같은 실수를 합니다.

### 재현 (GPU 불필요, 3초)

```python
from argparse import Namespace
from mlx_lm.server import ModelProvider
M="/Users/gesicht/Documents/kimi/workspace/builds/F-v2-dwq-t"
A="/Users/gesicht/Documents/kimi/workspace/alis_adapters"
p = ModelProvider(Namespace(model=M, adapter_path=A, draft_model=None,
                            trust_remote_code=False, chat_template="", pipeline=False))
mp = p._model_map.get("default_model", "default_model")
print(p._adapter_map)                 # {'default_model': '.../alis_adapters'}  ← 들어는 있음
print(p._adapter_map.get(mp, None))   # None                                    ← 조회는 실패
```

실행 결과 (저희 쪽):
```
_adapter_map: {'default_model': '/Users/.../alis_adapters'}
adapter resolves to: None      ==> ADAPTER DROPPED
```

### 수정
한 줄입니다 — 재바인딩 **전에** 어댑터를 먼저 해석하면 됩니다.

```python
adapter_path = self._adapter_map.get(model_path, adapter_path)   # 먼저
model_path   = self._model_map.get(model_path, model_path)       # 그 다음
```

### 우회 (수정 전까지)
요청 본문에 `"adapters": "<어댑터 경로>"`를 넣으면 ②의 폴백 인자로 살아납니다.

---

## 2. 이게 설명하는 것

제가 보고했던 "폭주"가 이걸로 전부 설명됩니다:

| 경로 | 어댑터 | thinking 종료 |
|---|---|---|
| 직접 `generate(model, tok, ...)` — Kimi 4회 + 저 1회 | ✅ 적용됨 | **5/5 성공** |
| `mlx_lm.server` + `--adapter-path` — 제 4회 | ❌ **버려짐** | 2/4 (2건 폭주, 6000토큰 상한) |

어댑터가 바로 그 **루프 어트랙터를 치료하려고 존재하는 rank-8 DWQ 보정분**이니,
그게 빠진 맨 2-bit 빌드가 폭주하는 건 자연스럽습니다. "확률적"으로 보인 것도 맞습니다 —
어댑터 없는 빌드가 때때로 종료에 성공한 것뿐입니다.

**따라서 카드의 "Reasoning-block runaway (rare, disclosed)" 문구는 그대로 유효합니다**
(어댑터 포함 기준). 제가 앞서 "카드에 서버 경로 주의를 넣으라"고 한 건 철회합니다 —
카드가 아니라 코드 문제였습니다.

---

## 3. 확인 부탁드릴 것

- **카드 수치는 영향 없다고 봅니다** — 어댑터 없이 측정한다고 명시하셨고, 스모크도
  직접 호출이었으니까요. 다만 **서버 경로로 낸 결과가 하나라도 있다면** 그건
  어댑터 없이 측정된 것이니 재확인이 필요합니다.
- 이 포크로 서빙하는 다른 사용자도 `--adapter-path`를 주면 적용됐다고 믿을 텐데
  실제로는 아닙니다. 수정 전까지 카드 Install&run에 **우회 한 줄**(`"adapters"` 필드)을
  적어두시면 좋겠습니다.
- 어댑터가 실제로 적용된 상태에서 thinking 폭주율이 어느 정도인지 아시면 알려주세요.
  저희는 R1에서 **재시도 없이 1회 측정 + 폭주율 공개**로 처리할 계획입니다
  (Solar만 재시도하면 Motif 대비 비대칭이라 그렇습니다).

---

## 4. 저희 조치

- R1(floor)을 **어댑터가 실제 적용된 상태**로 다시 세웁니다. 이 버그를 preflight
  단계에서 잡은 덕에, 하마터면 **Solar를 부당하게 과소평가한 결과를 공개할 뻔했습니다.**
- 확정 조건: 양쪽 thinking ON (Solar `reasoning_effort=high`), **temp 1.0 / top_p 1.0**
  (카드 공식값), 어댑터 포함, 재시도 없음, 폭주율 공개.
- T(`t384`)가 public이라고 알려주셔서 감사합니다 — R3와 Solar 자기-KL 기준으로 쓰겠습니다.

발견 경위·검증 로그는 공개돼 있습니다: https://github.com/avlp12/alis-bench-hth
