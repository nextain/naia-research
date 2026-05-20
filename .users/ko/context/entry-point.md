# naia-research (한국어)

naia public OSS — 한국어 voice cascade 평가 methodology + toolkit + paradigm + vendor-friendly baseline.

## 범주

| 영역 | 설명 |
|---|---|
| `methodology/` | 평가 metric 정의 (degen/d3/leak/cer/TTFP) |
| `toolkit/` | OSS code: metrics.py, transcribe.py, report.py, run_baseline.py |
| `seed_data/` | KO eval seed + ref_audio |
| `vendor_baselines/` | VoxCPM2 KO 30 prompts baseline (CER 0.000) |
| `paradigm/` | "검증된 외부 base + 우리 스택" |
| `guides/` | LiveKit ↔ ko-serve 결합 가이드 |

## 매 세션 mandatory reads

1. `.agents/context/agents-rules.json`
2. `.agents/context/project-index.yaml`

## Origin

[naia-labs](https://github.com/nextain/naia-labs) R&D ground에서 promote.
Production: [naia-agent](https://github.com/nextain/naia-agent), [naia-model-infra](https://github.com/nextain/naia-model-infra).
