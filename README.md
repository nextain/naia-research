# naia-research

Public OSS for Korean voice cascade evaluation — methodology, toolkit, paradigm, vendor-friendly baselines.

| Area | Description |
|---|---|
| `methodology/` | Metric definitions: `degen` / `d3` / `leak` / `truncation` / `tts_quality_cer` / `audio_TTFP_ms` |
| `toolkit/` | Reusable OSS code: `metrics.py` / `transcribe.py` (faster-whisper) / `report.py` / `run_baseline.py` |
| `seed_data/` | Korean evaluation seed prompts + reference audio |
| `vendor_baselines/` | VoxCPM2 Korean 30-prompt baseline (CER 0.000) |
| `paradigm/` | "Verified external base + our stack differentiation" — Naia's voice paradigm |
| `guides/` | Integration guides: LiveKit ↔ ko-serve voice cascade pattern |
| **`model/fine-tuning/deadpool-lora-demo/`** | **🎭 LLM 자유 교체 데모 — Qwen3에 캐릭터(데드풀)를 LoRA로 입혀 naia-omni에 올리는 전 과정 재현 키트 (스크립트·샘플데이터·페르소나·교체법). [바로가기](model/fine-tuning/deadpool-lora-demo/README.md)** |

## Quick Start

```bash
git clone https://github.com/nextain/naia-research.git
cd naia-research/.agents/toolkit
python metrics.py --results path/to/results.jsonl --transcripts path/to/transcripts.jsonl --out metrics.json
```

## License

- Code: Apache 2.0
- Context (methodology, paradigm, guides): CC-BY-SA 4.0

## Origin

This is the public OSS slice of [naia voice cascade R&D](https://github.com/nextain/naia-agent).
