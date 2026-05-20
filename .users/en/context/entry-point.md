# naia-research (English)

Public OSS for Korean voice cascade evaluation — methodology, toolkit, paradigm, vendor-friendly baselines.

## Categories

See `.agents/context/project-index.yaml` for full index.

| Area | Description |
|---|---|
| `methodology/` | Metric definitions (degen/d3/leak/cer/TTFP) |
| `toolkit/` | OSS code (metrics.py, transcribe.py, report.py, run_baseline.py) |
| `seed_data/` | Korean evaluation seed + reference audio |
| `vendor_baselines/` | VoxCPM2 Korean 30-prompt baseline |
| `paradigm/` | "Verified external base + our stack" |
| `guides/` | LiveKit ↔ ko-serve integration guide |

## Mandatory reads (every session)

1. `.agents/context/agents-rules.json`
2. `.agents/context/project-index.yaml`

## License

- Code: Apache 2.0
- Context: CC-BY-SA 4.0

## Origin

Promote from [naia-labs](https://github.com/nextain/naia-labs) R&D ground.
Production: [naia-agent](https://github.com/nextain/naia-agent), [naia-model-infra](https://github.com/nextain/naia-model-infra).
