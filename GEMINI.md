# naia-research

naia public OSS — Korean voice cascade evaluation methodology, toolkit, paradigm, vendor-friendly baselines.

Korean mirror: `.users/ko/context/entry-point.md`

## Mandatory Reads (every session start)

1. `.agents/context/agents-rules.json` — Project rules (SoT)
2. `.agents/context/project-index.yaml` — Index

## Project Structure

```
.agents/                      # AI SoT (English, JSON/YAML)
├── context/                  # Rules + index
├── methodology/              # metric 정의 (degen/d3/leak/cer/TTFP)
├── toolkit/                  # metrics.py, transcribe.py, report.py, run_baseline.py
├── seed_data/                # KO eval seed prompts + ref_audio
├── vendor_baselines/         # VoxCPM2 KO 30 prompts (vendor 협업)
├── paradigm/                 # "검증된 외부 base + 우리 스택" paradigm 글
└── guides/                   # LiveKit ↔ ko-serve voice cascade pattern

.users/                       # Human mirror
├── en/                       # English (primary)
└── ko/                       # Korean
```

## License

- **Code** (`.agents/toolkit/`, etc.): Apache 2.0 (see `LICENSE`)
- **Context** (`.agents/{methodology,paradigm,guides}/`, `.users/`): CC-BY-SA 4.0 (see `CONTEXT-LICENSE`)

## Origin

Promote from [nextain/naia-labs](https://github.com/nextain/naia-labs) R&D ground.
Production: [naia-agent](https://github.com/nextain/naia-agent), [naia-model-infra](https://github.com/nextain/naia-model-infra).

## Contributing

See `CONTRIBUTING.md`. Add multilingual mirror in `.users/{lang}/`.

cf [Naia repo structure standard](https://github.com/nextain/naia-adk/.agents/context/repo-structure-standard.yaml)
