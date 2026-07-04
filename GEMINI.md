# naia-research

Public OSS slice of naia's Korean voice cascade work: an evaluation toolkit, a
Korean evaluation seed, one reproducible VoxCPM2 baseline, and an LLM-swap
fine-tuning demo. Promoted from the private naia-labs R&D repo.

Human-facing entry: [README.md](README.md).
Korean mirror: `.users/ko/context/entry-point.md`.

## Mandatory reads (every session start)

1. `.agents/context/agents-rules.json` — Project rules (SoT)
2. `.agents/context/project-index.yaml` — Index

## Project structure

Only directories with real content exist. `methodology/`, `paradigm/`, and
`guides/` are on the roadmap (see README) and are intentionally absent until
there is content to publish — do not describe them as if they ship today.

```
.agents/                     # AI SoT (English, JSON/YAML)
├── context/                 # Rules + index
├── toolkit/                 # metrics.py, transcribe.py, report.py, run_baseline.py
├── seed_data/               # stress-test seed (ko_eval_seed.py, 30 ko) + cascade-input clips (ko_input_wavs/, 30) + reference audio
└── vendor_baselines/        # VoxCPM2 KO baseline over the stress-test seed (eval script reads the shipped seed; wavs, results, summary)

model/
└── fine-tuning/
    └── deadpool-lora-demo/  # LoRA persona training → export → load into Naia Omni

.users/                      # Human mirror
├── en/                      # English (primary)
└── ko/                      # Korean
```

Note: `run_baseline.py` imports `bridge_gemini_minicpm`, which lives in the
private serving stack and is not shipped here. It is a reference for how our
baselines are generated, not a standalone script.

## License

- **Code** (`.agents/toolkit/`, `model/`, etc.): Apache 2.0 (see `LICENSE`)
- **Docs and evaluation data** (`.users/`, seed and baseline data): CC-BY-SA 4.0
  (see `CONTEXT-LICENSE`)

## Origin

Promoted from [nextain/naia-labs](https://github.com/nextain/naia-labs) R&D
ground. Production runtime: [naia-agent](https://github.com/nextain/naia-agent),
[naia-model-infra](https://github.com/nextain/naia-model-infra).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Mirror any new document under
`.users/{lang}/`.

cf. [Naia repo structure standard](https://github.com/nextain/naia-adk/.agents/context/repo-structure-standard.yaml)
