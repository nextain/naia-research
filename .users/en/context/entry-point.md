# naia-research (English)

The public window onto naia's Korean voice work. Full guide: [README.md](../../../README.md).

## What this is

Most of our Korean voice research lives in a private repo (naia-labs). This repo
publishes the runnable part: the tools we use to measure how good a Korean voice
pipeline is, the prompts we measure it with, and one measured result you can
reproduce. A "voice cascade" is the chain behind a talking assistant — speech
recognition (STT), a language model (LLM), then speech synthesis (TTS) — and
this repo grades the Korean quality of that chain end to end.

The purpose is verifiability. We claim a good off-the-shelf voice model, used
as-is, can beat a model we fine-tuned for 27.66 hours. Rather than ask for
trust, we ship the ruler so a skeptic can check it.

## What actually works today

- **Evaluation toolkit** (`.agents/toolkit/`) — `transcribe.py` (faster-whisper
  over generated audio), `metrics.py` (character error rate, empty-response and
  truncation rates, non-Korean leak, time to first audio packet), `report.py`
  (Markdown report, linking the generated clips when they sit alongside the
  results).
- **Korean evaluation seed** (`.agents/seed_data/`) — two prompt sets plus one
  reference voice clip. `ko_eval_seed.py` is 30 stress-test sentences (minimal
  pairs, vowels, numerals, loanwords, proper nouns, long single-breath lines);
  `ko_input_wavs/` is 30 everyday questions (daily chat, museum docent, trivia)
  already rendered to audio for full-cascade input.
- **VoxCPM2 baseline** (`.agents/vendor_baselines/`) — the 30 stress-test
  sentences from `ko_eval_seed.py` (not the daily/museum/trivia clips)
  synthesized with zero training and scored by character error rate: median CER
  0.000, mean 0.107, pass rate (CER ≤ 0.30) 86.7% at n=30. For context, a 27.66h
  fine-tuned Talker scored median CER 0.138 at 83.3% pass; that fine-tune is
  private, so its number is carried over for comparison, not re-runnable here.
  The eval script reads its prompts from the shipped seed and needs VoxCPM2,
  faster-whisper, and a CUDA GPU.
- **LLM swap demo** (`model/fine-tuning/deadpool-lora-demo/`) — a reproducible
  kit that puts a character personality on Qwen3 with a LoRA adapter and loads
  it into the Naia Omni runtime, so you can change a voice avatar's "brain"
  without retraining anything else.

## Roadmap (not shipped yet)

Methodology writeups, a paradigm essay, and integration guides are planned. Their
directories do not exist until there is real content — metric definitions
currently live as docstrings in `metrics.py`.

## Mandatory reads (every session)

1. `.agents/context/agents-rules.json`
2. `.agents/context/project-index.yaml`

## License

Code: Apache 2.0. Docs and evaluation data: CC-BY-SA 4.0.

## Origin

Promoted from [naia-labs](https://github.com/nextain/naia-labs) R&D ground.
Production: [naia-agent](https://github.com/nextain/naia-agent),
[naia-model-infra](https://github.com/nextain/naia-model-infra).
