# naia-research

[한국어](.users/ko/context/entry-point.md) · [English](.users/en/context/entry-point.md)

## What this is

naia-research is the public window onto our Korean voice work. Most of the
research lives in a private repo (naia-labs); this repo publishes the small
part that other people can actually run: the tools we use to *measure* how good
a Korean voice pipeline is, the prompts we measure it with, and one measured
result you can reproduce.

The point is honesty. We make a specific claim — that a good off-the-shelf
voice model, used as-is, can beat a model we fine-tuned for 27.66 hours. A claim
like that is only worth anything if someone else can check it. So instead of
asking you to trust a number, we ship the ruler.

A "voice cascade" here means the usual chain behind a talking assistant: speech
in becomes text (speech recognition, STT), text goes through a language model
(LLM), and the reply becomes speech again (speech synthesis, TTS). This repo is
about grading the Korean quality of that chain end to end.

## What actually works today

Everything in this section is code you can run and data you can open.

- **An evaluation toolkit** (`.agents/toolkit/`). `transcribe.py` runs
  faster-whisper over generated audio to get a transcript; `metrics.py` turns
  the transcript plus the server's text into numbers (character error rate,
  empty-response rate, truncation rate, how much non-Korean text leaked in,
  time to first audio packet); `report.py` renders those numbers into a
  Markdown report, with links to the generated clips when they sit alongside
  the results.

- **A Korean evaluation seed** (`.agents/seed_data/`). Two prompt sets for two
  jobs, plus one reference voice clip (`ref_ko_485.wav`). The first set,
  `ko_eval_seed.py`, is thirty short Korean sentences grouped by what tends to
  break Korean speech: minimal pairs, tricky vowels, numerals, loanwords,
  proper nouns, long single-breath sentences, and a few plain baseline lines.
  The second set, `ko_input_wavs/`, is thirty everyday questions in three
  settings — daily chat, a museum docent, and general trivia — already rendered
  to audio so you can feed them into a full cascade as spoken input.

- **A VoxCPM2 baseline** (`.agents/vendor_baselines/`). We took the thirty
  stress-test sentences from `ko_eval_seed.py`, synthesized each one with
  VoxCPM2 doing zero training, and scored the audio by character error rate
  against a Whisper transcript. (These are the stress sentences, not the
  daily/museum/trivia clips above — a different set for a different question.)
  The measured result, reported exactly as it came out:

  | Setup | Median CER | Pass rate (CER ≤ 0.30) |
  |---|---|---|
  | VoxCPM2, zero training (n=30) | 0.000 | 86.7% |

  Character error rate (CER) is the share of characters the transcriber got
  wrong; lower is better, and a median of 0.000 means half the clips
  transcribed perfectly. Mean CER for this run was 0.107, pulled up by a few
  hard prompts. For context, a Talker we fine-tuned for 27.66 hours on the same
  reference voice and the same kind of Korean text scored a median CER of 0.138
  at an 83.3% pass rate. That fine-tune lives in our private research repo, so
  the 0.138 is a number we carry over for comparison, not one you can re-run
  from this repo; the VoxCPM2 side is the part you can rebuild here. The
  `voxcpm2_ko_eval.py` script pulls its prompts straight from the shipped seed,
  so reproducing it needs VoxCPM2, faster-whisper, and a CUDA GPU rather than
  any file outside the tree. Exact numbers still move a little with model and
  Whisper versions.

- **An LLM swap demo** (`model/fine-tuning/deadpool-lora-demo/`). A complete,
  reproducible kit that takes an off-the-shelf language model (Qwen3), teaches
  it a character personality with a small LoRA adapter, and loads it into the
  Naia Omni runtime — so you can change the "brain" of a voice avatar without
  retraining anything else. It ships the training, evaluation, export, and
  masking scripts plus sample data, and a writeup of the traps we hit (why
  completion-only masking matters, why a missing chat template makes the model
  ramble). See its
  [README](model/fine-tuning/deadpool-lora-demo/README.md).

## Why it matters

Our approach is to take a verified external model, use it as-is, and put the
effort into the layer around it: memory, privacy, retrieval, and local serving.
The VoxCPM2 result above is the first piece of evidence that this can work — an
untrained off-the-shelf model held its own against a fine-tune we spent 27.66
hours on. We publish the toolkit and the data, not just the number, because a
measurement is only useful if someone else can check it. If the ruler is wrong,
you can see where. If the number holds, you can rebuild it yourself.

## Repository layout

```
naia-research/
├── .agents/
│   ├── context/            # Rules and index for AI tooling
│   ├── toolkit/            # metrics.py, transcribe.py, report.py, run_baseline.py
│   ├── seed_data/          # stress-test seed (ko_eval_seed.py) + cascade-input clips (ko_input_wavs/) + reference audio
│   └── vendor_baselines/   # VoxCPM2 Korean baseline: eval script, audio, results, summary
├── model/
│   └── fine-tuning/
│       └── deadpool-lora-demo/   # LLM swap kit: LoRA training → export → load into Naia Omni
├── .users/                 # Human-readable mirror (en primary, ko)
├── LICENSE                 # Apache 2.0 (code)
└── CONTEXT-LICENSE         # CC-BY-SA 4.0 (docs and data)
```

## Getting started

Start by reading the VoxCPM2 baseline: open
`.agents/vendor_baselines/voxcpm2_ko_30prompts/results.jsonl` to see per-prompt
scores and `summary.json` for the aggregate. That is the shortest path to
understanding what this repo measures and why.

To score your own generated audio:

```bash
git clone https://github.com/nextain/naia-research.git
cd naia-research/.agents/toolkit

# 1. Transcribe generated wavs with faster-whisper
python transcribe.py --results path/to/results.jsonl --out transcripts.jsonl --model large-v3

# 2. Turn transcripts into metrics
python metrics.py --results path/to/results.jsonl --transcripts transcripts.jsonl --out metrics.json

# 3. Render a Markdown report
python report.py --metrics metrics.json --out report.md
```

You supply the `results.jsonl` (your pipeline's per-turn text, audio paths, and
timing). `run_baseline.py` shows how we generate ours, but note it imports a
bridge module (`bridge_gemini_minicpm`) that lives in our private serving stack
and is not shipped here — so treat it as a reference, not a turnkey script.

To try the LLM swap demo instead, follow
`model/fine-tuning/deadpool-lora-demo/README.md` end to end.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). If you add a
document, please mirror it under `.users/{lang}/`.

## Roadmap

Three areas are planned but not written yet. The directories do not exist in the
tree until there is real content to put in them, so nothing here promises more
than it delivers:

- **Methodology writeups** — prose definitions of each metric (why we measure
  leak, truncation, time to first audio packet, and how the thresholds were
  chosen). The definitions live as docstrings in `metrics.py` today.
- **Paradigm essay** — the longer argument for "verified external base plus our
  own stack" rather than fine-tuning our way to a benchmark.
- **Integration guides** — how to wire the cascade into a live setup (for
  example LiveKit talking to our Korean serving layer).

## Origin and license

This is the public slice of Korean voice cascade R&D done in the private
[naia-labs](https://github.com/nextain/naia-labs) repo. The production runtime
lives in [naia-agent](https://github.com/nextain/naia-agent) and
[naia-model-infra](https://github.com/nextain/naia-model-infra).

Code is Apache 2.0 (`LICENSE`). Documentation and evaluation data are CC-BY-SA
4.0 (`CONTEXT-LICENSE`).
