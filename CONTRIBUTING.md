# Contributing to naia-research

Thanks for looking. This is an early-stage repo, so the most valuable
contributions are the ones that keep it honest: reproduce a measurement, point
out where a number does not hold up, or add a metric that catches something our
current ones miss.

## What this repo is (and is not)

naia-research is the public slice of Korean voice cascade R&D. It ships an
evaluation toolkit, a Korean evaluation seed, one reproducible VoxCPM2 baseline,
and an LLM-swap fine-tuning demo. It is not the production runtime — that lives
in the private naia-agent and naia-model-infra repos — and it deliberately does
not promise features that are not in the tree. If a directory is empty in the
roadmap, we do not create it until there is real content.

## Ways to contribute

- **Reproduce a baseline.** Run the toolkit against your own generated audio and
  report whether the numbers line up. Disagreements are useful; open an issue
  with your `results.jsonl` shape and the metrics you got.
- **Improve a metric.** If `metrics.py` measures the wrong thing, or misses a
  failure mode (a leak class, a truncation pattern), propose a change with a
  small example that shows the gap.
- **Add evaluation prompts.** New Korean stress cases (minimal pairs, numerals,
  loanwords, mixed script) are welcome. Keep them short and single-speaker.
- **Write up methodology.** The metric definitions currently live as docstrings.
  A clear prose explanation of why a metric exists and how its threshold was
  chosen is exactly what the roadmap `methodology/` area is for.

## Ground rules

- **No local paths or secrets.** Do not commit absolute paths
  (`/home/...`, `/var/...`), internal IPs, or credentials. Evaluation data must
  reference files relative to the repo so others can reproduce it.
- **Measurements must be reproducible.** Report the sample size, the exact
  command, and the raw output. Do not round a claim into something the data does
  not support.
- **Mirror docs you add under `.users/`.** The human-readable context lives in
  `.users/{lang}/`, with English primary and Korean as the first mirror. If you
  add or change a document there, update the matching language copy so the two
  stay in step. (Root files like this one and `README.md` are surfaced through
  the `.users/{en,ko}/context/` entry points rather than copied one to one.)

## Licensing

By contributing you agree that code is licensed under Apache 2.0 (`LICENSE`) and
that documentation and evaluation data are licensed under CC-BY-SA 4.0
(`CONTEXT-LICENSE`).
