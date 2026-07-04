"""Markdown report 생성 (per-case 표 + 있으면 생성 wav 링크).

Usage:
    python report.py \
        --metrics ../runs/dryrun/metrics.json \
        --out ../reports/dryrun.md

또는 여러 run 비교:
    python report.py \
        --metrics ../runs/run_current/metrics.json ../runs/run_clean_short/metrics.json \
        --out ../reports/compare.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x*100:.1f}%"


def fmt_num(x: float | int | None, digits: int = 0) -> str:
    if x is None:
        return "—"
    if digits == 0:
        return f"{x:.0f}"
    return f"{x:.{digits}f}"


def render_single(run_label: str, metrics: dict, meta: dict, run_dir: Path) -> str:
    agg = metrics["aggregate"]
    per_persona = metrics.get("per_persona", {})
    cases = metrics["per_case"]

    out = []
    out.append(f"# {run_label}")
    out.append("")
    out.append(f"- **system_prompt_id**: `{meta.get('system_prompt_id', '?')}`")
    out.append(f"- **gateway**: {meta.get('gateway', '?')}")
    out.append(f"- **n_total / n_ok**: {agg['n_total']} / {agg['n_ok']}")
    out.append(f"- **elapsed_total_s**: {meta.get('elapsed_total_s', 0):.1f}")
    out.append(f"- **manifest**: `{meta.get('manifest', '?')}`")
    out.append("")
    out.append("## Aggregate metrics")
    out.append("")
    out.append("| metric | value |")
    out.append("|--------|-------|")
    out.append(f"| empty_response_rate     | {fmt_pct(agg['empty_response_rate'])} |")
    out.append(f"| truncation_rate         | {fmt_pct(agg['truncation_rate'])} |")
    out.append(f"| turn_completion_rate    | {fmt_pct(agg['turn_completion_rate'])} |")
    out.append(f"| lang_leak_rate (mean)   | {fmt_pct(agg['lang_leak_rate'])} |")
    out.append(f"| lang_leak_rate (p95)    | {fmt_pct(agg['lang_leak_rate_p95'])} |")
    out.append(f"| audio_TTFP_ms p50 / p95 | {fmt_num(agg.get('audio_TTFP_ms_p50'))} / {fmt_num(agg.get('audio_TTFP_ms_p95'))} |")
    cer_p50 = agg.get("tts_quality_cer_p50")
    cer_p95 = agg.get("tts_quality_cer_p95")
    out.append(f"| tts_quality_cer p50/p95 | {fmt_num(cer_p50, 3)} / {fmt_num(cer_p95, 3)} |")
    out.append("")
    out.append(f"*ttfp note: {agg.get('ttfp_note', '')}*")
    out.append("")

    if per_persona:
        out.append("## Per-persona breakdown")
        out.append("")
        out.append("| persona | n | empty | truncate | complete | leak | cer p50 |")
        out.append("|---------|---|-------|----------|----------|------|---------|")
        for p, m in per_persona.items():
            out.append(
                f"| {p} | {m['n']} | {fmt_pct(m['empty_rate'])} | "
                f"{fmt_pct(m['truncation_rate'])} | {fmt_pct(m['turn_completion_rate'])} | "
                f"{fmt_pct(m['lang_leak_rate_mean'])} | {fmt_num(m.get('tts_quality_cer_p50'), 3)} |"
            )
        out.append("")

    out.append("## Per-case detail")
    out.append("")
    out.append("| id | input | server_text | whisper_text | leak | trunc | cer | wav |")
    out.append("|----|-------|-------------|--------------|------|-------|-----|-----|")
    for c in cases:
        pid = c["id"]
        # md 셀에서 줄바꿈 / 파이프 escape
        def sane(s: str) -> str:
            return (s or "").replace("|", "\\|").replace("\n", " ")
        # input 은 results.jsonl 에 input_text 가 있으면 그걸. 아니면 per_case 에 빠짐 — server_text 에서 추론 X
        input_text = sane(c.get("server_text_input_text", ""))
        # wav link (run_dir 상대)
        wav_link = ""
        wav_candidate = run_dir / "out_wavs" / f"{pid}.wav"
        if wav_candidate.exists():
            try:
                wav_link = f"[wav]({wav_candidate.resolve()})"
            except Exception:
                wav_link = ""
        out.append(
            f"| {pid} | {input_text} | {sane(c['server_text'])[:80]} | "
            f"{sane(c['whisper_text'])[:80]} | "
            f"{fmt_pct(c['lang_leak_ratio'])} | {'Y' if c['truncated'] else ''} | "
            f"{fmt_num(c.get('audio_text_cer'), 3)} | {wav_link} |"
        )
    out.append("")
    return "\n".join(out)


def render_compare(runs: list[tuple[str, dict, dict]]) -> str:
    """여러 run 의 aggregate 만 column 비교."""
    out = []
    out.append("# Variant comparison")
    out.append("")
    metrics_keys = [
        ("empty_response_rate", "empty %", lambda v: fmt_pct(v)),
        ("truncation_rate", "trunc %", lambda v: fmt_pct(v)),
        ("turn_completion_rate", "complete %", lambda v: fmt_pct(v)),
        ("lang_leak_rate", "leak %", lambda v: fmt_pct(v)),
        ("audio_TTFP_ms_p50", "ttfp p50 ms", lambda v: fmt_num(v)),
        ("audio_TTFP_ms_p95", "ttfp p95 ms", lambda v: fmt_num(v)),
        ("tts_quality_cer_p50", "cer p50", lambda v: fmt_num(v, 3)),
    ]
    header = "| metric | " + " | ".join(r[0] for r in runs) + " |"
    sep = "|" + "---|" * (len(runs) + 1)
    out.append(header)
    out.append(sep)
    for k, label, fmt in metrics_keys:
        row_vals = []
        for _, m, _ in runs:
            v = m["aggregate"].get(k)
            row_vals.append(fmt(v) if v is not None else "—")
        out.append(f"| {label} | " + " | ".join(row_vals) + " |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, nargs="+", required=True,
                    help="하나면 single report, 여러 개면 variant compare")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = []
    for mp in args.metrics:
        run_dir = mp.parent  # eval/runs/<name>/
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metrics = json.loads(mp.read_text())
        runs.append((run_dir.name, metrics, meta))

    sections = []
    if len(runs) > 1:
        sections.append(render_compare(runs))
    for label, m, meta in runs:
        run_dir = Path(meta.get("manifest", ".")).parent.parent  # heuristic; fallback below
        # 더 안정적인 run_dir = metrics file's parent
        for mp in args.metrics:
            if json.loads(mp.read_text()) is m or json.loads(mp.read_text()).get("aggregate") == m["aggregate"]:
                run_dir = mp.parent
                break
        sections.append(render_single(label, m, meta, run_dir))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n---\n\n".join(sections))
    print(f"=== report → {args.out} ===")
    print(f"  open with: code {args.out}")


if __name__ == "__main__":
    main()
