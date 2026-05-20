"""faster-whisper 로 응답 wav 들을 KO transcribe.

Usage:
    python transcribe.py \
        --results ../runs/dryrun/results.jsonl \
        --out ../runs/dryrun/transcripts.jsonl \
        --model large-v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model name (large-v3, large-v3-turbo, base, etc.)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--compute-type", default="float16",
                    help="float16 (GPU), int8 (CPU)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--language", default="ko",
                    help='ISO code (ko/en/zh) or "auto" for multilingual auto-detect')
    args = ap.parse_args()
    lang_arg = None if args.language.lower() in ("auto", "none", "") else args.language

    print(f"=== loading Whisper {args.model} on {args.device} ({args.compute_type}) ===")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        device_index=args.gpu_id,
    )

    with args.results.open() as f:
        items = [json.loads(line) for line in f if line.strip()]

    out_lines = []
    print(f"transcribing {len(items)} items...")
    for i, it in enumerate(items, 1):
        pid = it["id"]
        audio_path = it.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            out_lines.append({
                "id": pid,
                "transcript": "",
                "transcript_segments": [],
                "language": None,
                "language_probability": 0.0,
                "no_speech": True,
            })
            print(f"[{i}/{len(items)}] {pid:14s}  NO-AUDIO")
            continue

        segments_iter, info = model.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        segments = []
        full_text = []
        for s in segments_iter:
            segments.append({
                "start": s.start, "end": s.end, "text": s.text,
                "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            })
            full_text.append(s.text)
        transcript = "".join(full_text).strip()

        out_lines.append({
            "id": pid,
            "input_text": it.get("input_text"),
            "server_text": it.get("text"),
            "audio_path": audio_path,
            "transcript": transcript,
            "transcript_segments": segments,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "no_speech": len(segments) == 0,
        })
        print(f"[{i}/{len(items)}] {pid:14s}  lang={info.language}({info.language_probability:.2f})  "
              f"text='{transcript[:50]}'")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for line in out_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\n=== done → {args.out} ===")


if __name__ == "__main__":
    main()
