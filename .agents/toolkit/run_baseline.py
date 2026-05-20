"""Baseline runner: 2 scenario × 3턴 × 2 repeat = 12 conversation 자동 실행.

각 conversation = bridge_gemini_minicpm.py 의 run_bridge() 호출.
끝나면 transcribe + metric + report 자동 chain.

Usage:
    python run_baseline.py --out ../runs/baseline_$(date +%Y%m%d-%H%M%S) --variant current
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")

# bridge module import (run_bridge 의 args namespace 와 동일)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bridge_gemini_minicpm import run_bridge, KO_SYSTEM_PROMPT

# system prompt variants (Codex 권고 + 우리 hypothesis)
SYS_PROMPTS = {
    "current": KO_SYSTEM_PROMPT,
    "clean_short": "당신은 한국어 음성 비서입니다. 한국어로만 짧고 자연스럽게 답하세요.",
    "persona_friendly": "당신은 친절한 한국인 비서입니다. 1-2 문장으로 자연스럽게 한국어로 답하세요.",
    "museum_vertical": "당신은 박물관 안내원입니다. 한국어로만 친절하고 정확하게 1-2 문장으로 안내하세요.",
    # Codex 권고 (1차 review): clean_short purity + persona completion 결합
    "clean_plus_persona": (
        "당신은 친절한 한국어 음성 비서입니다. 항상 자연스러운 한국어로만 "
        "1~2문장으로 답하세요. 고유명사와 외래어도 한국어 발음대로 풀어 말하고, "
        "답변은 끝까지 마무리하세요."
    ),
    # 2026-05-15 레버 A: 한국어-only 강제 + '1~2문장' 완화(완결성↑). baseline(purity .702/leak .298) 대비.
    "ko_strict": (
        "당신은 자연스러운 한국어로 대화하는 음성 비서입니다. "
        "반드시 한국어로만 말하세요 — 한자, 중국어, 일본어 문자나 단어를 "
        "절대 섞지 마세요. 외래어와 고유명사도 한글 발음으로 풀어 말하세요. "
        "답변은 2~3문장으로 자연스럽고 완결되게, 문장을 끝까지 마무리하세요."
    ),
}

# scenario 페르소나 set (Codex 권고: daily + museum 만, counsel/trivia v2)
SCENARIOS = {
    "daily": {
        "persona": "당신은 친구와 일상 대화 중인 한국인입니다. 메타 설명 없이, 1-2 문장 한국어 발화만 하세요.",
        "first_turns": [
            "친구에게 오늘 날씨가 쌀쌀하다고 인사하면서 짧게 한 마디 해.",
            "친구에게 오늘 점심 뭐 먹었냐고 자연스럽게 물어봐.",
            "친구에게 주말에 뭐 할 거냐고 가볍게 물어봐.",
        ],
    },
    "museum": {
        "persona": "당신은 박물관에 방문한 한국인 관람객입니다. 안내원에게 자연스럽게 한국어로 1-2 문장 질문하세요.",
        "first_turns": [
            "박물관 안내원에게 가장 유명한 작품이 무엇인지 자연스럽게 물어봐.",
            "조선시대 도자기 전시가 어디 있는지 안내원에게 물어봐.",
            "안내원에게 관람 동선을 추천해 달라고 물어봐.",
        ],
    },
    # Codex 권고: 비주력 vertical micro sanity (3-turn 1 rep). safety confound 회피 — 가벼운 톤.
    "counsel": {
        "persona": "당신은 가벼운 고민을 상담사에게 이야기하는 한국인입니다. 일상적 스트레스 수준의 가벼운 주제만, 자연스럽게 한국어로 1-2 문장씩 말하세요. 위기/자해/의료 표현은 절대 하지 마세요.",
        "first_turns": [
            "상담사에게 요즘 일이 많아서 좀 피곤하다고 가볍게 운을 떼.",
        ],
    },
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="current", choices=list(SYS_PROMPTS.keys()))
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--scenarios", nargs="+", default=["daily", "museum"])
    ap.add_argument("--repeats-per-scenario", type=int, default=2,
                    help="각 scenario 당 first_turn 다른 시드로 N번 (max=len(first_turns))")
    ap.add_argument("--gateway", default="wss://localhost:8006")
    ap.add_argument("--ref-audio", default="assets/ref_audio/ref_ko_485.wav")
    ap.add_argument("--voice", default="Aoede")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    minicpm_sys_prompt = SYS_PROMPTS[args.variant]
    print(f"=== baseline run {args.out.name} ===")
    print(f"  variant:   {args.variant}")
    print(f"  scenarios: {args.scenarios}")
    print(f"  turns:     {args.turns}")
    print(f"  repeats:   {args.repeats_per_scenario}")
    print()

    all_conv = []
    t0 = time.time()

    for scenario_id in args.scenarios:
        scn = SCENARIOS[scenario_id]
        seeds = scn["first_turns"][:args.repeats_per_scenario]
        for rep_idx, first_turn in enumerate(seeds, 1):
            conv_id = f"{scenario_id}_rep{rep_idx}"
            conv_dir = args.out / conv_id
            conv_dir.mkdir(parents=True, exist_ok=True)

            print(f"  → {conv_id}: '{first_turn[:40]}...'")
            t_conv = time.time()

            bridge_args = SimpleNamespace(
                out=conv_dir,
                persona=scn["persona"],
                first_turn=first_turn,
                minicpm_system_prompt=minicpm_sys_prompt,
                voice=args.voice,
                turns=args.turns,
                gateway=args.gateway,
                ref_audio=args.ref_audio,
                silence_padding_chunks=30,
            )

            try:
                await run_bridge(bridge_args)
            except Exception as e:
                print(f"    FAIL: {e}")
                all_conv.append({"id": conv_id, "scenario": scenario_id, "error": str(e)})
                continue

            scenario_json = conv_dir / "scenario.json"
            if scenario_json.exists():
                conv_meta = json.loads(scenario_json.read_text())
                conv_meta["id"] = conv_id
                conv_meta["scenario"] = scenario_id
                conv_meta["repeat"] = rep_idx
                all_conv.append(conv_meta)

            print(f"     done in {time.time()-t_conv:.1f}s")

    summary = {
        "variant": args.variant,
        "scenarios": args.scenarios,
        "turns": args.turns,
        "repeats_per_scenario": args.repeats_per_scenario,
        "n_conversations": len(all_conv),
        "elapsed_total_s": time.time() - t0,
        "conversations": all_conv,
    }
    (args.out / "baseline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Build results.jsonl (metric script 호환) — 각 turn 을 row 로
    results_path = args.out / "results.jsonl"
    with results_path.open("w") as f:
        for conv in all_conv:
            if "per_turn" not in conv:
                continue
            for t in conv["per_turn"]:
                f.write(json.dumps({
                    "id": f"{conv['id']}_t{t['turn']}",
                    "persona": conv.get("scenario"),
                    "input_text": t.get("gemini_transcript", ""),  # Gemini 가 한 말
                    "ok": bool(t.get("minicpm_end_of_turn")) or bool(t.get("minicpm_n_results", 0)),
                    "text": t.get("minicpm_text", ""),
                    "audio_path": t.get("minicpm_wav"),
                    "audio_sample_rate": 24000,
                    "elapsed_ms_wall": t.get("turn_wall_ms"),
                    "duration_ms_server": t.get("minicpm_ttlt_ms"),
                    "tokens": None,
                    "ttfp_ms": t.get("minicpm_ttfp_ms"),
                    # S1-pre① 실시간성·안정성
                    "ttfa_ms": t.get("minicpm_ttfa_ms"),
                    "deaf_ms": t.get("minicpm_deaf_ms"),
                    "audio_span_s": t.get("minicpm_audio_span_s"),
                    "n_speak_results": t.get("minicpm_n_speak_results"),
                    "tts_backend_counts": t.get("minicpm_tts_backend_counts") or {},
                    "closed_before_eot": bool(t.get("minicpm_closed_before_eot")),
                }, ensure_ascii=False) + "\n")
    print(f"\n  results.jsonl: {results_path}")
    print(f"  baseline.json: {args.out / 'baseline.json'}")
    print(f"  elapsed:       {summary['elapsed_total_s']:.1f}s for {len(all_conv)} conversations")


if __name__ == "__main__":
    asyncio.run(main())
