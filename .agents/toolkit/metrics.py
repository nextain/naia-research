"""Metrics — Day 0 갱신 (1차+2차 cross review 권고 반영).

Inputs:
    results.jsonl     — bridge/run_baseline 산출 (server text + audio + per-turn timing)
    transcripts.jsonl — transcribe.py 산출 (Whisper KO transcript + segments + avg_logprob)

Metrics (Day 0 확정):
    핵심 (Codex+Gemini 권고)
    - empty_response_rate
    - truncation_rate
    - turn_completion_rate
    - text_purity_rate         = 1 - lang_leak_rate (양의 metric)
    - lang_leak_rate           server text 의 비-한국어 (한자/일본어/영문) 문자 비율
    - tts_quality_cer          server text vs Whisper transcript CER. ★ sane turns 만 (lang_leak ≤ 5%)
                                Talker fidelity 측정 (server text 가 정상일 때만)
    - audio_TTFP_ms            per-turn 첫 audio 패킷 wall (현재 = minicpm_ttfp_ms)
    - whisper_avg_logprob      Whisper segments 의 avg_logprob 평균 (ASR confidence)

    Per-turn-index breakdown (Codex 가설: turn accumulation):
    - lang_leak_by_turn_index[i]

    Legacy (보조 — Q8 권고로 deprecate):
    - audio_text_cer           모든 turn 의 server text vs Whisper CER

Usage:
    python metrics.py \
        --results ../runs/<dir>/results.jsonl \
        --transcripts ../runs/<dir>/transcripts.jsonl \
        --out ../runs/<dir>/metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# 한국어 종결어미 (느슨한 정의)
KO_TERMINATORS = (
    "다.", "다", "까.", "까?", "까", "요.", "요!", "요?", "요",
    "네.", "네!", "네?", "네", "지.", "지?", "지", "세요.", "세요!",
    "세요?", "세요", "네요.", "네요?", "네요!", "네요",
    "어요.", "어요?", "어요!", "어요", "아요.", "아요?", "아요!", "아요",
)

# 비한국어 detection
RE_CJK_HAN = re.compile(r"[一-鿿㐀-䶿]")
RE_JP_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
RE_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
RE_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# Day 0 ship threshold (cross review 합의)
SANE_TURN_LEAK_THRESHOLD = 0.05
SHIP_THRESHOLDS = {
    "general_canary": {
        "text_purity_rate": 0.95,
        "turn_completion_rate": 0.85,
        "tts_quality_cer": 0.30,
    },
    "vertical_production": {
        "text_purity_rate": 0.975,
        "turn_completion_rate": 0.90,
        "tts_quality_cer": 0.25,
    },
}


def lang_leak_ratio(text: str) -> tuple[float, int, int, int, int]:
    """returns (ratio, n_han, n_kana, n_latin_word, n_cyrillic)"""
    total_chars = len(text)
    n_han = len(RE_CJK_HAN.findall(text))
    n_kana = len(RE_JP_KANA.findall(text))
    n_latin = sum(len(m.group(0)) for m in RE_LATIN_WORD.finditer(text))
    n_cyrillic = len(RE_CYRILLIC.findall(text))
    if total_chars == 0:
        return 0.0, 0, 0, 0, 0
    leaked_chars = n_han + n_kana + n_latin + n_cyrillic
    return leaked_chars / total_chars, n_han, n_kana, n_latin, n_cyrillic


def is_truncated(text: str) -> bool:
    if not text:
        return True
    s = text.rstrip()
    if not s:
        return True
    if s.endswith((".", "?", "!")):
        return False
    for t in KO_TERMINATORS:
        if s.endswith(t):
            return False
    return True


def cer(ref: str, hyp: str) -> float:
    ref = ref.strip()
    hyp = hyp.strip()
    if not ref:
        return 1.0 if hyp else 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[n] / m


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if c != f else s[f]


def turn_index_from_id(case_id: str) -> int | None:
    """`daily_rep1_t2` → 2. id pattern 추출 실패 시 None."""
    m = re.search(r"_t(\d+)$", case_id)
    return int(m.group(1)) if m else None


def whisper_logprob(transcript_record: dict) -> float | None:
    segs = transcript_record.get("transcript_segments", [])
    logprobs = [s.get("avg_logprob") for s in segs if s.get("avg_logprob") is not None]
    return statistics.mean(logprobs) if logprobs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-text-len", type=int, default=2)
    ap.add_argument("--sane-leak-threshold", type=float, default=SANE_TURN_LEAK_THRESHOLD,
                    help="이 비율 이하 lang_leak 인 turn 만 sane (TTS quality CER 계산용)")
    args = ap.parse_args()

    results = [json.loads(l) for l in args.results.open() if l.strip()]
    transcripts = {json.loads(l)["id"]: json.loads(l) for l in args.transcripts.open() if l.strip()}

    per_case = []
    for r in results:
        pid = r["id"]
        text = r.get("text") or ""
        tr = transcripts.get(pid, {})
        whisper_text = tr.get("transcript", "")
        wh_logprob = whisper_logprob(tr)
        leak_ratio, n_han, n_kana, n_latin, n_cyr = lang_leak_ratio(text)
        truncated = is_truncated(text)
        empty = len(text.strip()) < args.min_text_len
        # 모든 case 의 audio_text_cer (legacy)
        audio_cer = cer(text, whisper_text) if text and whisper_text else None
        # sane turn 만 tts_quality_cer (server text 가 lang_leak ≤ threshold)
        is_sane = (not empty) and (leak_ratio <= args.sane_leak_threshold)
        tts_quality = cer(text, whisper_text) if (is_sane and whisper_text) else None
        per_case.append({
            "id": pid,
            "persona": r.get("persona"),
            "turn_index": turn_index_from_id(pid),
            "ok": bool(r.get("ok")),
            "empty": empty,
            "truncated": truncated,
            "is_sane": is_sane,
            # timing
            "elapsed_ms_wall": r.get("elapsed_ms_wall") or r.get("wall_total_ms"),
            "audio_TTFP_ms": r.get("ttfp_ms"),
            "text_TTFP_ms": r.get("ttfp_ms"),  # bridge: text+audio TTFP separation 미구현, 동일 proxy
            "tokens": r.get("tokens"),
            "text_len": len(text),
            "server_text": text,
            "whisper_text": whisper_text,
            "whisper_avg_logprob": wh_logprob,
            "lang_leak_ratio": leak_ratio,
            "text_purity_ratio": 1.0 - leak_ratio,
            "n_han": n_han, "n_kana": n_kana, "n_latin": n_latin, "n_cyrillic": n_cyr,
            "audio_text_cer": audio_cer,        # legacy
            "tts_quality_cer": tts_quality,     # sane only
            "turn_completion": bool(r.get("ok")) and not empty and not truncated,
            # S1-pre① 실시간성·안정성
            "ttfa_ms": r.get("ttfa_ms"),
            "deaf_ms": r.get("deaf_ms"),
            "audio_span_s": r.get("audio_span_s"),
            "n_speak_results": r.get("n_speak_results"),
            "tts_backend_counts": r.get("tts_backend_counts") or {},
            "closed_before_eot": bool(r.get("closed_before_eot")),
        })

    total = len(per_case)
    ok_cases = [c for c in per_case if c["ok"]]
    sane_cases = [c for c in ok_cases if c["is_sane"]]
    n_ok = len(ok_cases)
    n_sane = len(sane_cases)
    n_empty = sum(1 for c in ok_cases if c["empty"])
    n_truncated = sum(1 for c in ok_cases if c["truncated"])
    n_complete = sum(1 for c in ok_cases if c["turn_completion"])
    leaks = [c["lang_leak_ratio"] for c in ok_cases]
    purity = [c["text_purity_ratio"] for c in ok_cases]
    legacy_cer = [c["audio_text_cer"] for c in ok_cases if c["audio_text_cer"] is not None]
    tts_cer = [c["tts_quality_cer"] for c in sane_cases if c["tts_quality_cer"] is not None]
    audio_ttfp = [c["audio_TTFP_ms"] for c in ok_cases if c.get("audio_TTFP_ms")]
    text_ttfp = [c["text_TTFP_ms"] for c in ok_cases if c.get("text_TTFP_ms")]
    wall = [c["elapsed_ms_wall"] for c in ok_cases if c.get("elapsed_ms_wall")]
    whisper_lp = [c["whisper_avg_logprob"] for c in ok_cases if c.get("whisper_avg_logprob") is not None]

    aggregate = {
        "n_total": total,
        "n_ok": n_ok,
        "n_sane_turns": n_sane,
        "sane_rate": n_sane / n_ok if n_ok else 0.0,
        "empty_response_rate": n_empty / n_ok if n_ok else 0.0,
        "truncation_rate": n_truncated / n_ok if n_ok else 0.0,
        "turn_completion_rate": n_complete / n_ok if n_ok else 0.0,
        "lang_leak_rate": statistics.mean(leaks) if leaks else 0.0,
        "lang_leak_rate_p95": percentile(leaks, 0.95),
        "text_purity_rate": statistics.mean(purity) if purity else 0.0,
        "text_purity_rate_p05": percentile(purity, 0.05),
        # ★ TTS Quality (sane turns only)
        "tts_quality_cer_p50": percentile(tts_cer, 0.50) if tts_cer else None,
        "tts_quality_cer_p95": percentile(tts_cer, 0.95) if tts_cer else None,
        "tts_quality_cer_mean": statistics.mean(tts_cer) if tts_cer else None,
        # TTFP per-turn
        "audio_TTFP_ms_p50": percentile(audio_ttfp, 0.50) if audio_ttfp else None,
        "audio_TTFP_ms_p95": percentile(audio_ttfp, 0.95) if audio_ttfp else None,
        "text_TTFP_ms_p50": percentile(text_ttfp, 0.50) if text_ttfp else None,
        "text_TTFP_ms_p95": percentile(text_ttfp, 0.95) if text_ttfp else None,
        "wall_total_ms_p50": percentile(wall, 0.50) if wall else None,
        # ASR confidence
        "whisper_avg_logprob_mean": statistics.mean(whisper_lp) if whisper_lp else None,
        # Legacy
        "audio_text_cer_p50_legacy": percentile(legacy_cer, 0.50) if legacy_cer else None,
        "audio_text_cer_p95_legacy": percentile(legacy_cer, 0.95) if legacy_cer else None,
        "ttfp_note": "text_TTFP / audio_TTFP 분리 미구현 — 현재는 둘 다 wall-elapsed proxy (bridge level).",
        "tts_quality_cer_note": f"sane turns only (lang_leak ≤ {args.sane_leak_threshold*100:.0f}%). pipeline 붕괴 turn 제외.",
    }

    # ── S1-pre① 실시간성·안정성 (VoxCPM2 vs native backend split) ──
    ttfa = [c["ttfa_ms"] for c in ok_cases if c.get("ttfa_ms") is not None]
    deaf = [c["deaf_ms"] for c in ok_cases if c.get("deaf_ms") is not None]
    bk_total: dict[str, int] = {}
    for c in ok_cases:
        for k, v in (c.get("tts_backend_counts") or {}).items():
            bk_total[k] = bk_total.get(k, 0) + int(v)
    n_speak_total = sum(bk_total.values())
    n_vox_ok = bk_total.get("voxcpm2", 0)
    n_buffering = bk_total.get("voxcpm2_buffering", 0)
    n_fallback = bk_total.get("voxcpm2_fallback", 0)
    n_native = bk_total.get("native", 0)
    n_closed_early = sum(1 for c in ok_cases if c.get("closed_before_eot"))
    backend_mode = "voxcpm2" if (n_vox_ok + n_buffering + n_fallback) > n_native else "native"

    realtime_stability = {
        "backend_mode": backend_mode,
        # 실시간성: ttfa = 첫 user chunk → 첫 가청 오디오 (buffering 무음 제외, 진짜 "들리기까지")
        "ttfa_ms_p50": percentile(ttfa, 0.50) if ttfa else None,
        "ttfa_ms_p95": percentile(ttfa, 0.95) if ttfa else None,
        # deaf = SPEAK 토큰 시작 → 첫 가청 오디오 무음 구간 (VoxCPM2 절-버퍼링 대가, native≈0)
        "deaf_ms_p50": percentile(deaf, 0.50) if deaf else None,
        "deaf_ms_p95": percentile(deaf, 0.95) if deaf else None,
        # 안정성
        "n_speak_results_total": n_speak_total,
        "tts_backend_counts": bk_total,
        "voxcpm2_substitution_rate": n_vox_ok / n_speak_total if n_speak_total else None,
        "voxcpm2_buffering_rate": n_buffering / n_speak_total if n_speak_total else None,
        "voxcpm2_fallback_rate": n_fallback / n_speak_total if n_speak_total else None,
        "closed_before_eot_rate": n_closed_early / n_ok if n_ok else 0.0,
        "note": ("ttfa=첫 가청 오디오 latency(buffering 무음 제외). "
                 "deaf=SPEAK 시작→첫 가청(VoxCPM2 절-버퍼링 대가, native≈0). "
                 "fallback_rate=VoxCPM2 합성 실패→원본 통과 비율(안정성). "
                 "closed_before_eot_rate=eot 전 ws 끊김=세션 비정상종료(/health-kill 등) proxy."),
    }

    # Per-persona breakdown
    personas = sorted({c.get("persona") or "?" for c in ok_cases})
    persona_breakdown = {}
    for p in personas:
        sub = [c for c in ok_cases if c.get("persona") == p]
        sub_sane = [c for c in sub if c["is_sane"]]
        if not sub:
            continue
        persona_breakdown[p] = {
            "n": len(sub),
            "n_sane": len(sub_sane),
            "empty_rate": sum(1 for c in sub if c["empty"]) / len(sub),
            "truncation_rate": sum(1 for c in sub if c["truncated"]) / len(sub),
            "turn_completion_rate": sum(1 for c in sub if c["turn_completion"]) / len(sub),
            "lang_leak_rate_mean": statistics.mean(c["lang_leak_ratio"] for c in sub),
            "text_purity_rate": 1.0 - statistics.mean(c["lang_leak_ratio"] for c in sub),
            "tts_quality_cer_p50": percentile(
                [c["tts_quality_cer"] for c in sub_sane if c["tts_quality_cer"] is not None], 0.50
            ) if sub_sane else None,
        }

    # Per-turn-index breakdown (Codex 가설 검증)
    turn_indices = sorted({c.get("turn_index") for c in ok_cases if c.get("turn_index") is not None})
    turn_index_breakdown = {}
    for ti in turn_indices:
        sub = [c for c in ok_cases if c.get("turn_index") == ti]
        if not sub:
            continue
        turn_index_breakdown[f"turn_{ti}"] = {
            "n": len(sub),
            "lang_leak_rate_mean": statistics.mean(c["lang_leak_ratio"] for c in sub),
            "text_purity_rate": 1.0 - statistics.mean(c["lang_leak_ratio"] for c in sub),
            "truncation_rate": sum(1 for c in sub if c["truncated"]) / len(sub),
            "turn_completion_rate": sum(1 for c in sub if c["turn_completion"]) / len(sub),
        }

    # Ship verdicts (general canary + vertical)
    ship_verdict = {}
    for tier, th in SHIP_THRESHOLDS.items():
        pass_purity = aggregate["text_purity_rate"] >= th["text_purity_rate"]
        pass_complete = aggregate["turn_completion_rate"] >= th["turn_completion_rate"]
        pass_cer = (aggregate["tts_quality_cer_p50"] is not None
                    and aggregate["tts_quality_cer_p50"] <= th["tts_quality_cer"])
        ship_verdict[tier] = {
            "threshold": th,
            "actual": {
                "text_purity_rate": aggregate["text_purity_rate"],
                "turn_completion_rate": aggregate["turn_completion_rate"],
                "tts_quality_cer_p50": aggregate["tts_quality_cer_p50"],
            },
            "pass_purity": pass_purity,
            "pass_completion": pass_complete,
            "pass_cer": pass_cer,
            "ship_ready": pass_purity and pass_complete and pass_cer,
        }

    out = {
        "aggregate": aggregate,
        "realtime_stability": realtime_stability,
        "ship_verdict": ship_verdict,
        "per_persona": persona_breakdown,
        "per_turn_index": turn_index_breakdown,
        "per_case": per_case,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"=== metrics for {args.results.parent.name} ===")
    print(f"  n_total / n_ok / n_sane    = {total} / {n_ok} / {n_sane}")
    print(f"  empty_response_rate        = {aggregate['empty_response_rate']:.1%}")
    print(f"  truncation_rate            = {aggregate['truncation_rate']:.1%}")
    print(f"  turn_completion_rate       = {aggregate['turn_completion_rate']:.1%}")
    print(f"  text_purity_rate (mean/p05)= {aggregate['text_purity_rate']:.1%} / {aggregate['text_purity_rate_p05']:.1%}")
    print(f"  lang_leak_rate (mean/p95)  = {aggregate['lang_leak_rate']:.1%} / {aggregate['lang_leak_rate_p95']:.1%}")
    if aggregate["tts_quality_cer_p50"] is not None:
        print(f"  tts_quality_cer (p50/p95)  = {aggregate['tts_quality_cer_p50']:.3f} / {aggregate['tts_quality_cer_p95']:.3f}  (sane n={n_sane})")
    else:
        print(f"  tts_quality_cer            = N/A (sane turns 0)")
    if aggregate["audio_TTFP_ms_p50"] is not None:
        print(f"  audio_TTFP_ms (p50/p95)    = {aggregate['audio_TTFP_ms_p50']:.0f} / {aggregate['audio_TTFP_ms_p95']:.0f}")
    if aggregate["whisper_avg_logprob_mean"] is not None:
        print(f"  whisper_avg_logprob (mean) = {aggregate['whisper_avg_logprob_mean']:.3f}")
    print()
    rs = realtime_stability
    _ms = lambda v: f"{v:.0f}" if v is not None else "N/A"
    _pct = lambda v: f"{v:.1%}" if v is not None else "N/A"
    print(f"  ── 실시간성·안정성 (backend={rs['backend_mode']}) ──")
    print(f"  ttfa_ms (p50/p95)          = {_ms(rs['ttfa_ms_p50'])} / {_ms(rs['ttfa_ms_p95'])}   (첫 가청 오디오)")
    print(f"  deaf_ms (p50/p95)          = {_ms(rs['deaf_ms_p50'])} / {_ms(rs['deaf_ms_p95'])}   (SPEAK→첫 가청 무음)")
    print(f"  tts_backend_counts         = {rs['tts_backend_counts']}")
    print(f"  voxcpm2_fallback_rate      = {_pct(rs['voxcpm2_fallback_rate'])}   (합성 실패→원본 통과)")
    print(f"  closed_before_eot_rate     = {rs['closed_before_eot_rate']:.1%}   (세션 비정상종료 proxy)")
    print()
    print("  Ship verdict:")
    for tier, v in ship_verdict.items():
        flag = "✓ ship-ready" if v["ship_ready"] else "✗ not yet"
        print(f"    {tier:24s} {flag}")
        for k in ("pass_purity", "pass_completion", "pass_cer"):
            print(f"        {k}: {v[k]}")
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
