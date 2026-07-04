"""Phase 6: Korean-only listening eval framework for MiniCPM-o.

32 cases across 10 stress buckets. Primary metric is CER (whisper transcript
vs server transcript), with hangul ratio as secondary observability metric.

Designed for:
  • baseline + Phase 7 epoch eval (regression tracking, fixed stratified manual sample)
  • A/B compare across models (--compare)
  • per-bucket pass rate (failures localized to specific stress categories)

Usage (from the eval harness root, inside your serving venv):
  source <your-venv>/bin/activate
  # Full 32-case eval
  python tools/eval_omni_voice.py --output_dir /tmp/eval_phase7a --label phase7a
  # Quick 9-case eval (per-epoch)
  python tools/eval_omni_voice.py --output_dir /tmp/eval_e3 --label epoch3 --quick
  # Bucket-filtered
  python tools/eval_omni_voice.py --output_dir /tmp/eval_min --label min --bucket minimal_pair,vowel
  # Compare two runs
  python tools/eval_omni_voice.py --compare /tmp/eval_baseline /tmp/eval_phase7a

Reframe (2026-04-29): EN/ZH PRIMARY cases removed (Korean-only deliverable);
2 observability cases retained for catastrophic-forgetting trajectory monitoring.
Server transcript (LLM textual response) is the CER reference, since the LLM's
exact reply varies and we measure audio-vs-intent rather than audio-vs-prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# ─────────────────────────────────────────────────────────────────────────────
# Test cases: 30 KO across 8 active buckets + 2 EN/ZH observability + 2
# conditional stubs (emotional, multi_speaker) populated by Phase 7-B.
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Bucket 1: minimal_pair (3) — plain/tense/aspirated stops ─────────────
    {"id": "minimal_pair_01", "bucket": "minimal_pair", "lang": "ko",
     "voice": "ko-KR-SunHiNeural", "manual_sample": True,
     "prompt": "가다와 까다 발음 차이를 한 번씩 짚어 주세요.",
     "system": "당신은 발음 코치입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "minimal_pair_02", "bucket": "minimal_pair", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "달과 딸의 차이를 알려 주세요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "minimal_pair_03", "bucket": "minimal_pair", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "사다와 싸다는 의미가 어떻게 다른가요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 2: vowel (3) — ㅓ/ㅡ/ㅗ/ㅜ confusables ─────────────────────────
    {"id": "vowel_01", "bucket": "vowel", "lang": "ko",
     "voice": "ko-KR-SunHiNeural", "manual_sample": True,
     "prompt": "어머니와 으뜸의 첫 글자 모음을 비교해 주세요.",
     "system": "당신은 한국어 선생님입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "vowel_02", "bucket": "vowel", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "오늘 점심 우리 어디로 갈까요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "vowel_03", "bucket": "vowel", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "음악을 들으면 기분이 어떠세요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 3: numeral (3) — native vs Sino-Korean ────────────────────────
    {"id": "numeral_01", "bucket": "numeral", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "지금 시각이 두 시 삼십 분이에요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "numeral_02", "bucket": "numeral", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "사과 세 개와 배 다섯 개 주문할게요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "numeral_03", "bucket": "numeral", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "오늘 날짜가 이천이십육년 사월 이십구일이에요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 4: loanword (3) — acronyms, English borrowings ────────────────
    {"id": "loanword_01", "bucket": "loanword", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "AI 기술이 정말 빠르게 발전하고 있죠.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "loanword_02", "bucket": "loanword", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "USB 메모리에 파일을 저장했어요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "loanword_03", "bucket": "loanword", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "RTX 3090 GPU 두 개로 학습 중이에요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 5: mixed_script (2) — Hangul + Latin + digits ─────────────────
    {"id": "mixed_script_01", "bucket": "mixed_script", "lang": "ko",
     "voice": "ko-KR-SunHiNeural", "manual_sample": True,
     "prompt": "오늘 GPU 사용률은 30 퍼센트 정도예요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "mixed_script_02", "bucket": "mixed_script", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "MiniCPM-o 모델을 한국어로 학습 중이에요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 6: proper_noun (3) — 고유명사, 지명 ────────────────────────────
    {"id": "proper_noun_01", "bucket": "proper_noun", "lang": "ko",
     "voice": "ko-KR-SunHiNeural", "manual_sample": True,
     "prompt": "강남역에서 두 시에 만나요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "proper_noun_02", "bucket": "proper_noun", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "청계천 산책 좋아하세요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "proper_noun_03", "bucket": "proper_noun", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "광화문 광장에 가본 적 있나요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 7: breath_group (2) — long single-breath sentences ────────────
    {"id": "breath_group_01", "bucket": "breath_group", "lang": "ko",
     "voice": "ko-KR-SunHiNeural", "manual_sample": True,
     "prompt": "오늘은 날씨가 정말 좋아서 친구들과 한강에서 자전거를 타고 강변을 따라 달렸어요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "breath_group_02", "bucket": "breath_group", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "한국어 음성 합성은 음소와 운율, 화자 특성이 모두 자연스럽게 어우러져야 좋은 결과를 만들 수 있어요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 8: short_answer (3) — single-token edge case ──────────────────
    {"id": "short_answer_01", "bucket": "short_answer", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "오늘 점심 같이 먹을래요?",
     "system": "한국어로 한 단어 또는 짧게 답하세요. 예: 네 / 아니요 / 좋아요."},
    {"id": "short_answer_02", "bucket": "short_answer", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "지금 시간 괜찮으세요?",
     "system": "한국어로 한 단어 또는 짧게 답하세요."},
    {"id": "short_answer_03", "bucket": "short_answer", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "도움이 필요하세요?",
     "system": "한국어로 한 단어 또는 짧게 답하세요."},

    # ── Bucket 9: baseline (8) — 인사/시간·날씨/지식/일상 ────────────────────
    {"id": "baseline_01", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "안녕하세요. 오늘 날씨가 어때요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_02", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "자기소개를 해주세요.",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_03", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "지금 몇 시예요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_04", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "한국에서 가장 높은 산은 무엇인가요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_05", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "주말에 보통 뭐 하세요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_06", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "한국어와 영어 중 어느 쪽이 더 어려운가요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_07", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-SunHiNeural",
     "prompt": "커피와 차 중에 무엇을 더 좋아하세요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},
    {"id": "baseline_08", "bucket": "baseline", "lang": "ko",
     "voice": "ko-KR-InJoonNeural",
     "prompt": "최근 본 영화 중 추천할 만한 게 있나요?",
     "system": "당신은 친절한 비서입니다. 한국어로 1-2문장으로 답하세요."},

    # ── Bucket 10: emotional (conditional, populate when 466 used) ────────────
    # — Phase 7-B trigger, intentionally empty in baseline run

    # ── Bucket 11: multi_speaker (conditional, populate when 542 used) ───────
    # — Phase 7-B trigger, intentionally empty in baseline run

    # ── Observability: 1 EN + 1 ZH (NOT optimization target) ─────────────────
    {"id": "obs_en_01", "bucket": "observability", "lang": "en",
     "voice": "en-US-AriaNeural",
     "prompt": "What is the capital of France?",
     "system": "You are a friendly assistant. Answer in 1-2 sentences."},
    {"id": "obs_zh_01", "bucket": "observability", "lang": "zh",
     "voice": "zh-CN-XiaoxiaoNeural",
     "prompt": "你好。今天天气怎么样？",
     "system": "你是一个友好的助手。用中文回答，1-2句话。"},
]

QUICK_BUCKETS = ["minimal_pair", "vowel", "baseline"]   # ~9 cases for per-epoch eval

# Cases marked manual_sample=True form the fixed stratified manual listening set
# (5 cases, 1 from each high-priority bucket — never random, for regression compare)


# ─────────────────────────────────────────────────────────────────────────────
# Char-class detection (Hangul / Hanja / Latin)
# ─────────────────────────────────────────────────────────────────────────────

_HANGUL_RE = re.compile(r'[가-힯ᄀ-ᇿ㄰-㆏]')
_HANJA_RE = re.compile(r'[一-鿿㐀-䶿]')
_LATIN_RE = re.compile(r'[a-zA-Z]')


def char_ratio(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"hangul": 0.0, "hanja": 0.0, "latin": 0.0, "other": 0.0, "total_chars": 0}
    h = len(_HANGUL_RE.findall(text))
    j = len(_HANJA_RE.findall(text))
    l = len(_LATIN_RE.findall(text))
    total = h + j + l
    if total == 0:
        return {"hangul": 0.0, "hanja": 0.0, "latin": 0.0, "other": 1.0, "total_chars": len(text)}
    return {"hangul": h / total, "hanja": j / total, "latin": l / total,
            "other": 0.0, "total_chars": len(text)}


# ─────────────────────────────────────────────────────────────────────────────
# CER — character error rate (primary metric)
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = cur[j] + 1
            sub = prev[j] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. 0.0 = perfect, 1.0+ = totally wrong.

    Whitespace collapsed (Korean often has space-segmentation ambiguity).
    """
    ref = re.sub(r'\s+', '', reference.strip())
    hyp = re.sub(r'\s+', '', hypothesis.strip())
    if not ref:
        return 1.0 if hyp else 0.0
    return _edit_distance(ref, hyp) / len(ref)


# ─────────────────────────────────────────────────────────────────────────────
# edge-tts → input.wav (16 kHz mono)
# ─────────────────────────────────────────────────────────────────────────────

async def synthesize_input_audio(text: str, voice: str, out_path: Path) -> np.ndarray:
    import edge_tts
    communicator = edge_tts.Communicate(text, voice, rate="+0%")
    mp3 = bytearray()
    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    proc = subprocess.run(
        ['ffmpeg', '-loglevel', 'error', '-i', '-', '-f', 'wav',
         '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', '-'],
        input=bytes(mp3), capture_output=True, check=True,
    )
    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype='float32')
    sf.write(str(out_path), audio, sr, subtype='PCM_16')
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# vllm-omni /v1/realtime — send audio, receive audio + server transcript
# ─────────────────────────────────────────────────────────────────────────────

async def query_realtime(ws_url: str, model: str, system_prompt: str,
                         input_audio: np.ndarray, out_path: Path,
                         timeout_s: float = 30.0) -> dict:
    import websockets
    base = ws_url.rstrip("/")
    uri = base if base.endswith("/v1/realtime") else f"{base}/v1/realtime"
    pcm16 = (np.clip(input_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    audio_b64 = base64.b64encode(pcm16).decode('ascii')

    response_pcm = bytearray()
    response_text = ""
    t0 = time.time()
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)   # session.created
            await ws.send(json.dumps({
                "type": "session.update", "model": model,
                "session": {"instructions": system_prompt,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16"},
            }))
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create",
                                       "response": {"modalities": ["audio", "text"]}}))
            while time.time() - t0 < timeout_s:
                try:
                    msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(msg_raw)
                t = msg.get("type", "")
                if t == "response.audio.delta":
                    response_pcm.extend(base64.b64decode(msg.get("delta", "")))
                elif t == "response.audio_transcript.delta":
                    response_text += msg.get("delta", "")
                elif t == "response.done":
                    break
                elif t == "error":
                    return {"error": msg.get("error", {}), "duration_s": time.time() - t0}
    except Exception as e:
        return {"error": str(e), "duration_s": time.time() - t0}

    elapsed = time.time() - t0
    if response_pcm:
        # vllm-omni realtime path emits 24 kHz PCM16 (per upstream
        # examples/online_serving/minicpm_o/realtime_voice_clone.py:153
        # "the omni pipeline emits 24 kHz PCM16"). Earlier code used 16 kHz
        # which played WAVs at 1.5× slow + lower pitch and distorted Whisper
        # transcription; verified 2026-05-01 against Phase 7-E outputs.
        audio = np.frombuffer(bytes(response_pcm), dtype=np.int16).astype(np.float32) / 32767.0
        sf.write(str(out_path), audio, 24000, subtype='PCM_16')
        duration = len(audio) / 24000.0
    else:
        duration = 0.0
    return {"duration_s": elapsed, "audio_seconds": duration,
            "transcript_from_server": response_text, "audio_bytes": len(response_pcm)}


# ─────────────────────────────────────────────────────────────────────────────
# Whisper ASR
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None


def whisper_transcribe(wav_path: Path, language: Optional[str] = None,
                        model_size: str = "base") -> str:
    global _whisper_model
    import whisper
    if _whisper_model is None:
        print(f'[whisper] loading model={model_size}...')
        _whisper_model = whisper.load_model(model_size)
    result = _whisper_model.transcribe(str(wav_path), language=language)
    return result.get("text", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main eval flow
# ─────────────────────────────────────────────────────────────────────────────

def select_cases(args) -> list:
    cases = TEST_CASES
    if args.bucket:
        wanted = set(args.bucket.split(','))
        cases = [c for c in cases if c['bucket'] in wanted]
    elif args.quick:
        cases = [c for c in cases if c['bucket'] in QUICK_BUCKETS]
    # Korean-only goal — observability(EN/ZH) cases는 기본 제외.
    # 사용자 directive (2026-05-03): "한국어 전용 모델이지 영/중/한 모델이 아니야"
    # 옛 동작 원하면 --include_observability 추가 (사실상 archive 용).
    if not args.include_observability:
        cases = [c for c in cases if c['bucket'] != 'observability']
    return cases


async def run_eval(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args)
    print(f'[eval] label={args.label} cases={len(cases)} output={out_dir}')

    results = []
    for tc in cases:
        case_id = tc['id']
        print(f'\n[case {case_id}] bucket={tc["bucket"]} prompt={tc["prompt"][:50]!r}')

        # 1. edge-tts input
        input_wav = out_dir / f'{case_id}_input.wav'
        try:
            audio = await synthesize_input_audio(tc['prompt'], tc['voice'], input_wav)
            print(f'  edge-tts → {input_wav.name} ({len(audio)/16000:.1f}s)')
        except Exception as e:
            print(f'  ❌ edge-tts: {e}')
            results.append({'case_id': case_id, 'bucket': tc['bucket'], 'lang': tc['lang'],
                            'error': f'edge-tts: {e}'})
            continue

        # 2. /v1/realtime
        response_wav = out_dir / f'{case_id}_response.wav'
        meta = await query_realtime(args.vllm_url, args.model, tc['system'],
                                     audio, response_wav, timeout_s=args.timeout)
        if 'error' in meta:
            print(f'  ❌ realtime: {meta["error"]}')
            results.append({'case_id': case_id, 'bucket': tc['bucket'], 'lang': tc['lang'],
                            'prompt': tc['prompt'], 'error': str(meta['error']),
                            'duration_s': meta.get('duration_s', 0)})
            continue
        print(f'  realtime → {response_wav.name} '
              f'({meta["audio_seconds"]:.1f}s audio, {meta["duration_s"]:.1f}s total)')

        # 3. whisper
        if response_wav.exists() and meta['audio_seconds'] > 0.1:
            try:
                whisper_lang = tc['lang'] if tc['lang'] in ('ko', 'en', 'zh') else 'ko'
                whisper_text = whisper_transcribe(response_wav, language=whisper_lang,
                                                   model_size=args.whisper_model)
            except Exception as e:
                whisper_text = f'<whisper error: {e}>'
        else:
            whisper_text = '<no audio>'
        print(f'  whisper: {whisper_text!r}')

        # 4. metrics
        ratios = char_ratio(whisper_text)
        server_text = meta.get('transcript_from_server', '').strip()
        case_cer = cer(server_text, whisper_text) if server_text else None
        # Per-language expected char class (informational)
        expected_class = {"en": "latin", "zh": "hanja", "ko": "hangul"}.get(tc['lang'], 'hangul')
        expected_ratio = ratios.get(expected_class, 0.0)
        print(f'  → {expected_class}={expected_ratio:.0%} '
              f'CER={case_cer if case_cer is not None else "N/A":.3f}'
              if case_cer is not None
              else f'  → {expected_class}={expected_ratio:.0%} CER=N/A (empty server transcript)')

        results.append({
            'case_id': case_id,
            'bucket': tc['bucket'],
            'lang': tc['lang'],
            'manual_sample': tc.get('manual_sample', False),
            'prompt': tc['prompt'],
            'system': tc['system'],
            'audio_seconds': meta['audio_seconds'],
            'duration_s': meta['duration_s'],
            'whisper_transcript': whisper_text,
            'server_transcript': server_text,
            'char_ratios': ratios,
            'expected_class': expected_class,
            'expected_ratio': expected_ratio,
            'cer': case_cer,
        })

    # ── Aggregate ───────────────────────────────────────────────────────────
    # Per-bucket pass rate (CER ≤ 0.30 OR observability lang≠ko skipped)
    by_bucket = {}
    for r in results:
        if 'error' in r:
            by_bucket.setdefault(r['bucket'], []).append({'pass': False, 'reason': 'error'})
            continue
        if r['bucket'] == 'observability':
            continue   # not part of pass criteria
        is_pass = (r['cer'] is not None and r['cer'] <= 0.30) and (r['expected_ratio'] >= 0.6)
        by_bucket.setdefault(r['bucket'], []).append({
            'pass': is_pass, 'cer': r['cer'], 'expected_ratio': r['expected_ratio'],
        })

    bucket_pass = {b: float(np.mean([1.0 if x['pass'] else 0.0 for x in v]))
                   for b, v in by_bucket.items()}

    ko_results = [r for r in results if 'error' not in r and r['lang'] == 'ko']
    ko_cer_vals = [r['cer'] for r in ko_results if r['cer'] is not None]
    ko_hangul_vals = [r['expected_ratio'] for r in ko_results]

    summary = {
        'label': args.label,
        'model': args.model,
        'vllm_url': args.vllm_url,
        'timestamp': time.time(),
        'n_cases': len(results),
        'mode': 'quick' if args.quick else ('bucket' if args.bucket else 'full'),
        'metrics': {
            'ko_cer_mean': float(np.mean(ko_cer_vals)) if ko_cer_vals else None,
            'ko_cer_median': float(np.median(ko_cer_vals)) if ko_cer_vals else None,
            'ko_hangul_ratio_mean': float(np.mean(ko_hangul_vals)) if ko_hangul_vals else None,
            'bucket_pass_rate': bucket_pass,
            'overall_ko_pass_rate': float(np.mean(
                [1.0 if (r['cer'] is not None and r['cer'] <= 0.30 and r['expected_ratio'] >= 0.6)
                 else 0.0 for r in ko_results]
            )) if ko_results else None,
        },
        'cases': results,
    }
    # Observability (EN/ZH) for trajectory
    obs_results = [r for r in results if 'error' not in r and r['bucket'] == 'observability']
    summary['observability'] = [
        {'lang': r['lang'], 'expected_ratio': r['expected_ratio'],
         'whisper': r['whisper_transcript'][:100]}
        for r in obs_results
    ]

    json_path = out_dir / 'summary.json'
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\n[eval] saved {json_path}')

    # ── Console report ─────────────────────────────────────────────────────
    m = summary['metrics']
    print(f'\n=== Summary ({args.label}) ===')
    if m['ko_cer_mean'] is not None:
        print(f'  Korean CER (mean / median): {m["ko_cer_mean"]:.3f} / {m["ko_cer_median"]:.3f}'
              f'  → {"PASS" if m["ko_cer_mean"] <= 0.30 else "FAIL"} (target ≤0.30)')
    if m['ko_hangul_ratio_mean'] is not None:
        print(f'  Korean hangul ratio (secondary): {m["ko_hangul_ratio_mean"]:.0%}')
    if m['overall_ko_pass_rate'] is not None:
        print(f'  Overall Korean case pass rate: {m["overall_ko_pass_rate"]:.0%}')
    print(f'\n  Per-bucket pass rate:')
    for b, rate in sorted(bucket_pass.items()):
        flag = '✅' if rate >= 0.6 else '⚠️'
        print(f'    {flag} {b:<16}  {rate:.0%}')
    if obs_results:
        print(f'\n  Observability (catastrophic-forgetting trajectory, NOT a pass criterion):')
        for r in obs_results:
            print(f'    {r["lang"]}: {r["expected_class"]}={r["expected_ratio"]:.0%} '
                  f'whisper={r["whisper_transcript"][:60]!r}')

    # ── report.md ──────────────────────────────────────────────────────────
    write_report(out_dir, summary)


def write_report(out_dir: Path, summary: dict):
    """Generate markdown report with per-case table + manual listening picks."""
    lines = [f'# Eval report — {summary["label"]}', '']
    m = summary['metrics']
    lines.append('## Metrics')
    lines.append('')
    if m['ko_cer_mean'] is not None:
        lines.append(f'- **Korean CER (primary)**: mean {m["ko_cer_mean"]:.3f}, median {m["ko_cer_median"]:.3f} (target ≤0.30)')
    if m['ko_hangul_ratio_mean'] is not None:
        lines.append(f'- Korean hangul ratio (secondary): {m["ko_hangul_ratio_mean"]:.1%}')
    if m['overall_ko_pass_rate'] is not None:
        lines.append(f'- Overall Korean pass rate: {m["overall_ko_pass_rate"]:.1%}')
    lines.append('')
    lines.append('| Bucket | Pass rate |')
    lines.append('|--------|----------:|')
    for b, rate in sorted(m['bucket_pass_rate'].items()):
        lines.append(f'| {b} | {rate:.0%} |')
    lines.append('')

    lines.append('## Manual listening sample (fixed stratified, 5/30)')
    lines.append('')
    manual = [r for r in summary['cases']
              if r.get('manual_sample') and 'error' not in r]
    for r in manual:
        wav = f'{r["case_id"]}_response.wav'
        lines.append(f'- **{r["case_id"]}** ({r["bucket"]}): `{wav}`')
        lines.append(f'  - prompt: {r["prompt"]!r}')
        lines.append(f'  - whisper: {r["whisper_transcript"]!r}')
        lines.append(f'  - server transcript: {r["server_transcript"]!r}')
        lines.append(f'  - CER: {r.get("cer", "N/A")}')
    lines.append('')

    lines.append('## Per-case detail')
    lines.append('')
    lines.append('| ID | Bucket | CER | Hangul | Whisper |')
    lines.append('|----|--------|----:|-------:|---------|')
    for r in summary['cases']:
        if 'error' in r:
            lines.append(f'| {r["case_id"]} | {r.get("bucket", "?")} | ERROR | - | {r.get("error", "")[:50]} |')
            continue
        c = f'{r["cer"]:.3f}' if r.get('cer') is not None else 'N/A'
        lines.append(f'| {r["case_id"]} | {r["bucket"]} | {c} | '
                      f'{r["expected_ratio"]:.0%} | {r["whisper_transcript"][:60]} |')
    lines.append('')

    if summary.get('observability'):
        lines.append('## Observability (EN/ZH catastrophic-forgetting trajectory)')
        lines.append('')
        for o in summary['observability']:
            lines.append(f'- **{o["lang"]}**: expected_ratio={o["expected_ratio"]:.0%}, whisper={o["whisper"]!r}')
        lines.append('')

    (out_dir / 'report.md').write_text('\n'.join(lines))
    print(f'[eval] saved {out_dir / "report.md"}')


def compare(dir_a: Path, dir_b: Path):
    a = json.loads((dir_a / 'summary.json').read_text())
    b = json.loads((dir_b / 'summary.json').read_text())
    ma, mb = a['metrics'], b['metrics']

    print(f'\n=== Compare {a["label"]} → {b["label"]} ===\n')
    if ma.get('ko_cer_mean') is not None and mb.get('ko_cer_mean') is not None:
        delta = mb['ko_cer_mean'] - ma['ko_cer_mean']
        arrow = '↓' if delta < -0.02 else ('↑' if delta > 0.02 else '·')
        print(f'  Korean CER:    {ma["ko_cer_mean"]:.3f} → {mb["ko_cer_mean"]:.3f}  ({delta:+.3f}) {arrow}'
              f'  (lower is better)')
    if ma.get('ko_hangul_ratio_mean') is not None and mb.get('ko_hangul_ratio_mean') is not None:
        delta = mb['ko_hangul_ratio_mean'] - ma['ko_hangul_ratio_mean']
        arrow = '↑' if delta > 0.05 else ('↓' if delta < -0.05 else '·')
        print(f'  Hangul ratio:  {ma["ko_hangul_ratio_mean"]:.0%} → {mb["ko_hangul_ratio_mean"]:.0%}  ({delta:+.0%}) {arrow}')

    if ma.get('overall_ko_pass_rate') is not None and mb.get('overall_ko_pass_rate') is not None:
        delta = mb['overall_ko_pass_rate'] - ma['overall_ko_pass_rate']
        arrow = '↑' if delta > 0.05 else ('↓' if delta < -0.05 else '·')
        print(f'  Pass rate:     {ma["overall_ko_pass_rate"]:.0%} → {mb["overall_ko_pass_rate"]:.0%}  ({delta:+.0%}) {arrow}')

    print(f'\n  Per-bucket pass rate:')
    bs = sorted(set(ma.get('bucket_pass_rate', {})) | set(mb.get('bucket_pass_rate', {})))
    for b_name in bs:
        va = ma.get('bucket_pass_rate', {}).get(b_name, 0.0)
        vb = mb.get('bucket_pass_rate', {}).get(b_name, 0.0)
        d = vb - va
        arrow = '↑' if d > 0.1 else ('↓' if d < -0.1 else '·')
        print(f'    {b_name:<16}  {va:.0%} → {vb:.0%}  ({d:+.0%}) {arrow}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', help='where to save audio + summary.json + report.md')
    p.add_argument('--label', default='run', help='run label for comparison')
    p.add_argument('--vllm_url', default='ws://localhost:8000')
    p.add_argument('--model', default='openbmb/MiniCPM-o-4_5')
    p.add_argument('--whisper_model', default='base', help='whisper size: tiny/base/small/medium/large-v3')
    p.add_argument('--bucket', help='comma-separated bucket names (filter); overrides --quick')
    p.add_argument('--quick', action='store_true',
                   help=f'9-case quick mode (buckets: {QUICK_BUCKETS}) for per-epoch eval')
    p.add_argument('--skip_observability', action='store_true',
                   help='[deprecated, default behavior now] EN/ZH cases 자동 제외 (한국어 전용)')
    p.add_argument('--include_observability', action='store_true',
                   help='Korean-only goal 외 EN/ZH 2 case 추가 (archive/legacy 검증용)')
    p.add_argument('--timeout', type=float, default=30.0)
    p.add_argument('--compare', nargs=2, metavar=('DIR_A', 'DIR_B'))
    args = p.parse_args()

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return
    if not args.output_dir:
        sys.exit('--output_dir required (unless --compare)')
    asyncio.run(run_eval(args))


if __name__ == '__main__':
    main()
