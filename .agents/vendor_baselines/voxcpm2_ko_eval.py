"""VoxCPM2 한국어 quality 비교 — ko-flow TEST_CASES 동일 텍스트 → CER.

공정 비교: Phase18 Talker 와 동일 ref 음색(ref_ko_485.wav) clone + 동일 한국어
텍스트 → VoxCPM2 합성 → faster-whisper KO transcribe → CER(입력텍스트, transcript).

Usage (dry-run 완료 후):
    .venv/bin/python vox_ko_eval.py \
        --ref /var/home/luke/.../assets/ref_audio/ref_ko_485.wav \
        --out vox_ko_eval_out
"""
from __future__ import annotations
import argparse, json, re, statistics, time
from pathlib import Path

KOFLOW_EVAL = Path("/var/home/luke/alpha-adk/projects/minicpm-o-4_5-ko-flow/tools/eval_omni_voice.py")


def load_testcases() -> list[dict]:
    """eval_omni_voice.py 의 TEST_CASES 를 numpy import 없이 추출 (텍스트만)."""
    src = KOFLOW_EVAL.read_text()
    m = re.search(r"TEST_CASES\s*=\s*\[", src)
    if not m:
        raise RuntimeError("TEST_CASES 못 찾음")
    # 균형 괄호로 list 끝 찾기
    i = src.index("[", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "[":
            depth += 1
        elif src[j] == "]":
            depth -= 1
            if depth == 0:
                break
        j += 1
    lst = eval(src[i:j + 1], {"True": True, "False": False, "None": None})
    return [c for c in lst if c.get("lang") == "ko"]


def cer(ref: str, hyp: str) -> float:
    ref = re.sub(r"\s+", "", ref.strip())
    hyp = re.sub(r"\s+", "", hyp.strip())
    if not ref:
        return 1.0 if hyp else 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for a in range(1, m + 1):
        prev = dp[0]
        dp[0] = a
        for b in range(1, n + 1):
            cur = dp[b]
            dp[b] = prev if ref[a - 1] == hyp[b - 1] else 1 + min(prev, dp[b - 1], dp[b])
            prev = cur
    return dp[n] / m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="ref voice wav (Phase18 동일 음색)")
    ap.add_argument("--out", default="vox_ko_eval_out")
    ap.add_argument("--whisper", default="large-v3-turbo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "wav").mkdir(exist_ok=True)

    cases = load_testcases()
    print(f"ko TEST_CASES: {len(cases)}")

    import numpy as np, soundfile as sf
    from voxcpm import VoxCPM
    t0 = time.time()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
    print(f"VoxCPM2 load {time.time()-t0:.1f}s")

    from faster_whisper import WhisperModel
    asr = WhisperModel(args.whisper, device="cuda", compute_type="float16", device_index=0)

    rows, cers = [], []
    for c in cases:
        cid, text = c["id"], c["prompt"]
        t1 = time.time()
        try:
            # 공식 app.py cloning kwargs (text= / reference_wav_path=). numeral/loanword
            # 정상화를 위해 normalize=True (VoxCPM2 내장 normalizer = 의도된 사용).
            wav = model.generate(
                text=text, reference_wav_path=args.ref,
                cfg_value=2.0, inference_timesteps=10, normalize=True,
            )
        except Exception as e:
            print(f"[{cid}] GEN FAIL {e}")
            rows.append({"id": cid, "text": text, "error": str(e)})
            continue
        wp = out / "wav" / f"{cid}.wav"
        sf.write(str(wp), wav, 48000)
        segs, info = asr.transcribe(str(wp), language="ko", beam_size=5)
        hyp = "".join(s.text for s in segs).strip()
        c_er = cer(text, hyp)
        cers.append(c_er)
        rows.append({"id": cid, "bucket": c.get("bucket"), "text": text,
                     "whisper": hyp, "cer": round(c_er, 4),
                     "gen_s": round(time.time() - t1, 1),
                     "audio_s": round(len(wav) / 48000, 1)})
        print(f"[{cid:18s}] CER={c_er:.3f} | {text[:30]} → {hyp[:30]}")

    summary = {
        "n": len(cers),
        "cer_median": round(statistics.median(cers), 4) if cers else None,
        "cer_mean": round(statistics.mean(cers), 4) if cers else None,
        "pass_rate_le_0.30": round(sum(1 for x in cers if x <= 0.30) / len(cers), 3) if cers else 0,
        "ref": args.ref,
        "compare_baseline": "Phase18 27.66h Talker: median CER 0.138, pass 83.3% (ko-flow)",
    }
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
