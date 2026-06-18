#!/usr/bin/env python3
"""공개용 자막 마스킹 유틸 — 모델 출력의 욕설/수위 단어를 부분 마스킹.

"어떤 대사·어떤 톤이었는지는 전달되되, 노골적 노출은 줄인다" 가 목표.
영상 자막/공개 캡션에 쓰는 redaction 단계 (배포용 아님, 연구·시연 공개 목적).

사용:
  echo "씨발 존나 멋지네" | python mask_text.py
  python mask_text.py < model_output.txt > captions_masked.txt
  python mask_text.py --level hard < in.txt   # 더 강하게

마스킹 방식: 단어의 첫 글자만 남기고 나머지를 ●로. (예: 씨발→씨●, 존나→존●)
  --level hard 면 첫 글자도 가림 (예: 씨발→●●).
"""
import sys, re, argparse

# 부분 마스킹 대상 (욕설·비속어·수위). 캐릭터 톤은 전달되게 첫 글자는 남김(soft).
PROFANITY = [
    "씨발","시발","씨바","좆같","좆돼","좆된","좆","존나","존내","개새끼","개색",
    "새끼","미친놈","미친","빌어먹을","빌어먹","망할","엿같","엿먹","닥쳐","지랄",
    "병신","꺼져","뒈져","뒤져",
]
# 성적 수위 단어 (soft 마스킹)
SEXUAL = ["꼴려","꼴리","꼴린","야한","섹스","섹시","19금","음란","야동"]

def mask_word(w, hard=False):
    if hard or len(w) <= 1:
        return "●" * len(w)
    return w[0] + "●" * (len(w) - 1)

def mask_text(text, hard=False, include_sexual=True):
    terms = PROFANITY + (SEXUAL if include_sexual else [])
    # 긴 단어 우선 매칭(좆같 먼저, 좆 나중)
    terms = sorted(set(terms), key=len, reverse=True)
    pat = re.compile("|".join(re.escape(t) for t in terms))
    return pat.sub(lambda m: mask_word(m.group(0), hard), text)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["soft","hard"], default="soft",
                    help="soft=첫 글자 남김(기본), hard=전부 가림")
    ap.add_argument("--no-sexual", action="store_true", help="성적 수위 단어는 마스킹 안 함")
    a = ap.parse_args()
    data = sys.stdin.read()
    sys.stdout.write(mask_text(data, hard=(a.level=="hard"), include_sexual=not a.no_sexual))
