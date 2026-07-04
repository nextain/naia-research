# naia-research (한국어)

naia 한국어 음성 작업을 외부에 공개하는 창구입니다. 전체 안내는 [README.md](../../../README.md)를 보세요.

## 무엇을 하는 프로젝트인가

한국어 음성 연구의 대부분은 비공개 저장소(naia-labs)에 있습니다. 이 저장소는
그중에서 남이 직접 돌려볼 수 있는 부분만 공개합니다. 한국어 음성 파이프라인의
품질을 재는 도구, 그 도구에 넣는 문장, 그리고 재현 가능한 측정 결과 하나입니다.

여기서 말하는 음성 cascade는 대화형 비서 뒤에 있는 흐름입니다. 말이 텍스트가
되고(음성 인식, STT), 텍스트가 언어 모델(LLM)을 지나고, 답이 다시 말이
됩니다(음성 합성, TTS). 이 저장소는 그 흐름의 한국어 품질을 처음부터 끝까지
채점하는 일을 다룹니다.

목적은 검증 가능성입니다. 잘 만들어진 외부 음성 모델을 그대로 쓰면, 우리가
27.66시간 파인튜닝한 모델을 이길 수 있다고 주장합니다. 믿어 달라고 말하는 대신,
회의적인 사람이 직접 확인할 수 있도록 잣대를 함께 공개합니다.

## 지금 실제로 되는 것

- **평가 toolkit** (`.agents/toolkit/`) — `transcribe.py`(faster-whisper로 생성
  음성을 전사), `metrics.py`(문자 오류율, 빈 응답률, 잘림률, 비한국어 혼입,
  첫 음성 패킷까지의 지연), `report.py`(Markdown 보고서 생성. 생성된 음성이
  결과 옆에 있으면 링크를 걸어 줍니다).
- **한국어 평가 seed** (`.agents/seed_data/`) — 목적이 다른 문장 세트 둘과 기준
  음색 클립 하나. `ko_eval_seed.py`는 한국어 발음이 흔히 무너지는 지점을 모은
  스트레스 문장 30개(최소대립쌍·모음·수사·외래어·고유명사·긴 한 호흡 문장 등)이고,
  `ko_input_wavs/`는 일상 대화·박물관 도슨트·상식 세 상황의 일상 질문 30개를 미리
  음성으로 만들어 둔 것으로, 전체 cascade에 입력으로 넣습니다.
- **VoxCPM2 baseline** (`.agents/vendor_baselines/`) — 위 일상/박물관/상식 클립이
  아니라 `ko_eval_seed.py`의 스트레스 문장 30개를 학습 없이 합성해 문자 오류율로
  채점한 결과입니다. 중앙값 CER 0.000, 평균 0.107, 통과율(CER ≤ 0.30)
  86.7%(n=30)입니다. 참고로 27.66시간 파인튜닝한 Talker는 중앙값 CER 0.138,
  통과율 83.3%였는데, 이 파인튜닝은 비공개라 그 수치는 비교용으로 옮겨 적은 것일
  뿐 이 저장소에서 다시 돌릴 수는 없습니다. 평가 스크립트는 동봉한 seed에서 문장을
  읽어 오며, VoxCPM2·faster-whisper·CUDA GPU가 필요합니다.
- **LLM 교체 데모** (`model/fine-tuning/deadpool-lora-demo/`) — Qwen3에 캐릭터
  성격을 LoRA로 얇게 입혀 Naia Omni 런타임에 올리는 재현 키트. 다른 것을 다시
  학습하지 않고 음성 아바타의 "머리"만 바꿀 수 있습니다.

## 로드맵 (아직 없음)

방법론 설명 글, paradigm 에세이, 결합 가이드는 예정입니다. 실제 내용이 생기기
전까지 해당 폴더는 만들지 않습니다. metric 정의는 현재 `metrics.py`의 docstring에
들어 있습니다.

## 매 세션 mandatory reads

1. `.agents/context/agents-rules.json`
2. `.agents/context/project-index.yaml`

## 라이선스

코드: Apache 2.0. 문서와 평가 데이터: CC-BY-SA 4.0.

## Origin

[naia-labs](https://github.com/nextain/naia-labs) R&D ground에서 promote.
Production: [naia-agent](https://github.com/nextain/naia-agent),
[naia-model-infra](https://github.com/nextain/naia-model-infra).
