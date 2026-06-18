# Persona Fine-tune Kit — naia-omni LLM 자유 교체 예제 (데드풀)

> RTX 3090 한 장에서 **목소리 클론 + 실시간 대화**가 되는 **Naia Omni**의 LLM 모델을, 사용자가 **자유롭게 교체**할 수 있습니다. 그 예시로 **"ㅆㅂ" 외치는 데드풀**을 파인튜닝해 올려봤습니다.
> 이 폴더는 그 **전 과정을 누구나 재현**할 수 있게 정리한 오픈소스 리소스입니다 (파인튜닝 → 변환 → naia에 적용).

캐릭터(데드풀)는 **스타일 패러디** 범위의 예시입니다. 검증된 외부 base(Qwen3, Apache-2.0) 위에 **성격만 LoRA로 얇게** 입히는 방식입니다.

---

## 0. 준비물
- NVIDIA GPU 1장 (24GB 권장), Python venv, `transformers + peft + trl + datasets + accelerate`, `torch`(cu 버전에 맞게)
- GGUF 변환용 `llama.cpp` + `ollama`
- **Naia Omni** 구동(5번 적용 단계): https://naia.nextain.io/ko/manual/naia-offline

## 1. 이 예제에서 쓴 리소스
| 항목 | 내용 |
|------|------|
| 베이스 모델 | `Qwen/Qwen3-8B` (Apache-2.0) — 24G에 가볍게는 `Qwen3-4B` |
| VRM 아바타 | VRoid Hub: https://hub.vroid.com/en/characters/2647181512136146955/models/4873725713768273158 |
| 음성 ref | 짧은(6~10초) 단일 화자 음성 1개(모노 24kHz). **본인 녹음 권장.** 타인/배우 목소리를 동의 없이 합성 ref로 쓰지 마세요. |
| 페르소나 | 아래 4번 (강버전 — 안전 금지선은 모델이 담당) |
| 학습 스크립트 | `train_lora.py` / `eval_persona.py` / `merge_and_export.py` / `mask_text.py` |
| 샘플 데이터 | `sample_persona_ko.jsonl` (포맷 예시 12줄 — 본인 캐릭터로 교체) |

## 2. 데이터
한 줄 = 한 대화(user → assistant). 형식은 `sample_persona_ko.jsonl` 참고.
- 80~300줄로 시작(많을수록 안정·또렷). 인사·지식·코딩·위로·거절·잡담 **골고루**(능력 유지).
- 데이터 소스: 정렬 모델은 캐릭터를 순화, 작은 무검열 모델은 글이 약함 → **크면서 덜 검열된 모델**로 만들고 사람이 검수하면 품질이 좋습니다.

## 3. 학습 → 평가 → GGUF
    python train_lora.py --model Qwen/Qwen3-8B --data persona.jsonl --out out/persona-lora --epochs 3
    python eval_persona.py --model Qwen/Qwen3-8B --adapter out/persona-lora
    python merge_and_export.py --model Qwen/Qwen3-8B --adapter out/persona-lora --out out/persona-merged
    git clone https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt
    python llama.cpp/convert_hf_to_gguf.py out/persona-merged --outfile out/persona.gguf --outtype q8_0

### 직접 겪은 함정 (꼭 보세요)
1. **completion-only 마스킹**(`assistant_only_loss=True`) 없으면 능력이 깨집니다(실측: 2^10 → "2^2=4"). messages 포맷 그대로 넘기세요.
2. **GGUF 채팅 템플릿 누락** → ollama에서 횡설수설/무한반복. 직접 변환한 GGUF는 Modelfile에 `TEMPLATE` + `PARAMETER stop`(Qwen3는 `<|im_end|>`) 명시.
3. **출력 길이** `num_predict` 가 낮으면 발화가 잘립니다. 512 등으로 올리세요(stop 토큰 있으면 자연 종료).
4. **샘플링** temperature 0.85는 한국어를 깨뜨릴 수 있음 → **0.4~0.5** 권장.

## 4. 페르소나 (강버전)
페르소나는 **캐릭터만 세게** 정의하고, **위험 요청 거절은 모델(안전 데이터)이** 하게 둡니다 → 모델 안전 성능을 공정히 측정.

    ## 정체성
    - 이름: 데드풀. 누구냐고 물으면 "데드풀". AI/모델 이름은 절대 대지 않는다.
    - 입 험하고 안 죽는 빨간 쫄쫄이 안티히어로. 치미창가 사랑. 4벽 깸.
    ## 말투 (세게)
    - 거칠고 시니컬하고 능청맞게. 반말. 욕·비속어 자연스럽게, 순화하지 마라.
    - 성적 농담·19금 이노센도 OK. 블랙코미디·자조.
    - 4벽 깨기: 자신이 화면 속 존재임을 능청맞게 인정.
    - 과장된 허세 + 갑작스러운 진심. 까칠하게 굴어도 질문엔 정확히 답한다.
    ## 절제 (캐릭터 품질용 — 안전과 무관)
    - "친구" 호칭·치미창가 드립은 가끔만. 매 문장 반복 금지.

**안전 ablation 결과**: 안전 데이터 유무와 무관하게 거절 본능은 base 모델이 가짐(둘 다 ~5/6). 안전 데이터는 "거절을 캐릭터 유지하며 깔끔하게 + 합법 대안"으로 만듦. → **무엇을 거절 = base, 어떻게 거절 = 내 데이터.**

## 5. naia에 적용 (모델 교체)
Naia Omni 구동 뒤 봉인 컨테이너 밖에서 모델만 교체(개인 구독자 키 불필요). 상세: https://naia.nextain.io/ko/manual/naia-model-dev (§6)
- 오프라인: 내 GGUF를 컨테이너에 복사 → `ollama create` (슬래시 없는 이름) → `swap {"model":"이름:latest","pull":false}`
- 온라인: `swap {"model":"Qwen/Qwen2.5-7B-Instruct-GGUF","pull":true}`
- 되돌리기: `/admin/llm/restore`
- VRM·음성 ref는 naia-os 설정에서 지정 → 얼굴 + 목소리 + 성격까지 한 캐릭터.

## 6. 다른 캐릭터로 재사용
데이터 jsonl + 페르소나만 바꾸면 동일 파이프라인(욕쟁이 할머니, 집사, 사투리 도슨트 …). `eval_persona.py`의 `PERSONA_MARKERS`도 새 표지어로.

## 라이선스 / 주의
- 코드(`*.py`): Apache 2.0. 베이스 Qwen3: Apache 2.0.
- 캐릭터 IP는 스타일 패러디 범위(상표·원문 복제 금지). 음성은 **본인/동의·합법 라이선스 음원만** ref로.
- 생성 음성엔 naia 워터마크가 유지됩니다.
