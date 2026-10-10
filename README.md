# 공개 앱 리뷰 기반 VOC 분석 자동화

공개된 앱 리뷰를 대상으로 반복적인 VOC 분류 업무를 자동화한 프로젝트다. 데이터 수집·정규화·변경 감지·명확한 조건 판정은 코드가 처리하고, 다중 의도처럼 문맥 판단이 필요한 리뷰만 GPT에 전달한다.

## 데이터

Mini-BAR가 공개한 Garmin Connect, Huawei Health, Samsung Health의 영어 리뷰 6,000건을 사용한다. 각 리뷰에는 다음 정답 라벨이 하나 이상 부여되어 있다.

- `problem_report`: 버그와 작동 문제
- `feature_request`: 기능 추가·변경 요청
- `irrelevant`: 두 요구사항 유형에 해당하지 않는 평가 또는 기타 의견

출처와 인용 정보는 [SOURCE.md](SOURCE.md)에 정리했다.

## 파이프라인

```text
원본 CSV 3개
  → 정규화 및 안정적인 review_id 생성
  → 7개 증분 배치 생성
  → SQLite 해시 비교로 신규·변경·중복 판정
  → 고신뢰 규칙으로 1차 분류
  → 모호한 리뷰만 GPT Structured Outputs로 분류
  → dry-run 검토
  → 승인된 결과 DB 반영
  → 정답 라벨과 비교해 정확도·시간·토큰 측정
```

## 특징

- 외부 패키지 없이 Python 표준 라이브러리만 사용
- API 응답을 JSON Schema로 고정
- 원문 해시가 같으면 재분석하지 않음
- 원문이 수정되면 같은 ID라도 재분석
- DB 반영 전 dry-run 결과 확인
- GPT 단독 방식과 하이브리드 방식 비교
- 입력·출력·총 토큰과 실제 실행 시간 기록
- 다중 라벨 Micro/Macro F1 및 Exact Match 계산

## 실행

Python 3.11 이상이 필요하다. 실제 모델 호출에는 사용할 제공자에 따라 `GEMINI_API_KEY` 또는 `OPENAI_API_KEY`를 설정한다.

```bash
cd outputs/voc_pipeline

# 1. 공개 데이터 다운로드
python3 voc_pipeline.py download

# 2. 데이터 정규화 및 7개 배치 생성
python3 voc_pipeline.py prepare

# 3. 네트워크 호출 없이 전체 파이프라인 점검
python3 voc_pipeline.py benchmark --provider oracle --sample-size 210

# 4. 실제 GPT 비교 실험
python3 voc_pipeline.py benchmark --provider openai --sample-size 210

# 무료 티어 Gemini 비교 실험
python3 voc_pipeline.py benchmark --provider gemini --sample-size 210

# 5. 신규 데이터 반영 전 미리보기
python3 voc_pipeline.py ingest --day 1 --dry-run --provider openai

# 6. 검증한 결과를 DB에 반영
python3 voc_pipeline.py ingest --day 1 --provider openai

# 7. 테스트
python3 -m unittest discover -s tests -v
```

`oracle` 제공자는 정답 라벨을 AI 결과처럼 반환하는 오프라인 검증용이다. 성과 수치에는 `openai` 또는 `gemini` 제공자로 실제 실행한 결과만 사용한다.

Gemini 기본 모델은 무료 티어가 제공되는 `gemini-3.5-flash-lite`다. Google AI Studio에서 키를 발급한 뒤 다음처럼 설정한다.

```bash
export GEMINI_API_KEY="발급받은_키"
```

또는 `.env.example`을 `.env`로 복사한 뒤 키를 입력한다. `.env`는 Git 제외 목록에 포함되어 있다.

```text
GEMINI_API_KEY=발급받은_키
```

무료 티어 요청에 포함된 콘텐츠는 Google 제품 개선에 사용될 수 있다. 이 프로젝트는 공개 데이터만 전송하며, 개인정보나 회사 내부 VOC를 무료 티어로 전송하지 않는다.

`--dry-run`은 DB 쓰기를 막지만 `--provider openai`로 지정하면 분류를 위한 실제 API 호출과 토큰 사용은 발생한다.

## 결과 파일

- `results/benchmark_*.json`: 방식별 시간·토큰·성능 비교
- `results/benchmark_predictions_*.csv`: 리뷰별 정답과 예측
- `results/dry_run_day_*.json`: DB 반영 전 변경 내역
- `results/voc.db`: 리뷰, 분석 결과, 실행 이력

## 성과 수치 사용 원칙

자기소개서에는 `provider=openai` 또는 `provider=gemini`로 실행된 결과 파일에 기록된 값만 사용한다. 정적 공개 데이터를 날짜별로 새로 수집한 것이 아니라, 공개 데이터 6,000건을 7개 증분 배치로 나누어 반복 처리했다고 설명한다. Gemini로 측정했다면 “GPT”가 아니라 “생성형 AI” 또는 “Gemini”라고 표기한다.

기존 채용공고 자동화 프로젝트에서 측정한 처리 시간 30% 단축과 토큰 55% 절감은 이 VOC 프로젝트의 성과로 옮겨 쓰지 않는다. VOC 프로젝트의 수치는 동일한 데이터와 모델로 GPT 단독 방식과 하이브리드 방식을 실행해 새로 산출한다.

전체 6,000건을 `gemini-3.5-flash-lite`로 비교한 결과, 역할 분리 방식은 처리 시간을 55.2% 단축하고 토큰을 51.6% 절감했다. AI 판단 대상은 55.5% 줄었으며 Micro F1은 89.81%에서 86.30%로 3.51%p 감소했다. 원본 결과는 `results/benchmark_20260929_172208_gemini.json`에 저장되어 있다.

규칙 개선 v2에서는 짧은 저평점 리뷰를 AI로 라우팅하고 간접 기능 요청 표현을 보강했다. 실제 6,000건 재실험 결과 AI 판단 대상은 48.82% 감소, 처리 시간은 47.98% 단축, 토큰은 45.99% 절감됐으며 역할 분리 Micro F1은 88.14%로 개선됐다. 근거 파일은 `results/benchmark_20261010_155156_gemini.json`이다.

현재 완료 범위와 실측 가능한 결과는 [PROJECT_STATUS.md](PROJECT_STATUS.md)에서 확인할 수 있다.
