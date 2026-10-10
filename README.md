# 공개 앱 리뷰 VOC 분류 자동화

Garmin Connect, Huawei Health, Samsung Health의 영어 리뷰 6,000건을 대상으로 VOC 분류 과정을 자동화한 프로젝트입니다.

처음에는 모든 리뷰를 Gemini에 보내 분류했습니다. 하지만 `app crashes`처럼 규칙만으로도 분류할 수 있는 리뷰까지 모델에 보내면서 호출 시간과 토큰이 불필요하게 늘었습니다. 현재는 명확한 리뷰를 Python 규칙으로 먼저 처리하고, 문맥을 확인해야 하는 리뷰만 Gemini에 전달합니다.

## 데이터

Mini-BAR 공개 데이터셋을 사용했습니다.

- `problem_report`: 버그·오류·작동 실패
- `feature_request`: 기능 추가·변경 요청
- `irrelevant`: 위 두 유형에 해당하지 않는 평가나 의견

원본 출처와 커밋 정보는 [SOURCE.md](SOURCE.md)에 정리했습니다. 데이터는 실행 시 다운로드하며 원본 CSV와 API 키는 저장소에 포함하지 않습니다.

## 처리 흐름

```text
리뷰 CSV 다운로드
  → 텍스트 정규화·review_id 생성
  → 본문 해시 비교로 신규·변경·중복 확인
  → 규칙 기반 1차 분류
  → 애매한 리뷰만 Gemini 분류
  → JSON Schema 검증
  → dry-run으로 저장 내용 확인
  → SQLite 저장 및 정답 라벨과 비교
```

리뷰 본문이 바뀌지 않으면 기존 분석 결과를 다시 사용합니다. DB에 반영하기 전에는 dry-run 결과로 신규·변경 리뷰와 분류 결과를 확인할 수 있습니다.

## 실행 환경

- Python 3.11 이상
- 외부 패키지 없음
- 실제 AI 실행 시 `GEMINI_API_KEY` 필요

```bash
# 저장소 루트에서 실행
python3 voc_pipeline.py download
python3 voc_pipeline.py prepare

# API 없이 파이프라인과 라우팅 점검
python3 voc_pipeline.py benchmark --provider oracle --sample-size 210

# Gemini 파일럿
python3 voc_pipeline.py benchmark --provider gemini --sample-size 210

# Gemini 전체 실험
python3 voc_pipeline.py benchmark --provider gemini --sample-size 6000

# DB 반영 전 확인 및 반영
python3 voc_pipeline.py ingest --day 1 --dry-run --provider gemini
python3 voc_pipeline.py ingest --day 1 --provider gemini

# 테스트
python3 -m unittest discover -s tests -v
```

키는 `.env`에 `GEMINI_API_KEY=...` 형식으로 넣을 수 있습니다. `.env`, 원본 데이터, 처리 결과 DB와 CSV는 `.gitignore`에 등록되어 있습니다.

## 실험 결과

6,000건을 대상으로 Gemini 단독 방식과 역할 분리 방식을 같은 조건에서 비교했습니다.

| 지표 | AI 단독 | 규칙 v1 역할 분리 | 규칙 v2 역할 분리 |
|---|---:|---:|---:|
| AI 판단 리뷰 | 6,000건 | 2,670건 | 3,071건 |
| 처리 시간 | 1,530.1초 | 684.9초 | 654.0초 |
| 전체 토큰 | 956,030 | 463,174 | 515,702 |
| 처리 시간 단축 | - | 55.24% | 47.98% |
| 토큰 절감 | - | 51.55% | 45.99% |
| Micro F1 | 89.81% | 86.30% | 88.14% |

규칙 v1은 비용과 시간을 더 많이 줄였고, 규칙 v2는 AI로 보내는 리뷰를 늘려 품질 하락 폭을 줄였습니다. v2에서는 짧은 저평점 리뷰를 성급하게 `irrelevant`로 확정하지 않고 AI로 보내며, 간접적인 기능 요청 표현도 규칙에 추가했습니다.

최종 v2 결과는 `results/benchmark_20261010_155156_gemini.json`에 기록되어 있습니다. 실행 과정과 변경 이유는 [PROJECT_STATUS.md](PROJECT_STATUS.md), 전체 정리는 [FINAL_RESULTS.md](FINAL_RESULTS.md)에서 확인할 수 있습니다.

## 프로젝트 구조

```text
voc_pipeline/
├── voc_pipeline.py       # 다운로드·정규화·분류·평가·DB 반영
├── config.json           # 모델·배치·재시도 설정
├── tests/                # 자동 테스트 11개
├── SOURCE.md             # 데이터 출처
├── PROJECT_STATUS.md     # 진행 과정과 검증 기록
├── FINAL_RESULTS.md      # 최종 수치와 해석
└── results/              # 실행 시 생성, Git에는 결과 파일을 올리지 않음
```
