# 작업 기록

새 작업은 맨 위에 날짜 섹션으로 추가한다 (`coding_guidelines.md` 9절).

---

## 2026-10-26 — 저장소 연결 및 데이터 재현 (`01_merge`)

### 1. GitHub 저장소 연결
- 로컬 폴더가 git 저장소가 아닌 상태였다. 원격 `https://github.com/ysb00/R2R_tension_data_analysis`를 clone해 로컬과 대조한 결과 **추적 파일 전체의 blob hash가 동일**(차이는 줄바꿈 표기뿐)함을 확인하고, 로컬을 `origin/main`(`498e921`) 추적 사본으로 연결했다.
- `core.autocrlf=false` 설정 (로컬 파일은 LF. git이 체크아웃 시 CRLF로 다시 쓰지 않도록 함).

### 2. 실행 환경 차단 요인 해소
- `feature_analysis_plan.md` 4절의 차단 요인(`scipy` / `scikit-learn` / `matplotlib` 미설치)은 **이미 해소된 상태**였다: scipy 1.17.1, scikit-learn 1.9.0, matplotlib 3.11.0, numpy 2.4.4, pandas 3.0.3, openpyxl 3.1.5.
- `pywt`(PyWavelets)는 여전히 미설치 → I8의 wavelet은 계획대로 제외하고 STFT만 수행한다.

### 3. `code/01_merge.py` 복원·실행
`data_dictionary.md`가 참조하던 생성 스크립트와 출력물이 저장소에 없어, 원본 zip에서 다시 생성했다.

- 입력: `data/data_tensionSetting_{3..7}.zip` (zip 5개 × xlsx 42개 = 210 run)
- 출력: `outputs/01_merge/` — `merged_data.csv`(268,933 × 8), `01_file_summary.csv`(210 × 15), `01_sample_interval.csv`(210행), `hyperparameters.csv`
- 실행 시간 52.1 s (대부분 xlsx 210개 읽기)
- 출력 경로는 `coding_guidelines.md` 3절의 `outputs/NN_이름/` 규칙을 따랐다. 기존 문서의 `outputs/merged_data.csv` 표기는 `data_dictionary.md`에서 갱신했다.
- 설정값 3개는 파일명이 아니라 **원본 컬럼의 실제 값**에서 읽고, run 내 상수인지 검증했다 (210 run 전체 통과).

### 4. 재현 검증 결과
`data_dictionary.md`의 모든 기재 수치가 일치했다.

| 항목 | 문서 | 재현 |
|---|---|---|
| merged_data 형상 | 268,933 × 8 | 일치 |
| run 수 | 210 | 일치 |
| run별 샘플 수 `n` | 1275–1300 | 일치 |
| 라벨 0이 없는 run | 26개 | 일치 |
| 라벨 2가 없는 run | 6개 | 일치 |
| steady 진입 후 재불안정 run | `T3_R7_V200`, `T5_R6_V50`, `T6_R7_V50` | 일치 |

`cost_modification_ideas.md` 1절 관찰값도 대조했고, **2개 항목이 재현되지 않아 문서를 정정**했다.

1. **steady 구간 e 표준편차의 설정값 의존성.** 기존 기재 "설정 3 kg: 0.039 → 7 kg: 0.043"(단조 증가)은 재현되지 않았다. run별 중앙값은 3: 0.0381, 4: 0.0379, 5: 0.0397, 6: 0.0412, 7: 0.0403으로 **6 kg에서 최대이고 단조가 아니다.** 전체 중앙값 0.0396 kg은 문서의 0.040과 일치한다. → 설정값에 따른 변동 크기 증가 근거는 약하므로, 비율 잔차 e / 설정값의 필요성은 재검토 대상.
2. **"위쪽 overshoot이 뚜렷한 run 184개 중 43개".** 판정 기준이 문서에 없어 재현 불가. `e_max_unstable` 임계값별 run 수를 대신 기록했다 (> 0 kg 120개, > 0.05 kg 66개, > 0.1 kg 48개, > 0.2 kg 15개). 어떤 단일 임계값도 43개를 주지 않는다. → "뚜렷함"의 기준은 허용대역 δ와 함께 ablation(E1–E7)에서 확정한다.

또한 샘플 간격을 210 run 전체에서 재측정했다. **run별 평균·중앙값은 0.108–0.110 s로 매우 균일**하여 등간격 0.109 s 가정은 합당하나, **개별 샘플 간격은 0.021–0.387 s**까지 벌어진다 (기존 기재는 한 파일 기준 0.021–0.343 s).

### 5. 다음 단계
- `feature_analysis_plan.md` **3절 세부 구현 사항 13개 항목이 사용자 검토 대기 중**이다. `coding_guidelines.md` 6절에 따라 승인 전에는 `01_folds.py` 이후를 구현하지 않는다.
- 승인이 필요한 주요 항목: 예약 fold 번호, I6 위상 AUROC 처리, G_freq 대역 3등분, I8 STFT 길이 16 / hop 8, I12 증강 σ 값, 로지스틱 회귀 `class_weight='balanced'`.
