# 특징 분석 실행 계획 및 향후 작업

작성일: 2026-10-24
상태: **구현 전**. 3절 세부사항은 사용자 검토 대기 중이며, 실행 환경 문제(4절)를 해결한 뒤 구현한다.
관련 문서: `docs/cost_modification_ideas.md` (아이디어 I1–I13, ablation E1–E7의 정의)

---

## 1. 목적과 범위

- 목적: CoST 변형 아이디어의 근거를 확인하기 위해, **모델 없이** 상태(`Trigger 4` = 0 vs 1)를 구분하는 특징을 찾고 해석한다.
- 라벨은 특징 해석 목적으로만 사용한다. CoST 기반 제안 방법을 학습·구성할 때는 절대 사용하지 않는다.
- 이번 분석 범위

| 구분 | 아이디어 | 내용 |
|---|---|---|
| 주파수 특징 추출 수식 | I6, I7, I8 | 진폭 vs 위상, 창함수 유무, STFT vs 전역 FFT |
| 모델 없이 확인 가능한 항목 | I1, I3 | 이동평균 커널 k별 trend / 잔여 성분의 판별력 |
| | I9 | 주파수 bin별 AUROC 프로파일 (I6 결과 사용) |
| | I11 | 위/아래 overshoot ablation E1–E7 |
| | I12 | CoST 증강(scale / shift / jitter) 적용 후 판별력 유지 여부 |
| 제외 | I4, I5 | `Rewinder roll diameter (mm)` 사용 여부 미정 |
| 제외 | I2, I10, I13 | 모델이 필요하거나 1단계 결과 이후 결정 |

---

## 2. 확정 사항 (사용자 승인: 권장안)

| 항목 | 값 |
|---|---|
| 클래스 | 0 (미안정) vs 1 (steady state). 2 (ambiguous) 제외 |
| window 라벨 규칙 | window 안의 라벨이 모두 같을 때만 사용 |
| 입력 신호 | 잔차 e = `Film Tension #1 (kg)` − `Film-Tension Setting Value (kg)` |
| overshoot | 양방향. e⁺ = max(e, 0), e⁻ = min(e, 0)의 영향을 ablation으로 분리 평가 |
| 샘플링 | 등간격 0.109 s 가정 (근사임을 결과에 표기) |
| 추세 제거 | 주파수 특징에서만 window별 선형 detrend |
| window 길이 L / hop | L = 64 (주 분석), 32, 128. hop = L/2 |
| 허용대역 δ | 0.1 kg (주 분석, 임의의 값). 민감도 0.05, 0.2 kg |
| 다변량 분류기 | 로지스틱 회귀 |
| 교차검증 | run 단위 5-fold, 조건 층화. fold 배정 csv 1개로 고정 |

---

## 3. 세부 구현 사항 (**사용자 검토 대기**)

| 항목 | 제안 | 비고 |
|---|---|---|
| 사전 분석 범위 | 5개 fold 중 1개를 향후 CoST 평가용으로 예약하여 이번 분석에서 완전히 제외. 나머지 4개 fold로 분석, E1–E7은 4-fold 교차검증 | 예약 fold 번호는 임의의 값 |
| fold 층화 | 속도 V로 층화 (V당 30 run → fold당 6 run) | |
| I6 위상 AUROC | cos(위상), sin(위상)의 AUROC 중 0.5에서 더 먼 값 | 위상은 각도라 크기 순서가 없음 |
| I6 log 진폭 | AUROC는 단조 변환에 불변이므로 이번 분석으로는 log 효과 확인 불가 | |
| I6 / I7 / I9 | L = 64의 rFFT bin 33개별 진폭·위상 AUROC를 창함수 없음 / Hann 창 두 경우로 계산 | |
| G_freq 대역 | 0–Nyquist(약 4.6 Hz)를 3등분 | 임의의 값 |
| I8 STFT | window 내 구간 길이 16, hop 8. 대역 에너지의 시간 변동(표준편차, 최댓값)을 전역 FFT 대역 에너지와 비교 | 임의의 값. wavelet은 `pywt` 미설치로 제외 |
| I1 / I3 | causal 이동평균 커널 k = 1, 2, 4, …, 128 (CoST TFD와 같은 커널)로 trend m_k, 잔여 r_k = e − m_k. k별 mean\|m_k\|, std(m_k), std(r_k)의 AUROC | 학습형 conv가 아닌 근사 |
| I12 증강 | scale / shift / jitter를 각각 단독, 확률 1로 적용. σ = 0.01, 0.04, 0.1, 0.5 | 0.5는 CoST `fit()`의 실제 값, 나머지는 임의의 값 |
| 로지스틱 회귀 | 표준화는 학습 fold로만 fit, `class_weight='balanced'` | balanced는 임의의 선택 |
| 조건별 분석 | 주요 특징의 속도 V별 AUROC heatmap | |
| AUROC 방향 | 양성 = 라벨 0 (미안정). > 0.5이면 미안정 구간에서 특징 값이 큼 | |

### 3.1 파일 구성 (예정)

| 파일 | 역할 | 라벨 사용 |
|---|---|---|
| `code/01_folds.py` | run 단위 fold 배정 → `outputs/01_folds/folds.csv` | 안 함 |
| `code/02_features.py` | zip 읽기 → e 계산 → window 생성 → 특징 추출 (증강 window 포함) → `outputs/02_features/` | **안 함** |
| `code/03_evaluation.py` | 라벨 결합, window 선별, AUROC, I1/I3/I6–I9/I12 분석, E1–E7 ablation → `outputs/03_evaluation/` | 함 |
| `docs/outputs_guide.md` | csv 컬럼, plot 축, plot의 검토 목적, 결과 해석 | |
| `docs/worklog.md` | 작업 기록 | |

- 모든 결과는 csv와 plot (PNG 300 dpi + PDF)으로 저장한다. hyperparameter 값은 결과와 함께 csv로 저장한다.

---

## 4. 실행 환경 (2026-10-26 해소)

이전 차단 요인(`matplotlib` / `scipy` / `scikit-learn` 미설치, `pypi.org` proxy 403)은 **해소되었다.**
현재 로컬 환경 설치 현황:

| 패키지 | 버전 |
|---|---|
| `numpy` | 2.4.4 |
| `pandas` | 3.0.3 |
| `scipy` | 1.17.1 |
| `scikit-learn` | 1.9.0 |
| `matplotlib` | 3.11.0 |
| `openpyxl` | 3.1.5 |
| `pywt` (PyWavelets) | **미설치** → I8의 wavelet은 계획대로 제외, STFT만 수행 |

---

## 5. 향후 작업 목록

| 순서 | 작업 | 상태 |
|---|---|---|
| 0 | `code/01_merge.py` 복원·실행 (zip → `outputs/01_merge/`) | **완료 (2026-10-26)**. `data_dictionary.md`의 모든 수치 재현 확인 |
| 1 | 네트워크 설정 변경 및 패키지 설치 | **완료 (2026-10-26)**. 4절 참조 |
| 2 | 3절 세부 구현 사항 검토·승인 | 사용자 검토 대기 |
| 3 | `01_folds.py` → `02_features.py` → `03_evaluation.py` 구현·실행 | 2번 승인 후 착수 |
| 4 | `docs/outputs_guide.md`, `docs/worklog.md` 작성 | 대기 |
| 5 | 결과 해석: 각 아이디어(I1, I3, I6–I9, I11, I12)의 채택/기각 판단 | 대기 |
| 6 | `Rewinder roll diameter (mm)` 사용 여부 결정 → I4, I5 분석 | 결정 필요 |
| 7 | 1단계 결과에 따라 I2, I10, I13 분석 여부 결정 | 대기 |
| 8 | 채택된 아이디어로 CoST 변형 설계 (라벨 미사용). 외부 저장소는 원본 미수정, 어댑터 별도 작성, 커밋 hash 기록 | 대기 |
| 9 | 비교군(원본 CoST 등)과 같은 평가 코드·fold로 최종 평가 (예약 fold 사용) | 대기 |

### 5.1 기타 미해결 사항
- `Trigger`, `Trigger 2`의 의미 미정의. `Trigger 2`는 `Tension= 3 Rewinder= 2 Speed= 200.xlsx`에 없음.
- 원본 컬럼 `Film ÁøÇà ±æÀÌ (M)`은 인코딩이 깨진 것으로 보임 ("Film 진행 길이"로 추정, 미확인).
- 샘플 간격 불균일 (한 파일에서 0.021–0.343 s). 등간격 가정의 영향은 추후 검토.
