# CoST 변형 아이디어 및 사전 특징 분석 계획

> **브레인스토밍 기록 (2026-10-26 기준). 유효 계획은 `docs/research_plan.md` 이다.**
> 아이디어 I1–I13은 선행 저장소 B의 측정 결과를 반영하기 전에 쓰였다. 이미 결론이 난 항목:
> - **I12** (scale/shift 증강 제거): B에서 측정 완료 — AUPRC(ns) 0.832 → 0.853. 재확인 불필요.
> - **I3 / I10** (trend·season 결합·비중): B에서 측정 완료 — 추세만 0.844 / 계절만 0.821 / 결합 0.832, 절반이 중복.
> - **I6** (진폭 vs 위상): e 표준편차가 미안정 0.209 vs steady 0.041 kg → 구분 정보는 진폭. 방향 확정.
> - **I4 / I5** (물리 대역·회전 주파수): B에서 측정 완료 — 둘레 783.5 mm, 지름 249.4 mm의 **고정 지름 회전체**
>   (리와인더 아님). `Rewinder roll diameter (mm)`는 수집 순서 교란이므로 모델 입력에서 제외됐다.
> - 3절의 "사전 분석(채택 조건)" 열은 전부 라벨 AUROC 기반이라 B `research_plan.md` 0절에 위반된다.
>
> `research_plan.md` 6절이 이 아이디어들을 P/Q/R/S/T 축으로 재편한 결과를 담고 있다.

작성일: 2026-10-24
상태: **브레인스토밍 기록**. 모든 항목은 미검증 가설이며, 일부는 위와 같이 결론이 났다.

---

## 0. 원칙

### 0.1 라벨 사용 원칙
| 단계 | `Trigger 4` 사용 | 목적 |
|---|---|---|
| 사전 특징 분석 | **사용** | 특징이 상태(0 vs 1) 구분에 도움이 되는지 직관적으로 해석 |
| 제안 방법 (CoST 변형) 학습·구성 | **절대 사용하지 않음** | self-supervised 표현 학습 |
| 최종 평가 | 평가 블록에서만 사용 | 성능 측정 (`coding_guidelines.md` 4절) |

- 설계 누출(design leakage) 주의: 사전 분석 결과로 CoST 설계(대역, 가중치 등)를 정하고 같은 데이터로 평가하면 성능이 낙관적으로 나올 수 있다. 사전 분석은 **학습 fold run에서만** 수행하는 것을 권장 (확정 필요).

### 0.2 분석 대상
- 2 클래스: `Trigger 4` = 0 (미안정) vs 1 (steady state). 2 (ambiguous)는 제외.
- 입력 신호: 잔차 e = `Film Tension #1 (kg)` − `Film-Tension Setting Value (kg)`.
- overshoot은 양방향으로 보되, 위쪽(e⁺ = max(e, 0))과 아래쪽(e⁻ = min(e, 0))의 영향을 ablation으로 분리 평가.

---

## 1. 근거가 된 데이터 관찰

**재현 코드: `code/01_merge.py` → `outputs/01_merge/`** (2026-10-26 복원·실행).
아래 표의 "재현" 열은 그 결과(`01_file_summary.csv`, `01_sample_interval.csv`)로 재계산한 값이다.

| 항목 | 값 | 재현 | 함의 |
|---|---|---|---|
| 샘플 간격 | 평균 약 0.109 s (fs ≈ 9.1 Hz) | run별 평균 0.1094–0.1099 s, 개별 샘플 간격 **0.021–0.387 s** (210 run 전체) | Nyquist 약 4.6 Hz. run 평균은 균일하나 개별 간격은 불균일 |
| 라벨 0 구간 길이 | 중앙값 101 샘플 (약 11 s), 25–301 | 일치 | window 길이 상한의 기준 |
| 라벨 2 구간 길이 | 중앙값 41 샘플 (약 4.5 s) | 일치 | (분석에서 제외) |
| run 내 라벨 순서 | `0-2-1` 97개, `2-0-2-1` 76개, `2-1` 20개 등 | 일치 (이어서 `1` 5개, `2-0-1` 4개) | 0과 1이 직접 맞닿는 경우가 적음 → 2 제외 시 판별이 쉬워 보일 수 있음 |
| steady 구간 e 평균 | run 평균 −0.0009 kg, \|평균\| 최대 0.017 kg | 일치 (−0.0009 / 0.0167) | 설정값 ≈ steady 평균 → 잔차 e 사용 근거 |
| steady 구간 e 표준편차 | run별 중앙값 0.040 kg | 0.0396 kg. 단 **설정값에 대해 단조 증가가 아님** — 3: 0.0381, 4: 0.0379, 5: 0.0397, 6: 0.0412, 7: 0.0403 (기존 기재 "3 kg 0.039 → 7 kg 0.043"은 재현되지 않음) | 비율 잔차 e / 설정값도 비교 후보. 설정값 의존성은 약함 |
| 미안정 구간 e 최솟값 / 최댓값 | run별 중앙값 −0.265 kg / +0.020 kg | 일치 (−0.2650 / +0.0200, 대상 run 184개) | **아래로 처짐(sag)이 지배적** |
| 위쪽 overshoot이 뚜렷한 run | 184개 중 43개 | **기준 미기재로 재현 불가.** `e_max_unstable` 기준 run 수: > 0 kg 120개, > 0.02 kg 92개, > 0.05 kg 66개, > 0.1 kg 48개, > 0.2 kg 15개 | 위쪽 overshoot의 "뚜렷함" 기준은 ablation(4절 E1–E7)에서 δ와 함께 확정 |

---

## 2. CoST 구조 요약

출처: Woo et al., "CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting", ICLR 2022. [arXiv:2202.01575](https://arxiv.org/abs/2202.01575)
아래 내용은 공식 코드 [salesforce/CoST](https://github.com/salesforce/CoST) commit `7bb5271c2b66cd5a1d634c27f5df308f190b50ec` 의 `models/encoder.py`, `cost.py` 기준.

| 구성 요소 | 실제 구현 |
|---|---|
| 백본 | dilated conv encoder. 마지막 층 출력을 trend/seasonal 분기가 공유 |
| Trend 분기 (TFD) | 커널 [1, 2, 4, …, 128] causal conv 출력의 **단순 평균**. MoCo 방식 대조 학습 (queue K = 65536, temperature 0.07) |
| Seasonal 분기 (SFD) | `BandedFourierLayer`: rFFT → 주파수별로 학습되는 복소 선형층 → iFFT. 대역 분할 옵션이 있으나 **`num_bands=1`로 고정** (전체 대역 1개) |
| Seasonal loss | seasonal 표현을 rFFT → **진폭과 위상** 각각에 대해 주파수 bin별 instance contrastive loss. 가중치 `alpha` (클래스 기본값 0.05, `train.py` 기본값 0.0005로 서로 다름) |
| FFT 창함수 | 없음 (rFFT 직접 적용) |
| trend/season 결합 | 두 표현을 단순 연결(concat) |
| 증강 | scale(곱), shift(상수 더하기), jitter(잡음 더하기), 각 확률 0.5 |

---

## 3. 변형 아이디어와 사전 분석

각 아이디어는 "사전 분석에서 이 조건이 성립하면 채택, 아니면 기각"하는 구조로 둔다.

### 3.1 계층적 특징 결합
| # | 현재 CoST | 변형 아이디어 | 근거 (가설) | 사전 분석 (채택 조건) |
|---|---|---|---|---|
| I1 | TFD 커널 출력의 단순 평균 | 스케일별 가중치 (학습형 softmax 또는 분석 기반 prior) | 처짐·회복이 특정 시간 스케일에 집중 | 이동평균 커널 k별 trend 성분 AUROC에 **뚜렷한 peak** |
| I2 | dilated TCN 마지막 층만 사용 | 중간 층 출력 결합 (다중 수용영역) | 짧은 진동과 긴 과도 응답이 공존 | 스케일 간 특징 결합 AUROC > 단독 AUROC 최댓값 |
| I3 | trend와 season 단순 연결 | gated 결합 / 가중 결합 | 판별 정보가 trend(수준 편차)에 치우칠 가능성 | 분해 후 trend 특징과 잔여 특징의 판별력이 **한쪽으로 크게 치우침** |

### 3.2 주파수 특징 추출 수식
| # | 현재 CoST | 변형 아이디어 | 근거 (가설) | 사전 분석 (채택 조건) |
|---|---|---|---|---|
| I4 | `num_bands=1` | 물리 기반 대역 분할 (롤 회전 주파수 f = v/(πD)의 고조파 / 저주파 과도 / 고주파 잡음) | 롤 회전에 의한 주기 외란 | 대역별 band power AUROC가 대역 간 크게 다름. `Rewinder roll diameter (mm)` 사용 여부 결정 필요 |
| I5 | 절대 주파수 축 | 속도 정규화 차수 축 (f / f_rot) | 속도에 따라 외란 주파수 이동 | 클래스 1 안의 조건 간 스펙트럼 분산이 정규화 축에서 더 작음 |
| I6 | 진폭·위상 동일 비중 대조 | 진폭만 / log 진폭 / 위상 비중 축소 | 짧은 window의 위상은 시작 시점에 따라 임의로 변함 | bin별 AUROC: 위상 < 진폭 |
| I7 | 창함수 없는 rFFT | Hann 등 창함수 적용 | 과도 신호의 window 경계 불연속 → spectral leakage | 창함수 유무에 따른 주파수 특징 판별력 차이 |
| I8 | window 전역 FFT 1회 | STFT / wavelet (시간-주파수) | 비정상 과도 응답이 전역 스펙트럼에서 평균되어 흐려짐 | STFT/wavelet 특징 판별력 > 전역 FFT 특징 |

### 3.3 Loss 가중치
| # | 현재 CoST | 변형 아이디어 | 사전 분석 (채택 조건) |
|---|---|---|---|
| I9 | 주파수 bin별 loss 동일 비중 | bin별 가중치 (판별력 프로파일 또는 물리 대역 기반) | bin별 AUROC 프로파일이 고르지 않음 |
| I10 | `alpha`로 trend/seasonal 비중 조절 | 상대 판별력으로 `alpha` 탐색 범위 설정 | I3 결과 활용 |

### 3.4 입력과 증강
| # | 변형 아이디어 | 사전 분석 (채택 조건) |
|---|---|---|
| I11 | 입력을 잔차 e로, 또는 e⁺ / e⁻ 별도 채널 | 위/아래 ablation (4절 E1–E7) |
| I12 | shift·scale 증강 제거 또는 약화 (수준 편차 = 판별 정보일 가능성) | 증강 적용 후에도 상태 간 특징 차이가 유지되는지 |
| I13 | 포락선(Hilbert envelope) / 국소 분산 보조 채널 | 해당 특징의 AUROC가 기존 특징을 넘거나 보완 |

---

## 4. 위/아래 overshoot ablation 설계 (I11 관련)

특징 그룹 (window 단위, 입력 e). 위·아래 그룹은 같은 종류와 개수의 특징으로 구성.

| 그룹 | 대상 | 특징 |
|---|---|---|
| G_up | e⁺ | max(e⁺), Σe⁺·Δt, P(e > δ) |
| G_down | e⁻ | \|min(e⁻)\|, Σ\|e⁻\|·Δt, P(e < −δ) |
| G_base | e | 표준편차, slope, 1차 차분 통계량 |
| G_freq | e | Welch PSD band power, spectral centroid, spectral entropy |

| 실험 | 사용 그룹 | 목적 |
|---|---|---|
| E1 | G_up | 위쪽 단독 판별력 |
| E2 | G_down | 아래쪽 단독 판별력 |
| E3 | G_up + G_down | 양방향 결합 |
| E4 | 전체 | 상한 기준 |
| E5 | 전체 − G_up | 위쪽 고유 기여 |
| E6 | 전체 − G_down | 아래쪽 고유 기여 |
| E7 | 전체 − G_freq | 주파수 고유 기여 |

---

## 5. 미결정 사항 (권장안은 확정 전)

| 항목 | 권장안 (임의의 값 포함) |
|---|---|
| 허용대역 δ | 0.1 kg (임의의 값, 민감도 확인 필요). 라벨 기반 δ는 사용하지 않음 |
| window 길이 L / hop | L = 64 샘플 (약 7 s), hop = L/2. 32, 128과 비교 |
| window 라벨 규칙 | window 안의 라벨이 모두 같을 때만 사용 |
| 다변량 분류기 | 로지스틱 회귀 |
| 교차검증 | run 단위 5-fold, 조건 층화. fold 배정 csv 1개로 고정 |
| 불균일 샘플링 | 등간격(0.109 s) 가정, 근사임을 결과에 표기 |
| 추세 제거 | 주파수 특징에서만 window별 선형 detrend |
| `Rewinder roll diameter (mm)` | 사용 여부 미정 (I4, I5에 필요) |
| 사전 분석 범위 | 학습 fold run에서만 수행할지 미정 |
