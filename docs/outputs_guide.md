# 산출물 안내 — csv 컬럼과 plot 해석

- 작성일: 2026-10-28
- 대상: `docs/research_plan.md` 의 feature 사전 분석 실행 결과 (`outputs/02_features` ~ `outputs/05_visualize`)
- 평가 후단과 fold 배정은 B 저장소(`R2R_steadyState_identification_for_changing_condition`,
  커밋 `caaac99`)에서 **수정 없이 복사**했다. 기록은 `code/external_versions.csv`.

---

## 0. 폴더 구조 — **제안 4계열(P1~P4) 기준으로 나눈다**

단계마다 `outputs/NN_이름/` 아래에 계열 폴더 6개 + `_cross/` 를 둔다. 전 단계가 같은 이름을 쓴다.

| 폴더 | feature | 비고 |
|---|---|---|
| **`P1_spectral/`** | `spec_centroid`, `spec_rolloff85`, `spec_bandwidth` | 제안 1 — 스펙트럼 |
| **`P2_phase/`** | `phase_vel_std` | 제안 2 — 위상. **PLV 는 계획 3.1절에서 기각** |
| **`P3_nonstationarity/`** | `movkurt`, `movskew`, `hurst` | 제안 3 — 비정상성 |
| **`P4_physical/`** | `env_mean`, `env_std` | 제안 4 — 물리기반. **Tension-Speed 상관은 기각** |
| `P0_baseline/` | `movstd`, `movslope` | 제안 아님 — B 통계 baseline 승계 기준점. 모든 비교의 출발점 |
| `P5_additional/` | `movmean_abs_e`, `movrange`, `movmad`, `detrend_resid_std`, `half_mean_diff` | 제안 아님 — 계획 3.2절에서 추가한 진폭·추세 계열 |
| `_cross/` | 계열에 속하지 않는 공통·통합 산출물 | 전진 선택, 창 민감도, A-5, 상관행렬, `*_all.csv` |

계열 폴더의 csv 는 그 계열 행만 담고, `_cross/` 에 `*_all.csv` 통합본이 함께 있다.
**그림의 색도 이 구분을 따른다** — `P0` 검정 / 제안 4계열 유채색 / `P5` 회색이라, 산점도에서
제안한 feature 가 먼저 눈에 들어온다. 계열 폴더에는 그 계열만 강조하고 나머지를 회색으로 깐
산점도가 따로 저장된다.

기각된 2종(PLV, Tension-Speed 상관)은 feature 가 없으므로 `P2_phase/`·`P4_physical/` 에
해당 파일이 들어가지 않는다. 기각 근거는 `docs/research_plan.md` 3.1절이다.

---

## 0.1 실행 순서

```
python code/02_features.py      # 시점별 feature 산출 (라벨 미사용)        -> outputs/02_features/
python code/03_screening.py     # D·T, 승계 검증, 증분, 전진 선택, A-5     -> outputs/03_screening/
python code/04_overlap.py       # A-4 상태 vs 조건                         -> outputs/04_overlap/
python code/05_visualize.py     # A-6 설명용 그림                          -> outputs/05_visualize/
```

`03_screening.py` 가 약 22분, 나머지는 1분 이내다. **`04_overlap.py` 와 `05_visualize.py` 는
`03_screening.py` 가 만든 `_cross/D_summary_all.csv` 를 읽으므로 순서를 지켜야 한다.**

### 0.2 두 지표의 정의 (모든 표에 공통)

| 기호 | 정의 | 묻는 것 |
|---|---|---|
| **D** | **조건 내**. run 안에서 `Trigger 4` 0 vs 1 의 AUROC (ambiguous 제외) 를 run 평균. 방향 보정(`max(a, 1-a)`) | 이 feature 에 정보가 있는가 |
| **T** | **조건 간**. 학습 fold 에서 표준화+2군집 GMM 적합 → 검증 fold 의 **AUPRC(non-steady)** | 그 정보를 **고정 경계**로 쓸 수 있는가 |

T 는 fold 묶음 2개로 따로 낸다. `tension` = 미학습 장력 수준 5-fold, `final_test` = 저속(50–125)
학습 → 고속(150–200) 테스트. **두 묶음은 다른 질문이므로 평균내지 않는다.**

---

## 1. `outputs/02_features/` — 시점별 feature

라벨(`Trigger 4`)을 읽지 않는 단계다. feature csv 에도 라벨이 없다.

### 1.1 `{계열}/features_W{64,144,256}.csv.gz`

시점 1개 = 행 1개 (268,933행). gzip 이고 `pd.read_csv` 가 확장자로 바로 읽는다.

| 컬럼 | 단위 | 설명 |
|---|---|---|
| `run_id`, `k` | – | run 식별자와 run 내 시점 index. `merged_data.csv` 와 1:1 |
| `movstd` | kg | 이동 표준편차 (ddof=1). **B `rolling_std_slope` 를 그대로 복사** |
| `movslope` | kg/s | 이동 최소제곱 1차 기울기. 부호 보존 (평가에서는 크기만 쓴다) |
| `movmean_abs_e` | kg | `\|e\|` 의 창 평균 |
| `movrange` | kg | 창 최댓값 − 최솟값 |
| `movmad` | kg | 창 평균에 대한 **평균절대편차** `mean(\|x − mean(x)\|)`. 중위절대편차가 아니다 — 신호가 0.010 kg 로 양자화돼 있어 중위 기반은 값이 몇 단계로만 변한다 |
| `movkurt` | – | 창 첨도. **excess 규약** (정규 = 0, `scipy.stats.kurtosis` 기본) |
| `movskew` | – | 창 왜도 |
| `spec_centroid` | Hz | **창 평균 제거 + Hann** 후 rfft 진폭스펙트럼의 1차 모멘트 |
| `spec_rolloff85` | Hz | 누적 진폭이 전체의 85 % 가 되는 최저 주파수 |
| `spec_bandwidth` | Hz | centroid 기준 2차 모멘트 |
| `detrend_resid_std` | kg | 창 내 1차 추세 제거 후 잔차 표준편차 (분모 `W−1`) |
| `half_mean_diff` | kg | 창 후반 평균 − 전반 평균 |
| `env_mean`, `env_std` | kg | 창 평균 제거 후 Hilbert 포락선의 창 평균·표준편차 |
| `phase_vel_std` | rad/sample | `std(diff(unwrap(angle(hilbert(x − mean)))))`. **`unwrap` 이 들어간다** — 빼면 2π 점프가 섞여 값이 5배 이상 커진다 |
| `hurst` | – | R/S 기반 Hurst 지수. 블록 척도 `W, W/2, …, ≥8` 의 log-log 회귀 기울기 |

**창 미완성 구간**: 앞 `W−1` 시점은 버리지 않고 **첫 완전창의 값으로 backfill** 한다 (B 승계).
그래서 창 길이가 달라도 채점 대상 시점 수가 268,933 로 같다. 대신 `k ≤ W−2` 인 시점이 전부
같은 값을 받으므로 **과도 구간 내부에서 feature 가 상수**가 된다 (3.3절 한계 L2).

### 1.2 나머지 csv

| 파일 | 내용 |
|---|---|
| `{계열}/feature_summary.csv` | 창별·feature별 기술통계 (`mean`, `std`, `min`, `q05`, `median`, `q95`, `max`, `n_unique`). `_cross/feature_summary_all.csv` 가 통합본 |
| `_cross/feature_scaling.csv` | feature 정의표. **`amplitude_scale_exponent`** 가 핵심 — `e → c·e` (c 는 run 내 상수) 일 때 feature 가 `c^지수` 배가 된다. 1 = 진폭 비례(std 계열), 0 = 진폭 불변(형상·주파수·위상·R/S). A-5 는 이 지수로 환산하므로 feature 를 다시 산출하지 않는다 |
| `_cross/run_scales.csv` | run별 진폭 정규화 분모. `mad_run` = run 전체 e 의 **중위절대편차** (0.0250–0.0500 kg) |
| `P3_nonstationarity/hurst_fit_quality.csv` | F7 추정 안정성. log-log 결정계수 run 평균이 창 64 에서 0.986, 144 에서 0.974, 256 에서 0.971 이고 5분위도 0.93 이상이다 — R/S 적합 자체는 안정적이다 |
| `_cross/hyperparameters.csv` | 실행에 쓴 설정값 전체 |

---

## 2. `outputs/03_screening/` — feature 선별

### 2.1 `P0_baseline/inheritance_check.csv` — **가장 먼저 볼 표**

A 의 F0(`movstd`, `|movslope|`) 수치가 B 의 통계 baseline 과 같은지 확인한다. 맞지 않으면
데이터 처리가 B 와 다르다는 뜻이므로 이후 수치를 믿을 수 없다.

| W | 묶음 | A | B | 차이 | B 쪽 방법 |
|---|---|---|---|---|---|
| 64 | tension | 0.718073 | 0.7181 | −2.7e−5 | `MovStd-Slope-fixed` |
| 64 | final_test | 0.846694 | 0.8467 | −6e−6 | `MovStd-Slope-fixed` |
| 144 | tension | 0.608091 | 0.6081 | −9e−6 | `MovStd-Slope-fixed-rule1b` |
| 144 | final_test | 0.541445 | 0.5414 | +4.5e−5 | `MovStd-Slope-fixed-rule1b` |

차이는 전부 **B 쪽 참조값을 소수 4자리로 적은 반올림 오차**다. 승계 검증 통과.

### 2.2 `{계열}/D_summary.csv` / `{계열}/state_mean_diff.csv`

| 컬럼 | 설명 |
|---|---|
| `D_raw`, `D_raw_std` | 방향 보정 전 run 평균 AUROC 와 run 간 표준편차 |
| `D` | 방향 보정 AUROC (`max(a, 1−a)` 의 run 평균). 산점도의 가로축 |
| `mean_transient`, `mean_steady`, `diff`, `sign` | 상태별 feature 평균과 그 차이. **계획 3.1절의 가설 방향을 전 구간에서 재확인하는 표다** |

### 2.3 `{계열}/DT_single.csv` — 단일 feature 의 D·T (창 144)

창 144 에서 feature 1개만 후단에 넣었을 때의 값이다. 주요 행:

| feature | D | T(tension) | T(final_test) |
|---|---|---|---|
| `half_mean_diff` | 0.934 | 0.669 | **0.904** |
| `phase_vel_std` | 0.941 | **0.718** | 0.867 |
| `spec_centroid` | 0.945 | 0.674 | 0.839 |
| `movslope` | 0.931 | 0.588 | 0.802 |
| `hurst` | 0.943 | 0.637 | 0.787 |
| `movstd` | 0.975 | 0.632 | 0.570 |
| `movmean_abs_e` | **0.995** | 0.677 | 0.565 |
| `movkurt` | 0.892 | 0.208 | 0.211 |

**읽는 법**: D 는 거의 모든 feature 가 0.84 이상이다 — 조건 안에서는 다 잘 가른다. 갈리는 것은
T 다. 특히 **D 가 가장 높은 `movmean_abs_e` 가 고속 외삽에서는 0.565 로 바닥권**이고, D 가 중간인
`half_mean_diff`·`phase_vel_std` 가 0.90·0.87 로 앞선다. **D 만 보고 고르면 안 된다**는 것이
이 표의 결론이다.

### 2.4 `{계열}/increment.csv` / `_cross/forward_selection.csv`

`{계열}/increment.csv` 는 F0 2차원에 후보를 **1개씩** 더했을 때의 ΔT 다 (창 144, 3차원).
ΔT(tension) > 0 인 feature 는 14개 중 6개: `hurst` +0.0324, `movskew` +0.0141,
`spec_centroid` +0.0079, `spec_rolloff85` +0.0069, `spec_bandwidth` +0.0056, `movkurt` +0.0048.
진폭 계열(`movmean_abs_e`, `movmad`, `env_mean`, `detrend_resid_std`)은 전부 **악화**시킨다 —
`movstd` 와 상관이 높아 중복이고, diag GMM 에서 차원만 늘린다 (`_cross/corr_matrix_W144.csv` 참조).

`_cross/forward_selection.csv` 는 전진 선택 경로다. 기준은 **장력축 CV** 로 두었다 — 속도축
`final_test` 로 고르면 B 의 최종 테스트 집합에 선택이 새어 들어간다.

| 차원 | 추가 | T(tension) | T(final_test) | ΔT | 판정 |
|---|---|---|---|---|---|
| 2 | F0 | 0.6081 | 0.5414 | – | 기준점 |
| 3 | `hurst` | 0.6405 | 0.5488 | +0.0324 | 채택 |
| 4 | `spec_rolloff85` | 0.6449 | – | +0.0044 | **중단** (기준 0.005 미달) |

→ 선택 집합 = **`movstd` + `\|movslope\|` + `hurst` (3차원)**.

### 2.5 `_cross/window_sensitivity.csv` — **가장 중요한 결과**

| W | 집합 | T(tension) | T(final_test) | 진입 MAE(final_test) |
|---|---|---|---|---|
| 64 | F0 | **0.7181** | **0.8467** | **5.09 s** |
| 64 | selected | 0.6832 | 0.8382 | 5.21 s |
| 144 | F0 | 0.6081 | 0.5414 | 14.70 s |
| 144 | selected | 0.6405 | 0.5488 | 14.58 s |
| 256 | F0 | 0.3843 | 0.3400 | 25.31 s |
| 256 | selected | 0.4021 | 0.3620 | 23.90 s |

**창이 짧을수록 일관되게 좋다.** 창 144 의 feature 선택 이득(+0.032)은 창을 144 → 64 로 줄여
얻는 이득(+0.110 / +0.305)보다 훨씬 작다. 1.1절의 backfill 성질이 원인으로 보인다 — 창 144 는
`k ≤ 142` 가 전부 같은 값이고 transient 의 90 % 가 `k < 139` 이므로 과도 구간 분해능이 0이다.

### 2.6 `_cross/a5_normalization.csv` — 진폭 정규화 3종 (창 144)

| 좌표 | causal | F0 T(tension) | F0 T(final_test) | selected T(tension) | selected T(final_test) |
|---|---|---|---|---|---|
| `e` | 예 | 0.6081 | 0.5414 | 0.6405 | 0.5488 |
| `e ÷ 설정값` | 예 | 0.6092 | **0.7617** | 0.6317 | **0.7711** |
| `e ÷ MAD(run)` | **아니오** | 0.6205 | 0.7025 | 0.6200 | 0.6931 |

**계획 4.5절의 사전 예측이 틀렸다.** "비율 잔차는 악화 예상" 이라고 적었으나 고속 외삽에서
0.5414 → 0.7617 로 **크게 개선**된다. 장력축에서는 거의 변하지 않는다(+0.001). 즉 설정값 나누기는
장력 조건에 중립이면서 속도 외삽을 돕는다. `e ÷ MAD(run)` 도 개선되지만 run 전체를 봐야 해
**온라인 적용이 불가능**하고, 설정값 나누기보다 낮다.

### 2.7 그림

| 파일 | 축 | 목적 / 해석 |
|---|---|---|
| `_cross/DT_scatter_W144` | x = D, y = T. 왼쪽=장력축, 오른쪽=속도축. 점선 = F0 기준점 | **본 분석의 핵심 그림.** 오른쪽 위가 좋은 feature. 두 패널에서 순위가 뒤집히는 점들이 "조건별 재보정이 필요한 feature" 다. 계열 폴더에는 **그 계열만 색으로 강조하고 나머지를 회색으로 깐 같은 그림**이 들어 있다 |
| `_cross/increment_W144` | x = feature, y = ΔT | 막대가 0 위면 F0 에 더해 이득. 두 fold 묶음을 나란히 둔다 |
| `_cross/forward_selection` | x = 차원 수, y = T | 선택 경로. 차원이 늘어도 T 가 안 오르면 diag GMM 과적합 신호 |
| `_cross/corr_heatmap_W144` | feature × feature | feature 중복 확인. 진폭 계열끼리 0.9 이상으로 묶인다 |
| `_cross/window_sensitivity` | x = 창 길이, y = T | 2.5절 표의 그림판. **단조 감소**가 핵심 |
| `_cross/a5_normalization` | x = 좌표, y = T | 2.6절 표의 그림판 |

---

## 3. `outputs/04_overlap/` — 상태 정보와 조건 정보의 중첩 (A-4)

### 3.1 `{계열}/state_vs_condition.csv`

| 컬럼 | 설명 |
|---|---|
| `D_state` | 상태 식별력 (03_screening 의 D 와 같은 값) |
| `cond_auroc_scored` | **주 지표**. 조건 수준 쌍마다 AUROC 를 내서 평균. 학습이 없고 우연 수준 0.5 라 `D_state` 와 같은 잣대다 |
| `cond_auroc_steady` | 같은 값을 steady 시점만으로. 상태 변화가 섞이지 않은 순수 조건 정보 |
| `cond_accuracy` | 계획 4.4절이 적은 "정확도". feature 1차원 분위구간(20) Bayes 분류기, **run 단위 2-fold**. 우연 수준은 장력 0.200 / 속도 0.143 |

**읽는 법**: `state_vs_condition_W144` 그림에서 **왼쪽 위**(조건은 못 맞히고 상태는 잘 맞힘)가
이상적이다. 속도축에서 `half_mean_diff`(조건 0.519) 와 `movslope`(0.513) 가 가장 왼쪽이고,
`spec_centroid`(0.739)·`hurst`(0.721) 가 가장 오른쪽이다 — 스펙트럼·장기기억 계열은 상태와
조건을 함께 담는다.

### 3.2 `{계열}/state_diff_by_condition.csv` / `{계열}/state_diff_variation.csv`

조건 수준마다 `transient 평균 − steady 평균` 을 낸 표와, 그 값의 수준 간 변동계수다.
`abs_cv` 가 작을수록 "상태 신호의 크기가 조건에 무관" 하다는 뜻이다. 속도축에서
`movmean_abs_e` 0.34 / `phase_vel_std` 0.53 / `hurst` 0.60 이 작은 쪽이고,
`movkurt` 2.33 / `movskew` 2.19 가 큰 쪽이다.

### 3.3 그림

| 파일 | 축 | 해석 |
|---|---|---|
| `_cross/state_vs_condition_W144` | x = 조건 AUROC, y = 상태 AUROC | 왼쪽 위가 이상적. 세로 점선 = 조건 우연 수준 0.5. 계열 폴더에 계열 강조판이 따로 있다 |
| `_cross/state_diff_by_speed_W144` | x = 속도 설정, y = 상태 평균차 (kg) | 선이 평탄하면 조건 불변, 기울면 조건 의존 |

---

## 4. `outputs/05_visualize/` — 계열별 시각화 + A-6

판정용이 아니라 **설명용 그림**이다. 두 질문에 답한다 —
**Q1. steady 와 transient 가 다른가?  Q2. feature 를 실제로 뽑으면 결과가 어떻게 나오는가?**
대표 run 7개 (`T5_R5_V50` ~ `T5_R5_V200`, 장력·ReWinder 는 중앙값 5 kg). 목록은
`_cross/representative_runs.csv`.

### 4.1 계열 폴더마다 들어가는 그림 3종

| 파일 | 축 | 답하는 질문 / 해석 |
|---|---|---|
| `state_distribution_W144` | x = feature 값, y = 확률밀도. 파랑 = steady, 주황 = transient | **Q1.** 두 분포를 겹쳐 그린다. 겹치면 그 feature 로는 못 가른다. 제목의 `D` 는 run 내 AUROC(03_screening 산출), `d` 는 Cohen's d (평균차 ÷ 합동 표준편차, 부호는 + 면 transient 쪽이 크다) |
| `timeline_W144` | x = 시간(s). 행 = 원신호 `e` + 그 계열 feature, 열 = 저속 50 / 고속 200 run. 배경색 = 라벨 구간 | **Q2.** 추출 결과를 원신호와 같은 시간축에 쌓아 본다. **창 144 에서는 feature 가 과도 구간 내내 평평하다가 전이 직후에야 움직인다** — backfill 규약의 효과(한계 L2)가 눈으로 보이는 그림이다 |
| `window_separation` | x = feature, y = D. 막대 3개 = 창 64/144/256 | **Q1+Q2.** 같은 feature 를 창 3종에서 뽑았을 때 분리도가 어떻게 변하는지 |

### 4.2 A-6 그림 2종 (계획 4.6절) — 해당 제안 계열 폴더에 들어간다

| 파일 | 축 | 목적 / 해석 |
|---|---|---|
| `P1_spectral/spectrogram_W64` | x = 시간(s), y = 주파수(Hz, 0–2), 색 = PSD(dB) | 과도 → 정상 전이에서 스펙트럼이 어떻게 바뀌는지. 노란 선 = 전력가중 스펙트럼 중심, 주황 세로선 = 첫 steady 라벨, 흰 점선 = 회전 외란 `f_rot`. **진입 전 중심이 0.2–0.3 Hz 로 낮고 진입 후 0.5–0.7 Hz 로 올라간다** — 계획 3.1절의 방향 수정(과도가 저주파)과 일치한다 |
| `P2_phase/phase_portrait_lag1` | x = e(t), y = e(t+0.109 s) | steady(파랑)는 원점 근처에 뭉치고 transient(주황)는 대각선을 따라 음의 방향으로 길게 늘어난다. 과도가 충격이 아니라 **느린 처짐**임을 보여준다 (lag-1 상관이 높아 궤적이 대각선에 붙어 있다) |

**spectrogram 의 분석창은 64 샘플(7.0 s)이다.** 주 창 144(15.7 s)를 쓰면 과도 구간 전체가 창
하나에 들어가 전이 자체가 보이지 않는다 — 설명용 그림의 목적이 성립하지 않는다.

### 4.3 `_cross/` — 전 계열 한눈에 보기

`state_separation_heatmap` 은 16 feature × 창 3종의 D 를 셀 값까지 찍어 보여준다.
`state_separation.csv` 에 창 144 의 D 와 Cohen's d 가 들어 있다.

**여기서 T 와 반대 방향의 사실이 나온다** — D 는 대부분의 feature 에서 **창 144 가 최고**다
(`movstd` 0.891 → 0.975 → 0.948, `half_mean_diff` 0.731 → 0.934 → 0.933). 반면 조건 간
전이성 T 는 창 64 가 최고다(2.5절). 즉 **긴 창은 run 안에서는 더 잘 가르지만 조건을 넘으면 더
못 쓴다.** 창 선택이 "분리도" 가 아니라 "전이성" 기준으로 이뤄져야 하는 이유다.

효과크기 |d| 상위는 `movmean_abs_e` +2.80, `phase_vel_std` −2.57, `spec_centroid` −2.43,
`spec_rolloff85` −2.14 다. 부호가 음수인 셋은 **transient 쪽이 작다** — 스펙트럼·위상 계열에서
과도 구간이 저주파·저변동이라는 3.1절 방향 수정과 일치한다.

---

## 5. 결과를 B 로 넘길 때 반드시 함께 적을 것

| # | 내용 |
|---|---|
| 1 | feature **선택**에 라벨을 썼다. 정의에는 안 들어가지만 선택은 라벨 의존이므로 "A 에서 라벨로 고른 집합" 임을 명시한다 |
| 2 | 선택 기준은 **장력축 CV** 다. 속도축 `final_test` 는 고른 뒤 확인용으로만 봤다 |
| 3 | 창 길이 효과(2.5절)가 feature 선택 효과보다 훨씬 크다. CoST 변형의 창 설정을 먼저 정해야 한다 |
| 4 | 진폭 좌표 `e ÷ 설정값` 의 이득(2.6절)은 feature 선택과 독립이고, B 에서 바로 시험할 수 있다 |
