# P1 — 스펙트럼 계열 (Spectral features)

- 대상 폴더: `outputs/{02_features,03_screening,04_overlap,05_visualize}/P1_spectral/`
- feature 3종: `spec_centroid`, `spec_rolloff85`, `spec_bandwidth`
- 작성일: 2026-10-28 / 근거 실행: 창 64·144·256, seed 0–4, fold 6개

---

## 1. 방법론 — 무엇을 재는가

창 안의 신호를 주파수 영역으로 옮겨 **에너지가 어느 주파수에 몰려 있는지**를 세 숫자로 요약한다.
"과도 상태와 정상 상태는 진동의 성격이 다르다"는 가정을 쓰며, 진폭이 아니라 **스펙트럼의 모양**을
보므로 조건이 바뀌어 진폭이 커져도 값이 따라 움직이지 않는다 — 이것이 진폭 계열과의 결정적 차이다.

| feature | 정의 | 직관 |
|---|---|---|
| `spec_centroid` | `Σ f·\|X(f)\| / Σ\|X(f)\|` | 스펙트럼의 무게중심. 높으면 고주파가 강하다 |
| `spec_rolloff85` | 누적 진폭이 전체의 85 %가 되는 최저 주파수 | 에너지의 85 %가 이 주파수 아래에 있다 |
| `spec_bandwidth` | `√(Σ \|X(f)\|·(f−centroid)² / Σ\|X(f)\|)` | centroid 기준 퍼짐 폭. 넓으면 여러 주파수에 흩어져 있다 |

---

## 2. 어떻게 적용했는가

### 2.1 산출 절차 (`code/02_features.py`)

```
e(t) = Film Tension #1 (kg) − Film-Tension Setting Value (kg)
  → 시점 t 에서 뒤쪽(causal) 창 x = e[t−W+1 … t]
  → 창 내 평균 제거  x_c = x − mean(x)          ★ 필수 (2.2절)
  → Hann 창 곱       x_w = x_c · hann(W)
  → rfft 진폭        |X(f)|,  f = rfftfreq(W, 0.109 s)
  → centroid / rolloff(85 %) / bandwidth
```

- **창 길이** `W ∈ {64, 144, 256}` 샘플 (7.0 / 15.7 / 27.9 s). 주 설정 144 (승인 #1).
- **창 미완성 구간**: 앞 `W−1` 시점은 버리지 않고 **첫 완전창의 값으로 backfill** (B 승계).
- **주파수 분해능** = `fs/W`. 창 144에서 0.0637 Hz, Nyquist 4.587 Hz.
- DC bin 을 따로 버리지 않는다 — 평균 제거 + Hann 이후에도 남는 미세한 DC 는 포함해 계산한다.

### 2.2 평균 제거는 선택이 아니라 필수다

`Film Tension #1` 의 평균은 2–7 kg 이라 **DC bin 이 전체 진폭의 85 %를 넘는다.** 평균을 빼지 않으면
누적 진폭이 첫 bin에서 이미 85 %를 넘어 **`spec_rolloff85` 가 모든 창에서 0.0000 Hz 로 고정**된다
(= 상수 feature, 정보량 0). 실측(창 144, `T7_R7_V200` steady 구간):

| | 평균 제거 전 | 평균 제거 후 |
|---|---|---|
| `spec_centroid` | 0.0223 Hz | **1.2818 Hz** |
| `spec_rolloff85` | 0.0000 Hz | **2.5484 Hz** |

제안 코드에는 이 처리가 없었다. 잔차 `e` 를 쓰면 설정값만큼 평행이동되지만 DC 가 완전히 사라지지는
않으므로 **창 내 평균을 뺀다**.

### 2.3 임의로 정한 값

| 값 | 설정 | 근거 |
|---|---|---|
| roll-off 비율 | **85 %** | 제안문 승계 (승인 #4). 대안 90·95 % |
| 창 함수 | **Hann** | B 의 회전 외란 분석과 같은 창 (승인 #5) |
| DC bin 처리 | 포함 | 평균 제거로 이미 억제됨 |

---

## 3. 결과 해석

### 3.1 가설의 방향이 반대였다 — 논문 서술을 바꿔야 한다

제안문은 "transient 에서 충격으로 에너지가 고주파로 이동" 이라고 적었다. **실측은 정반대다.**

| | transient 평균 | steady 평균 | Cohen's d |
|---|---|---|---|
| `spec_centroid` | 낮다 | 높다 | **−2.43** |
| `spec_rolloff85` | 낮다 | 높다 | **−2.14** |

이 공정의 과도 구간은 기계적 충격이 아니라 **7–11 s에 걸친 느린 처짐(sag)** 이어서 에너지가 저주파에
쏠린다. `outputs/05_visualize/P1_spectral/spectrogram_W64` 가 같은 사실을 그림으로 보여준다 —
steady 진입 전 전력가중 스펙트럼 중심이 0.2–0.3 Hz, 진입 후 0.5–0.7 Hz 다.

AUROC·AUPRC 는 부호에 무관하므로 **feature 자체는 유효하다.** 바꿔야 하는 것은 설명 논리다.

### 3.2 식별력 D 와 전이성 T

| feature | D(W64) | D(W144) | D(W256) | T(tension) | T(final_test) |
|---|---|---|---|---|---|
| `spec_centroid` | 0.838 | 0.945 | 0.948 | 0.674 | **0.839** |
| `spec_rolloff85` | 0.798 | 0.914 | 0.926 | 0.524 | 0.774 |
| `spec_bandwidth` | 0.757 | 0.838 | 0.841 | 0.390 | 0.637 |

- **`spec_centroid` 의 T(final_test) 0.839 는 16개 feature 중 3위다** (1위 `half_mean_diff` 0.904,
  2위 `phase_vel_std` 0.867). 기준점 F0 의 0.541 을 크게 넘는다.
- 세 feature 모두 **창이 길수록 D 가 올라간다** — 주파수 분해능이 좋아지기 때문이다.
  다른 계열이 창 144에서 꺾이는 것과 달리 P1 은 창 256까지 단조 증가한다.
- 다만 T 는 창 64가 가장 좋다(`_cross/window_sensitivity.csv`). **분해능과 전이성이 반대 방향**이다.

### 3.3 증분 기여도 — **세 feature 모두 F0 를 개선한다**

F0(`movstd`+`|movslope|`, 2차원)에 1개씩 더했을 때 (창 144):

| 추가 | ΔT(tension) | ΔT(final_test) |
|---|---|---|
| `spec_centroid` | **+0.0079** | **+0.0037** |
| `spec_rolloff85` | **+0.0069** | +0.0002 |
| `spec_bandwidth` | **+0.0056** | −0.0011 |

**P1 은 세 feature 전부 ΔT(tension) > 0 인 두 계열 중 하나다** (나머지 하나는 P3 비정상성).
P4 물리기반과 P5 추가 계열은 **전원 음수**이고 P2 위상도 음수다 — 계열별로 보면 다음과 같다.

| 계열 | ΔT(tension) 부호 |
|---|---|
| **P1 스펙트럼** | `spec_centroid` + / `spec_rolloff85` + / `spec_bandwidth` + — **전원 양수** |
| P3 비정상성 | `hurst` + / `movskew` + / `movkurt` + — 전원 양수 |
| P2 위상 | `phase_vel_std` − |
| P4 물리기반 | `env_std` − / `env_mean` − |
| P5 추가 | 5종 전부 − |

**진폭을 재는 계열(P4·P5)은 전부 악화시키고, 모양·주파수를 재는 계열(P1·P3)은 전부 개선한다.**
`movstd` 가 이미 진폭을 담당하고 있으므로, 거기에 또 진폭을 더하면 diag GMM 후단에서 차원만
늘어난다 — 스펙트럼 모양은 `movstd` 와 중복되지 않는 정보를 담고 있다는 뜻이다.

전진 선택에서 `spec_rolloff85` 가 4번째 후보로 뽑혔으나 ΔT +0.0044 가 중단 기준 0.005 에 미달해
집합에 들어가지 못했다 (`_cross/forward_selection.csv`).

### 3.4 조건 정보가 가장 많이 섞인 계열이다 — CoST 변형 설계에 직결

| feature | 상태 AUROC | 속도 조건 AUROC | steady 구간만 | 속도 분류 정확도 (우연 0.143) |
|---|---|---|---|---|
| `spec_centroid` | 0.945 | **0.739** | 0.783 | **0.368** |
| `spec_rolloff85` | 0.914 | 0.682 | 0.717 | 0.305 |
| `spec_bandwidth` | 0.838 | 0.614 | 0.635 | 0.198 |

`spec_centroid` 는 **16개 중 조건 식별력 1위**다. 회전 외란 주파수가 `f_rot = V/783.5 mm` 로
속도에 비례해 0.064 → 0.255 Hz 로 4배 이동하므로, 스펙트럼 무게중심이 속도를 그대로 읽는다.

→ **CoST 변형에서 조건 불변화(설정값 적대적 제거)를 택하면 이 계열이 가장 먼저 지워진다.**
조건 조건화(FiLM 등 설정값 주입)를 택해야 P1 의 정보를 살릴 수 있다.

---

## 4. csv 파일과 컬럼

### `outputs/02_features/P1_spectral/`
| 파일 | 컬럼 | 의미 |
|---|---|---|
| `features_W{64,144,256}.csv.gz` | `run_id`, `k`, `spec_centroid`, `spec_rolloff85`, `spec_bandwidth` | 시점별 값 (268,933행). 단위 Hz. `k` 는 run 내 시점 index, `merged_data.csv` 와 1:1 |
| `feature_summary.csv` | `W`, `feature`, `family`, `unit`, `mean`, `std`, `min`, `q05`, `median`, `q95`, `max`, `n_unique` | 창별 기술통계. `n_unique` 가 1이면 상수 feature (평균 제거 실패 신호) |

### `outputs/03_screening/P1_spectral/`
| 파일 | 핵심 컬럼 | 의미 |
|---|---|---|
| `D_summary.csv` | `D_raw` / `D_raw_std` / `D` | `D_raw` = 방향 보정 전 run 평균 AUROC, `D` = `max(a, 1−a)` 의 run 평균. **P1 은 `D_raw < 0.5` 인 run 이 많다** (transient 가 저주파라서) — 그래서 `D` 와 `D_raw` 가 크게 다르다 |
| `DT_single.csv` | `D`, `T_tension`, `T_final_test`, `entry_mae_*`, `contam_*` | feature 1개만 후단에 넣었을 때. `entry_mae` = steady 진입 시점 오차(s), `contam` = steady 로 판정한 시점 중 실제 non-steady 비율 |
| `increment.csv` | `dT_tension`, `dT_final_test` | F0 에 더했을 때의 T 변화. **양수면 넣을 가치가 있다** |
| `state_mean_diff.csv` | `mean_transient`, `mean_steady`, `diff`, `sign` | 3.1절 방향 확인표. `sign` 이 `−` 면 transient 쪽이 작다 |

### `outputs/04_overlap/P1_spectral/`
| 파일 | 핵심 컬럼 | 의미 |
|---|---|---|
| `state_vs_condition.csv` | `D_state`, `cond_auroc_scored`, `cond_auroc_steady`, `cond_accuracy`, `cond_accuracy_chance` | 조건 AUROC 는 **수준 쌍마다 AUROC 를 내서 평균**한 값이라 우연 수준이 0.5 다. 정확도는 분위구간 Bayes 분류기(run 단위 2-fold)라 우연 수준이 장력 0.200 / 속도 0.143 로 다르다 — **두 열을 같은 잣대로 비교하면 안 된다** |
| `state_diff_by_condition.csv` | `level`, `mean_transient`, `mean_steady`, `diff`, `auroc_within_level` | 조건 수준마다 따로 낸 상태 평균차 |
| `state_diff_variation.csv` | `mean_diff`, `std_diff`, `abs_cv` | `abs_cv = std_diff / |mean_diff|`. 작을수록 상태 신호 크기가 조건에 무관하다 |

---

## 5. plot — 축·색·마커의 의미

### 공통 색 규약 (전 그림)

| 대상 | 색 | 쓰이는 곳 |
|---|---|---|
| **P1 (이 계열)** | 주황 `#d55e00`, 마커 ▽ | 산점도, 시계열 선 |
| P0 기준점 / P2 / P3 / P4 / P5 | 검정 ● / 보라 ✕ / 초록 △ / 노랑 ✚ / 회색 ■ | 산점도 |
| **transient** | 주황 `#d55e00` | 상태별 분포, 시계열 배경, phase portrait |
| **steady** | 파랑 `#0072b2` | 〃 |
| ambiguous | 회색 `#999999` | 〃 (채점에서는 제외) |
| 창 64 / 144 / 256 | 연한 → 진한 파랑 | 창별 분리도 막대 |

### 5.1 `03_screening/P1_spectral/DT_scatter_W144`
- 가로 = **D** (run 내 상태 AUROC, 방향 보정). 세로 = **T** (AUPRC non-steady).
- 왼쪽 패널 = 미학습 장력 수준 5-fold, 오른쪽 = 저속 학습 → 고속 테스트.
- **검은 가로 점선 = F0 기준점의 T.** 이 선 위에 있어야 F0 보다 낫다.
- **P1 점만 주황이고 나머지 계열은 회색**이다. `_cross/DT_scatter_W144` 에는 전 계열이 색으로 있다.
- 읽는 법: 오른쪽 위가 좋다. 두 패널에서 순위가 뒤집히는 점 = 조건별 재보정이 필요한 feature.

### 5.2 `05_visualize/P1_spectral/state_distribution_W144`
- 가로 = feature 값(Hz), 세로 = 확률밀도. 파랑 = steady, 주황 = transient.
- 제목의 `D` = run 내 AUROC, `d` = Cohen's d (**부호 포함** — 음수면 transient 쪽이 작다).
- 가로 범위는 0.5–99.5 분위로 잘랐다 (꼬리를 두면 본체가 한 칸으로 눌린다).
- 읽는 법: 두 분포가 겹칠수록 못 가른다. P1 은 **transient 가 왼쪽(저주파)** 에 있다.

### 5.3 `05_visualize/P1_spectral/timeline_W144`
- 가로 = 시간(s), 0–60 s. 행 = 원신호 `e` → `spec_centroid` → `spec_rolloff85` → `spec_bandwidth`.
- 열 = 대표 run 2개 (저속 50 / 고속 200 mm/s, 장력·ReWinder 는 중앙값 5 kg).
- **배경색 = 라벨 구간** (주황 transient / 파랑 steady / 회색 ambiguous).
- 읽는 법: **창 144 에서는 feature 가 과도 구간 내내 평평하다.** backfill 규약 때문에 `k ≤ 142` 가
  전부 같은 값을 받기 때문이다 (한계 L2). 전이 직후부터 비로소 움직인다.
- `spec_rolloff85` 가 계단처럼 보이는 것은 정상이다 — 값이 FFT bin 격자(0.0637 Hz 간격)에만 놓인다.

### 5.4 `05_visualize/P1_spectral/window_separation`
- 가로 = feature, 세로 = D. 막대 3개 = 창 64 / 144 / 256 (연한 → 진한 파랑).
- 회색 점선 = 우연 수준 0.5. 세로축은 0.5–1.0 으로 고정.
- 읽는 법: P1 은 창이 길수록 D 가 올라간다 (주파수 분해능 효과).

### 5.5 `05_visualize/P1_spectral/spectrogram_W64` — A-6 설명용 그림
- 가로 = 시간(s), 세로 = 주파수(Hz, **0–2 Hz 로 잘랐다** — Nyquist 4.59까지 그리면 정보 대역이 눌린다).
- 색 = 전력스펙트럼밀도(dB), 밝을수록 약하고 진할수록 강하다. 색 범위는 전 패널 공통(2–98 분위).
- **노란 선 = 전력가중 스펙트럼 중심**, **주황 세로선 = 첫 steady 라벨**, **흰 점선 = 회전 외란 `f_rot`**.
- 패널 7개 = 속도 50 → 200 mm/s.
- **분석창은 64 샘플(7.0 s)이다.** 주 창 144(15.7 s)를 쓰면 과도 구간 전체가 창 하나에 들어가
  전이 자체가 보이지 않는다.
- 읽는 법: 주황 세로선 **왼쪽**에서 노란 선이 낮고(0.2–0.3 Hz) 오른쪽에서 올라간다(0.5–0.7 Hz).
  3.1절의 방향 수정을 눈으로 확인하는 지점이다.

### 5.6 `04_overlap/P1_spectral/state_vs_condition_W144`
- 가로 = **조건 AUROC**(수준 쌍 평균), 세로 = **상태 AUROC**. 왼쪽 패널 = 장력축, 오른쪽 = 속도축.
- 세로 점선 = 조건 우연 수준 0.5.
- 읽는 법: **왼쪽 위가 이상적**(상태는 맞히고 조건은 못 맞힘). **P1 은 오른쪽에 치우쳐 있다** —
  조건 정보를 많이 담았다는 뜻이고, 3.4절의 설계 판단으로 이어진다.

---

## 6. 한계

| # | 내용 |
|---|---|
| 1 | **창 144 + backfill 이면 과도 구간 내부에서 값이 상수**다. transient 의 90 %가 `k < 139` 이므로 과도 구간 분해능이 사실상 0이다 (계획 L2) |
| 2 | 신호가 **0.010 kg 로 양자화**돼 있다. steady 변동(약 0.038 kg)이 양자화 간격의 4배뿐이라 고주파 성분은 양자화 잡음과 섞인다 (계획 L4) |
| 3 | `spec_rolloff85` 는 FFT bin 격자 위의 **이산값**이다. 창 144에서 간격 0.0637 Hz |
| 4 | 조건(속도) 정보를 가장 많이 담는다 (3.4절). 조건 불변화 설계와 충돌한다 |
| 5 | 등간격 0.109 s 가정 위에 서 있다. 개별 샘플 간격은 0.021–0.387 s 로 벌어진다 (계획 L6) |
