# 연구 계획 — 장력 신호의 transient / steady-state 식별력 feature 사전 분석

- 작성일: 2026-10-27
- 상태: **계획 (구현 전)**. 6절 승인 대기
- 저장소: `https://github.com/ysb00/R2R_tension_data_analysis` (본 저장소, 이하 **A**)
- 제안 방법 저장소: `https://github.com/ysb00/R2R_steadyState_identification_for_changing_condition` (이하 **B**, 로컬 `C:\Users\KIMM\research\informs-paper`)
- 이전 계획 문서 4개는 `docs/reference_*.md` 로 **참고용 보존** 상태다 (`reference_research_plan.md`, `reference_analysis_ideas_Claude.md`, `reference_feature_analysis_plan.md`, `reference_cost_modification_ideas.md`). **본 문서가 유효 계획이고**, 참고 문서의 전제(좌표계 중심 설계, 컬럼 추가, label-free 설계)는 본 문서에 승계되지 않는다.

---

## 0. 목적과 범위

B에서 표현학습 비교군 8개(변형 23종)를 평가한 결과 **CoST가 일관되게 1위**였다. 제안 방법은
**CoST의 변형**으로 가고, A는 그 변형에 넣을 **입력 feature를 결정하는 사전 분석**이다.

> **장력 신호에서 어떤 feature extraction을 하면 transient / steady-state 식별력이 올라가는가?**

### 0.1 범위 (사용자 확정 2026-10-27)

| 항목 | 확정 |
|---|---|
| **사용 신호** | **`Film Tension #1 (kg)` 단독.** `#2`·`#3`·`Rewinder roll diameter`·`Film 진행 길이`·`Web Current Speed` 는 쓰지 않는다 |
| **상태** | **2상태만.** `Trigger 4 = 2` (ambiguous, 17,717 샘플)는 **채점에서 제외** |
| **조건 축** | **장력축·속도축 모두** 검토 |
| **라벨** | A에서는 **사용한다** (feature 채점용). B의 제안 방법 구현에서는 **사용하지 않는다** |
| **데이터 처리** | **B의 방식을 그대로 따른다** (1절). A의 결론이 B의 비교군과 동등한 설정이어야 한다 |

단일 신호 제약의 결과: 설정값(T·R·V)은 **조건 분할과 잔차 정의에만** 쓰고 feature 입력으로는 쓰지 않는다.
다중 센서·권취 반경·속도 상관 계열 아이디어는 전부 **범위 밖**이고, `merged_data.csv` 의 **컬럼 추가도 불필요**하다.

### 0.2 라벨 사용의 경계

`coding_guidelines.md` 4절("`Trigger 4`는 평가 블록에서만 읽는다")과 충돌하지 않는다. B가 이미
같은 구조를 쓴다 — `evaluation.py` 의 `_labels()` **한 곳에서만** 라벨을 읽고, 방법 스크립트는
`usecols` 에 `Trigger 4` 를 넣지 않는다. A도 이 구조를 그대로 가져온다.

**지켜야 하는 제약**: feature의 **정의**(창 길이, 대역, 임계)에 라벨이 들어가면 B에서 재현할 수 없다.
feature 정의에 쓰이는 값은 **설정값과 신호 자체에서만** 나와야 한다. 라벨은 정의가 아니라
**순위 매기기**에만 쓴다.

---

## 1. B의 데이터 처리 방식 — 그대로 승계한다

`code/03_baselines/evaluation.py`, `01_stat_ssd.py`, `04_cost.py` 에서 확인한 실제 설정이다.
**이 절의 값은 전부 승계값이며 A에서 바꾸지 않는다.**

| 항목 | B의 값 | 출처 |
|---|---|---|
| 샘플 간격 | `DT = 0.109 s` (fs 9.174 Hz, Nyquist 4.587 Hz) | `evaluation.py:40` |
| 잔차 | `e = Film Tension #1 (kg) − Film-Tension Setting Value (kg)` | `01_stat_ssd.py` |
| **창 미완성 구간 처리** | **버리지 않는다.** 앞 `w−1` 시점은 **첫 완전창의 값으로 채운다**(backfill). 창 길이가 달라도 채점 대상 시점 수가 같게 하기 위함 | `01_stat_ssd.py:117-118` |
| CoST 쪽 같은 처리 | `SLIDING_PADDING = W − 1`, `CASUAL = True` — 좌측 causal 패딩으로 전 시점에 표현 부여 | `04_cost.py:205` |
| **주 창 길이** | **144 샘플 (15.7 s).** 회전 외란 주기에서 유도 — 둘레 783.5 mm ÷ V=50 의 1주기. **본문 표가 144** | `04_cost.py:69-73`, `01_stat_ssd.py:64` |
| 통계 baseline 창 | `WINDOW = 64` (약 7.0 s, **임의의 값**). `fixed` 변형이 이 값을 쓴다 | `01_stat_ssd.py:55` |
| 창 민감도 격자 (CoST) | `[144, 160, 256, 306, 512]` | `04_cost.py:73` |
| 후단 | `N_GMM_COMPONENTS = 2`, `GMM_COVARIANCE_TYPE = 'diag'`, 나머지 sklearn 기본값 | `evaluation.py:42-43` |
| 군집→상태 지정 | 학습 fold 에서 **`\|e\| 평균`이 작은 군집 = steady** | `evaluation.py:9` |
| **ambiguous 처리** | **`Trigger 4 != 2` 만 채점.** 단 **진입 시점은 ambiguous 포함 전체 시계열에서** 읽는다 | `evaluation.py:184, 187` |
| positive 규약 | **non-steady = positive** (AUPRC(ns)) | `evaluation.py:185` |
| 진입 판정 | `ENTRY_MIN_CONSEC = 10` — 10회 연속 steady 로 판정된 첫 시점 (**임의의 값**) | `evaluation.py:41` |
| 지표 기록 | **pooled**(fold 전체를 한 덩어리) 와 **run 평균** 둘 다 | `evaluation.py:156` |
| 시드 | `SEED = 0`, `N_SEEDS = 5` | `04_cost.py:66-67` |

### 1.1 승계의 직접 결과 — 이전 판단을 정정한다

본 계획 초안은 "창을 완전히 채운 시점부터 산출"을 가정해 **"창 144는 210 run 중 156개의
transient 를 전멸시키므로 사용 불가"** 라고 적었다. **B의 backfill 규약에서는 성립하지 않는다.**
B는 시점을 버리지 않으므로 채점 대상은 창 길이와 무관하게 동일하다. **창 144를 주 설정으로 쓴다.**

다만 backfill 규약의 성질 하나는 결과 해석에 반드시 들어가야 한다:

- `Trigger 4 = 0` 샘플의 `k` 분위는 **중앙값 64, 75% 99, 90% 139, 99% 264** 다.
- 창 144 + backfill 이면 `k ≤ 142` 의 **모든 시점이 같은 feature 값**(창 `[0..143]` 하나)을 받는다.
- 즉 **이동창 통계 feature 는 transient 내부에서 변하지 않는다.** transient 의 90% 가 `k < 139` 이므로
  transient 구간의 분해능이 사실상 0이다. 이것이 통계 baseline 의 진입 MAE 가 약 5 s 인 이유로 보인다.
- CoST 는 dilated conv + causal 패딩이라 좌측에서도 시점마다 다른 표현이 나오므로 같은 제약을 덜 받는다.

→ **창 144를 주 설정으로 두되, 민감도 격자를 함께 돌려 이 효과를 측정해 보고한다** (6절 #1).
격자에 **64**(B 통계 baseline 의 `fixed` 값)를 반드시 포함해 두 비교군과 모두 동등한 지점을 만든다.

### 1.2 데이터

| 항목 | 값 |
|---|---|
| 입력 | `outputs/01_merge/merged_data.csv` (**재생성 불필요**) |
| 규모 | 210 run = T 5수준 × R 6수준 × V 7수준, 268,933 샘플 (조건당 run 1개) |
| 신호 | `Film Tension #1 (kg)`. 범위 1.935–7.455 kg, **양자화 간격 0.010 kg** (전체 고유값 524개) |
| 라벨 | 0 = transient 20,429 / 1 = steady 230,787 / **2 = 채점 제외 17,717** |

`Trigger`·`Trigger 2`는 사용자 지시에 따라 무시한다.

### 1.3 회전 외란 주기 — 위상 계열이 막히는 이유

회전 둘레 `C = 783.5 mm` (B 측정값) 기준 `f_rot = V / C`:

| V (mm/s) | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| `f_rot` (Hz) | 0.064 | 0.096 | 0.128 | 0.160 | 0.191 | 0.223 | 0.255 |
| 1주기 (s) | 15.67 | 10.45 | 7.84 | 6.27 | 5.22 | 4.48 | 3.92 |
| 창 144 내 주기 수 | **1.00** | 1.50 | 2.00 | 2.50 | 3.00 | 3.51 | 4.01 |

- 창 144는 **V=50 에서 정확히 1주기**다. 위상 고정(PLV)은 여러 주기가 필요하므로 저속에서 성립하지 않는다.
- 게다가 1.1절에 따라 창 144의 feature 는 transient 내부에서 상수다. **위상 계열은 transient 에서 정보가 없다.**
- 창 144의 주파수 분해능은 `fs/144 = 0.0637 Hz` 이므로 `f_rot`(0.064–0.255 Hz)는 **bin 1–4** 에 놓인다.
  DC 를 제거하면 회전 성분이 분리 가능하다 — 창 144에서는 스펙트럼 feature 가 회전 대역을 볼 수 있다.

---

## 2. 평가 후단 — B의 `evaluation.py` 를 그대로 쓴다

```
시점별 d차원 feature -> StandardScaler(학습 조건) -> 2군집 diag GMM(학습 조건)
                     -> |e| 평균이 작은 군집 = steady -> 사후확률 >= 0.5 이면 steady
                     -> Trigger 4 != 2 로 채점 (non-steady = positive)
```

`coding_guidelines.md` 3절은 평가 전용 공통 파일을 허용하고, "외부 저장소 코드는 원본을 수정하지 않고
**커밋 hash 를 기록**한다"고 정한다. 따라서 **B의 `evaluation.py` 와 `splits.py` 를 수정 없이 복사**해
`code/evaluation.py` · `code/splits.py` 로 두고, B의 커밋 hash 를 `hyperparameters.csv` 에 기록한다.
이렇게 하면 A의 D·T 수치가 B의 비교군 수치와 **같은 코드로 산출된** 것이 된다.

A는 "학습된 표현" 대신 "**손으로 정의한 feature**"를 같은 후단에 넣는 **feature 수준 ablation** 이다.

---

## 3. feature 후보

기준 잔차: `e = Film Tension #1 (kg) − Film-Tension Setting Value (kg)` (B와 동일).
설정값은 run 내 상수이므로 `e` 는 `#1` 의 평행이동이다 — 단일 신호 제약을 위반하지 않는다.

### 3.1 사용자 제안 4계열의 실현성 판정

실측 확인 결과다. 근거는 1.1·1.3절과 아래 각 항목.

| # | 제안 feature | 판정 | 근거 |
|---|---|---|---|
| **P1** | Spectral centroid | **채택** | 단 **평균 제거 필수** (아래) |
| **P1** | Spectral roll-off (85%) | **채택** | 평균 제거 필수 |
| **P1** | Spectral bandwidth | **채택** | centroid 의 2차 모멘트 |
| **P2** | Phase-Locking Value | **기각** | 창 144가 V=50 에서 1주기뿐이고, backfill 로 transient 내부가 상수다 (1.1·1.3절) |
| **P2** | Instantaneous phase velocity | **조건부** | 광대역 신호의 Hilbert 위상은 물리적 의미가 약하다. **`unwrap` 필수** |
| **P3** | Rolling kurtosis | **채택** | scipy 기본은 **excess(정규=0)**. 제안문의 "3 기준"은 `fisher=False` 규약 |
| **P3** | Hurst exponent | **조건부** | 창 144(= 144점)에서 R/S·DFA 추정 분산이 크다. 추정 불안정성을 함께 보고 |
| **P4** | Tension-Speed correlation | **기각** | 0.1절 단일 신호 제약 위배. `Web Speed setting value` 는 run 내 상수라 상관계수 미정의 |
| **P4** | Hilbert energy envelope | **채택** | 창 144에서는 edge effect 비중이 짧은 창보다 작다 |

**P1 의 평균 제거 (필수)**: `Film Tension #1` 은 평균 2–7 kg 이므로 DC bin 이 전체 크기의 85% 를 넘는다.
평균을 제거하지 않으면 **`roll-off` 가 모든 창에서 0.0000 Hz 로 고정**되어 상수 feature 가 된다.
실측(창 144, `T7_R7_V200` steady): centroid 0.0223 → 평균 제거 후 **1.2818**, roll-off 0.0000 → **2.5484**.
`e` 를 입력으로 쓰면 설정값만큼 평행이동되지만 DC 가 완전히 사라지지는 않으므로 **창 내 평균을 뺀다.**

**P1 의 가설 방향 수정**: 제안문은 "transient 에서 충격으로 에너지가 고주파로 이동" 이라 했으나,
실측은 transient centroid(0.6772) < steady centroid(1.2818) 로 **반대**다. 이 공정의 과도 구간은
기계적 충격이 아니라 **7–11 s 에 걸친 느린 처짐(sag)** 이어서 에너지가 저주파에 쏠린다.
AUROC 는 부호에 무관하므로 feature 는 유효하지만 **논문의 설명 논리를 이 방향으로 써야 한다.**
(창 1개 표본이므로 4.2절에서 전 구간 재확인한다.)

**P2 의 `unwrap` 버그**: 제안 코드의 `np.std(np.diff(np.angle(...)))` 는 2π 점프를 섞는다.
실측(창 144): 점프 11회, `std` 1.609 → `unwrap` 적용 시 **0.278**.

### 3.2 feature 목록 (확정 대기)

전부 **단일 신호에서 causal 이동창(창 길이 `W`, backfill)** 으로 시점마다 산출하고, 라벨을 쓰지 않는다.

| 계열 | feature | 정의 |
|---|---|---|
| **F0 기준점** | `movstd`, `movslope` | 이동 표준편차, 이동 1차 기울기. **B의 `rolling_std_slope` 를 그대로 쓴다** (MovStd-Slope 와 동일) |
| **F1 진폭** | `movmean_abs_e`, `movrange`, `movmad` | `\|e\|` 이동평균, 이동 range, 이동 MAD |
| **F2 분포형상** | `movkurt`, `movskew` | 이동 첨도(excess), 이동 왜도 — P3 채택분 |
| **F3 스펙트럼** | `spec_centroid`, `spec_rolloff85`, `spec_bandwidth` | **창 평균 제거 + Hann 창** 후 rfft 기반 — P1 채택분 |
| **F4 추세/정상성** | `detrend_resid_std`, `half_mean_diff` | 창 내 1차 추세 제거 후 잔차 std, 전·후 반창 평균차 |
| **F5 포락선** | `env_mean`, `env_std` | Hilbert 포락선의 창 통계 — P4 채택분 |
| **F6 위상** | `phase_vel_std` | `std(diff(unwrap(angle(hilbert(e)))))` — P2 조건부 |
| **F7 장기기억** | `hurst` | R/S 기반 Hurst 지수 — P3 조건부. 추정 분산을 함께 보고 |

**F0 는 반드시 기준점으로 둔다** — B에서 이 2차원 통계 baseline 이 표현학습 5종을 이겼다
(C 실험 `final_test` AUPRC(ns) 0.847, CoST-W144 0.907 다음 2위).

---

## 4. 분석 설계

### 4.1 A-1. 조건 분할 — B의 분할을 그대로 쓴다

`splits.py` 를 복사해 쓰므로 fold 배정이 B와 동일하다 (`coding_guidelines.md` 3절: fold 배정은 csv 1개로 고정).

| 분할 | 학습 | 미학습 테스트 | run 수 |
|---|---|---|---|
| **장력축** (`tension`, leave-one-level-out) | T 5수준 중 4수준 | 남은 1수준 | 168 / 42, **5 fold** |
| **속도축** (`c`, 저속 → 고속) | V ∈ {50, 75, 100, 125} | V ∈ {150, 175, 200} | 120 / 90, **final_test** |

속도축 분할은 B가 실제 실행한 C 분할과 동일해야 한다 — A의 결론이 B의 C 실험 수치와 직접 비교된다.
`R` 은 분할축으로 쓰지 않고 두 분할 모두에서 6수준 전부 포함한다.

### 4.2 A-2. 두 지표

feature(또는 조합)마다 두 수치를 산출한다. **둘은 다른 질문이다.**

| 지표 | 정의 | 묻는 것 |
|---|---|---|
| **D (식별력)** | **조건 내**. 각 run 안에서 `Trigger 4` 0 vs 1 의 AUROC (ambiguous 제외). run별 산출 후 평균 | 이 feature 에 정보가 있는가 |
| **T (전이성)** | **조건 간**. 2절 후단을 학습 fold 에 적합 → 검증/테스트 fold 에서 **AUPRC(ns)**·오염률·진입 MAE | 그 정보를 **고정 경계**로 쓸 수 있는가 |

- 두 지표 모두 B의 `evaluation.py` 함수로 산출한다 (동일 채점).
- **채택 기준: D 높고 T 높음.** D 만 높은 feature 는 "조건별 재보정 필요"로 분류해 따로 기록한다.
- **D·T 산점도가 본 분석의 핵심 그림이다.**
- 이 단계에서 3.1절의 "가설 방향"(transient 가 저주파)을 **전 구간에서 재확인**한다 — feature 평균의
  상태별 차이 부호를 표로 낸다.

### 4.3 A-3. 증분 기여도

개별 순위만으로는 "무엇을 넣을까"에 답하지 못한다. 상관된 feature 는 중복이다.

| 단계 | 내용 |
|---|---|
| 1 | F0 기준점(`movstd` + `movslope`, 2차원)의 D·T 를 고정 — **B의 MovStd-Slope 수치와 일치하는지 먼저 확인한다** (승계 검증) |
| 2 | 후보를 **1개씩 F0 에 추가**해 `ΔD`·`ΔT` 산출 |
| 3 | `ΔT > 0` 인 feature 만 남겨 **전진 선택**으로 집합을 키운다 |
| 4 | 각 단계의 **차원 수 대비 이득**을 기록 — 후단이 diag GMM 이므로 차원이 늘면 과적합한다 |
| 5 | feature 간 상관행렬을 함께 내어 중복을 명시 |

1단계의 **승계 검증**이 중요하다. A의 F0 수치가 B의 `MovStd-Slope-fixed` 와 (같은 창·같은 fold·같은 시드에서)
일치하지 않으면 데이터 처리가 B와 다르다는 뜻이므로, 그 전에는 다음 단계로 가지 않는다.

전진 선택의 기준으로 라벨 기반 **T** 를 쓴다. 0.2절 정책상 허용된다 — 선택되는 것은
"어떤 feature 를 쓸지" 이고 feature 정의에는 라벨이 안 들어가기 때문이다.
단 **B 로 넘길 때 "A 에서 라벨로 고른 집합"임을 명시**한다 (7절 L1).

### 4.4 A-4. 상태 정보와 조건 정보의 중첩

CoST 변형 설계의 분기점이다. transient 의 처짐이 속도와 상관되면
(저속 −0.23 → 고속 −0.46 kg), **조건 정보를 지우면 상태 정보도 같이 지워진다.**

| 단계 | 내용 |
|---|---|
| 1 | 각 feature 에서 **상태**(0 vs 1) 를 맞히는 AUROC |
| 2 | 같은 feature 에서 **조건**(T 또는 V 수준)을 맞히는 정확도 |
| 3 | 두 값의 산점도 → **상태는 맞히고 조건은 못 맞히는 feature** 가 이상적 |

→ CoST 변형에서 **조건 불변화**(설정값 적대적 제거)와 **조건 조건화**(FiLM 등 설정값 주입) 중
어느 쪽을 택할지를 이 결과가 정한다.

### 4.5 A-5. 진폭 정규화 3종

| 좌표 | 근거 | 사전 예측 |
|---|---|---|
| `e` (kg) | **B의 현재 입력** | steady 변동이 T 3→7 kg 에서 0.0379–0.0412 kg 로 평탄해 이미 상당히 불변 |
| `e ÷ 설정값` | 비율 잔차 | **악화 예상** — 평탄한 변동을 7 kg 에서 2.3배 축소시켜 오히려 조건 의존성을 만든다 |
| `e ÷ MAD(run)` | run 전체 MAD 로 무차원화 | 조건마다 자동 적응하지만 **run 전체를 보므로 causal 하지 않다** |

`MAD(run)` 은 "steady 가 run 의 대부분(85.8%)" 이라는 분포 가정을 쓴다. 온라인 적용 한계로 기록한다.

### 4.6 A-6. 보조 시각화 (사용자 제안)

판정용이 아니라 **설명용 그림**이다. 대표 run 소수에만 적용한다.

| 그림 | 목적 |
|---|---|
| Time-frequency spectrogram | 과도 → 정상 전이에서 스펙트럼이 어떻게 좁아지는지 (3.1절의 방향 수정 근거) |
| Phase portrait `e(t)` vs `e(t+Δt)` | steady 의 limit cycle vs transient 의 이탈. `Δt` 는 **임의의 값** (6절 #7) |

---

## 5. 파일 구성 (예정)

| 파일 | 역할 | `Trigger 4` |
|---|---|---|
| `code/01_merge.py` | **기존 그대로.** 재생성 불필요 | 요약 통계만 |
| `code/evaluation.py` | **B에서 수정 없이 복사.** 커밋 hash 기록 | **읽음 (`_labels()` 유일 지점)** |
| `code/splits.py` | **B에서 수정 없이 복사.** fold 배정 동일 | 안 읽음 |
| `code/02_features.py` | 3.2절 feature 산출 (causal 이동창, backfill, 시점별) → `outputs/02_features/` | **안 읽음** |
| `code/03_screening.py` | A-2·A-3·A-5. D·T, 증분 기여도 → `outputs/03_screening/` | `evaluation.py` 경유 |
| `code/04_overlap.py` | A-4. 상태 vs 조건 → `outputs/04_overlap/` | `evaluation.py` 경유 |
| `code/05_visualize.py` | A-6. spectrogram, phase portrait → `outputs/05_visualize/` | 그림 주석용만 |
| `docs/outputs_guide.md` | csv 컬럼, plot 축·목적·해석 | |
| `docs/worklog.md` | 단계별 기록 | |

**라벨 격리**: `02_features.py` 는 `usecols` 에서 `Trigger 4` 를 제외하고 feature csv 에도 라벨을
저장하지 않는다. 라벨은 `evaluation.py::_labels()` 가 `run_id`·`k` 로 조인해 가져온다. **B와 동일 구조.**

`coding_guidelines.md` 1·7절에 따라 hyperparameter 는 코드 상단에 모아 `hyperparameters.csv` 로
결과와 함께 저장하고, 결과는 csv + plot (PNG 300 dpi + PDF, 축·범례 영어)으로 남긴다.

---

## 6. 승인 대기 — 구현 전 결정이 필요한 값

`coding_guidelines.md` 6절에 따라 아래는 임의로 정하지 않는다.

| # | 항목 | 제안 | 성격 |
|---|---|---|---|
| **1** | **이동창 길이** | **144 를 주 설정**(B 본문 표와 동일), 민감도용으로 **{64, 144, 256}** 추가. 64 는 B 통계 baseline 의 `fixed` 값이라 두 비교군과 모두 동등한 지점이 생긴다 | 144·64 는 **승계값**, 256 은 B의 CoST 격자에서 가져온 값 |
| 2 | 창 미완성 구간 처리 | **B와 동일 — 첫 완전창 값으로 backfill.** 시점을 버리지 않는다 | **승계값** |
| 3 | `Trigger 4 = 2` 처리 | **채점 제외, 진입 시점은 전체 시계열에서 읽음** — B `evaluation.py` 동작 그대로 | **승계값** (사용자 확정과 일치) |
| 4 | roll-off 에너지 비율 | **85%** (제안문 승계). 대안 90·95% | 승계값 |
| 5 | 스펙트럼 창 함수 | **Hann** (B의 회전 분석과 동일) | 승계값 |
| 6 | kurtosis 규약 | **excess (scipy 기본, 정규 = 0)** 로 통일하고 문서에 명시 | 결정 |
| 7 | Phase portrait 의 `Δt` | **1 샘플(0.109 s)** | **임의의 값** |
| 8 | D 지표 | **run 내 AUROC 평균** (B의 `runmean_` 규약과 동일) | 승계값 |
| 9 | T 지표 우선순위 | **AUPRC(ns) 주 지표**, 오염률·진입 MAE 병기 | **승계값** |
| 10 | 전진 선택 중단 기준 | `ΔT < 0.005` 이면 중단, **최대 6차원** | **임의의 값** |
| 11 | F6(위상)·F7(Hurst) 포함 여부 | **포함하되 추정 불안정성을 한계로 기록.** 제외도 가능 | 결정 |
| 12 | `e ÷ MAD(run)` 사용 승인 | 사용 제안 (비causal 한계 기록) | 결정 |
| 13 | A-6 대표 run 선정 | **각 속도 1 run, R·T 는 중앙값 수준** | **임의의 값** |
| 14 | 시드 반복 | **`SEED = 0`, `N_SEEDS = 5`** (B와 동일) | **승계값** |

승인이 필요한 실질 항목은 **#6, #7, #10, #11, #12, #13** 이고, 나머지는 B 승계값이다.

---

## 7. 한계

| # | 한계 |
|---|---|
| L1 | feature **선택**에 라벨을 쓴다 (4.3절). 정의에는 안 들어가지만 선택은 라벨 의존이므로 B 에 넘길 때 명시해야 한다 |
| L2 | **창 144 + backfill 이면 이동창 feature 가 transient 내부에서 상수다** (1.1절). transient 의 90% 가 `k < 139` 이므로 과도 구간 분해능이 사실상 0이다. 창 64 결과와 대조해 이 효과를 분리한다 |
| L3 | **위상 고정(PLV) 계열을 측정할 수 없다** (1.3절). 창 144가 V=50 에서 회전 1주기뿐이다 |
| L4 | 신호가 **0.010 kg 로 양자화**되어 있다. steady 변동(약 0.038 kg)이 양자화 간격의 약 4배뿐이라 고주파 성분은 양자화 잡음과 섞인다 |
| L5 | 조건당 run 이 1개다. "조건 X 에서 이 값이 크다"가 조건 효과인지 그 run 하나의 특성인지 분리할 수 없다 |
| L6 | 등간격 0.109 s 가정 위에 서 있다. 개별 샘플 간격은 0.021–0.387 s 로 벌어진다 |
| L7 | 손으로 정의한 feature 의 ablation 결과가 **CoST 내부 표현**에 그대로 옮겨진다는 보장이 없다. A 는 "무엇을 넣을지"의 근거이고, 효과는 B 에서 확인된다 |
| L8 | 라벨 시간 분해능은 5 샘플 = 0.55 s 다 (전이 인덱스의 93.3% 가 `k ≡ 1 (mod 5)`) |
