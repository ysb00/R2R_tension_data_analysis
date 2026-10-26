# Data Dictionary

생성 스크립트: `code/01_merge.py` (실행 시간 약 52 s, 대부분 xlsx 210개 읽기)
원본: `data/data_tensionSetting_{3..7}.zip` (zip 5개 × xlsx 42개 = 210 run, 1 xlsx = 1 운전조건)
출력 위치: `outputs/01_merge/` (`coding_guidelines.md` 3절의 `outputs/NN_이름/` 규칙)

설정값 3개(`Film-Tension`, `ReWinder-Tension`, `Web Speed`)는 파일명이 아니라 **원본 컬럼의 실제 값**에서 읽는다.
210 run 전체에서 세 설정값은 run 내 상수임을 확인했다.

---

## 1. `outputs/01_merge/merged_data.csv`

268,933 행 × 8 열. 원본 컬럼 6개는 **이름을 변경하지 않고** 그대로 사용하며, run 구분용 컬럼 2개를 추가했다.
정렬 순서: `Film-Tension Setting Value (kg)` → `ReWinder-Tension Setting Value (kg)` → `Web Speed setting value (mm/sec)` → `k`

### 1.1 원본 컬럼 (연구자 지정)

| 컬럼 | 역할 | 설명 |
|---|---|---|
| `Trigger 4` | label | expert 수동 라벨. 0 = 장비 신호가 아직 안정화되지 않은 영역, 1 = 장비가 steady state에 진입한 영역, 2 = ambiguous(라벨 부여가 모호). 제품 품질의 이상/정상을 뜻하지 않음 |
| `Film Tension #1 (kg)` | target | monitoring 대상 실측 신호 |
| `Film-Tension Setting Value (kg)` | 설정 조건 | 3, 4, 5, 6, 7 (run 내 상수) |
| `ReWinder-Tension Setting Value (kg)` | 설정 조건 | 2, 3, 4, 5, 6, 7 (run 내 상수) |
| `Web Speed setting value (mm/sec)` | 설정 조건 | 50, 75, 100, 125, 150, 175, 200 (run 내 상수) |
| `Web Current Speed (mm/sec)` | 실측 속도 | 활용 여부 미정 |

### 1.2 추가 컬럼

| 컬럼 | 설명 |
|---|---|
| `run_id` | run(원본 xlsx 1개) 식별자. 형식 `T{Film-Tension 설정}_R{ReWinder-Tension 설정}_V{Web Speed 설정}`. 예: `T3_R2_V50` |
| `k` | run 내부 샘플 순번 (0부터). 샘플 간격은 약 0.109 s (원본 `Date` 컬럼 기준 근사값, 3절 참조) |

라벨 분포 (샘플 단위): `Trigger 4` = 0 → 20,429 / 1 → 230,787 / 2 → 17,717.

원본 `Date` 컬럼은 `merged_data.csv`에 저장하지 않는다 (샘플 간격 산출에만 사용, 3절).

---

## 2. `outputs/01_merge/01_file_summary.csv`

210 행 × 15 열(1 행 = 1 run). 설정값 컬럼 3개는 원본 이름을 사용한다.
이하에서 e = `Film Tension #1 (kg)` − `Film-Tension Setting Value (kg)` (설정값 대비 장력 오차).

| 컬럼 | 설명 |
|---|---|
| `run_id` | 위와 동일 |
| `n` | run의 샘플 수 (1275–1300) |
| `n_label0` | `Trigger 4` = 0 (미안정) 샘플 수. 0 라벨이 없는 run: 26개 |
| `n_label1` | `Trigger 4` = 1 (steady state) 샘플 수 |
| `n_label2` | `Trigger 4` = 2 (ambiguous) 샘플 수. 2 라벨이 없는 run: 6개 |
| `first_steady_k` | `Trigger 4`가 처음 1이 되는 `k`, 즉 expert가 판단한 steady state 진입 시점 |
| `first_steady_sec` | `first_steady_k` × 0.109 s. **근사값** (평균 샘플 간격 가정) |
| `unstable_after_steady` | 첫 steady 이후에 다시 0으로 표시된 샘플 수. > 0 이면 steady 진입 후 재불안정 run (`T3_R7_V200`, `T5_R6_V50`, `T6_R7_V50`) |
| `Film-Tension Setting Value (kg)` | 설정 조건 (원본 컬럼명) |
| `ReWinder-Tension Setting Value (kg)` | 설정 조건 (원본 컬럼명) |
| `Web Speed setting value (mm/sec)` | 설정 조건 (원본 컬럼명) |
| `e_mean_steady` | steady(1) 구간 e의 평균: 설정값 대비 편향 |
| `e_std_steady` | steady(1) 구간 e의 표본표준편차(ddof=1): steady state의 변동 크기 |
| `e_min_unstable` | 미안정(0) 구간 e의 최솟값: 설정값 대비 최대 처짐(sag). 0 라벨이 없는 run(26개)은 빈칸 |
| `e_max_unstable` | 미안정(0) 구간 e의 최댓값: 설정값 대비 최대 위쪽 overshoot. 0 라벨이 없는 run(26개)은 빈칸 |

`first_steady_k`, `first_steady_sec`, `unstable_after_steady`는 steady(1) 라벨이 없는 run에서는 빈칸이다.

---

## 3. `outputs/01_merge/01_sample_interval.csv`

210 행(1 행 = 1 run). 등간격 0.109 s 가정의 근거를 확인하기 위한 보조 파일.
원본 `Date` 컬럼의 1차 차분으로 산출한다.

| 컬럼 | 설명 | 210 run 전체 범위 |
|---|---|---|
| `run_id` | 위와 동일 | |
| `n` | run의 샘플 수 | 1275–1300 |
| `dt_mean_sec` | run 내 샘플 간격의 평균 | 0.1094–0.1099 s (전체 평균 0.1094 s) |
| `dt_median_sec` | run 내 샘플 간격의 중앙값 | 0.1080–0.1090 s |
| `dt_min_sec` | run 내 샘플 간격의 최솟값 | 0.0210–0.1000 s |
| `dt_max_sec` | run 내 샘플 간격의 최댓값 | 0.1190–0.3870 s |

- run별 **평균·중앙값은 0.108–0.110 s로 매우 균일**하므로 등간격 0.109 s 가정(fs ≈ 9.1 Hz, Nyquist ≈ 4.6 Hz)은 합당하다.
- 다만 **개별 샘플 간격은 0.021–0.387 s까지 벌어진다**. 주파수 특징의 영향은 추후 검토 대상.

---

## 4. `outputs/01_merge/hyperparameters.csv`

`coding_guidelines.md` 1절에 따라 실행에 사용한 hyperparameter 값을 결과와 함께 저장한 파일.
컬럼: `name`, `value`.
