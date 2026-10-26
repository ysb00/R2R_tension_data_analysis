"""
01_merge.py

역할: 원본 zip 5개(각 xlsx 42개 = 210 run)를 읽어 분석용 통합 csv와 run 단위 요약 csv를 생성한다.
입력: data/data_tensionSetting_{3..7}.zip
출력: outputs/01_merge/merged_data.csv, outputs/01_merge/01_file_summary.csv,
      outputs/01_merge/hyperparameters.csv, outputs/01_merge/01_sample_interval.csv

주의: 원본 컬럼명은 변경하지 않는다. 라벨(`Trigger 4`)은 요약 통계 산출에만 사용하며,
      이 단계는 특징 추출/학습 단계가 아니다.
"""

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Hyperparameter (사용자 지정 영역)
# =============================================================================
RANDOM_SEED = 42                      # 랜덤 시드 (본 단계는 무작위성 없음, 재현성 위해 고정)

DATA_DIR = Path("data")               # 원본 zip 위치
OUT_DIR = Path("outputs/01_merge")    # 결과 저장 위치
TENSION_SETTINGS = [3, 4, 5, 6, 7]    # Film-Tension Setting Value (kg), zip 파일명에 대응

SAMPLE_DT_SEC = 0.109                 # 샘플 간격 가정값 (s). 원본 Date 기준 평균의 근사값 — 임의의 값

LABEL_UNSTABLE = 0                    # Trigger 4 = 0 : 미안정 (transient)
LABEL_STEADY = 1                      # Trigger 4 = 1 : steady state
LABEL_AMBIGUOUS = 2                   # Trigger 4 = 2 : ambiguous (분석 제외 대상)

# 원본 컬럼명 (변경 금지). merged_data.csv에 남길 6개 컬럼
COL_LABEL = "Trigger 4"
COL_TENSION = "Film Tension #1 (kg)"
COL_TENSION_SET = "Film-Tension Setting Value (kg)"
COL_REWINDER_SET = "ReWinder-Tension Setting Value (kg)"
COL_SPEED_SET = "Web Speed setting value (mm/sec)"
COL_SPEED_CUR = "Web Current Speed (mm/sec)"
COL_DATE = "Date"                     # 샘플 간격 확인에만 사용, merged_data.csv에는 저장하지 않음

KEEP_COLS = [
    COL_LABEL,
    COL_TENSION,
    COL_TENSION_SET,
    COL_REWINDER_SET,
    COL_SPEED_SET,
    COL_SPEED_CUR,
]

np.random.seed(RANDOM_SEED)


# =============================================================================
# 블록 1. zip 내 xlsx 전체를 순회하며 run 단위로 읽어 누적
#   - run_id = T{Film-Tension 설정}_R{ReWinder-Tension 설정}_V{Web Speed 설정}
#   - k = run 내부 샘플 순번 (0부터)
#   - 설정값은 파일명이 아니라 원본 컬럼의 실제 값에서 가져온다 (파일명-데이터 불일치 방지)
# =============================================================================
t_start = time.time()
t_block = time.time()

run_frames = []          # run별 DataFrame 누적
interval_rows = []       # run별 샘플 간격 통계 (등간격 가정의 근거 확인용)

for tension in TENSION_SETTINGS:
    zip_path = DATA_DIR / f"data_tensionSetting_{tension}.zip"
    zf = zipfile.ZipFile(zip_path)
    xlsx_names = sorted(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
    print(f"[zip] {zip_path.name}: xlsx {len(xlsx_names)}개")

    for name in xlsx_names:
        df_raw = pd.read_excel(zf.open(name))

        # run 내 상수인 설정값 3개를 실제 데이터에서 추출
        v_tension = int(df_raw[COL_TENSION_SET].iloc[0])
        v_rewinder = int(df_raw[COL_REWINDER_SET].iloc[0])
        v_speed = int(df_raw[COL_SPEED_SET].iloc[0])
        run_id = f"T{v_tension}_R{v_rewinder}_V{v_speed}"

        # 설정값이 run 내에서 정말 상수인지 확인 (데이터 가정 검증)
        n_uniq = (
            df_raw[COL_TENSION_SET].nunique()
            + df_raw[COL_REWINDER_SET].nunique()
            + df_raw[COL_SPEED_SET].nunique()
        )
        if n_uniq != 3:
            print(f"  [경고] {run_id}: 설정값이 run 내에서 상수가 아님 (unique 합 {n_uniq})")

        # 원본 Date로 샘플 간격 통계 산출 (merged_data.csv에는 Date를 저장하지 않음)
        dt_sec = pd.to_datetime(df_raw[COL_DATE]).diff().dt.total_seconds().to_numpy()[1:]
        interval_rows.append(
            {
                "run_id": run_id,
                "n": len(df_raw),
                "dt_mean_sec": dt_sec.mean(),
                "dt_median_sec": np.median(dt_sec),
                "dt_min_sec": dt_sec.min(),
                "dt_max_sec": dt_sec.max(),
            }
        )

        # 분석에 사용할 6개 원본 컬럼 + 추가 컬럼 2개
        df_run = df_raw[KEEP_COLS].copy()
        df_run["run_id"] = run_id
        df_run["k"] = np.arange(len(df_run))
        run_frames.append(df_run)

print(f"[블록 1] run {len(run_frames)}개 읽기 완료 ({time.time() - t_block:.1f} s)")


# =============================================================================
# 블록 2. 통합 및 정렬
#   정렬: Film-Tension 설정 → ReWinder-Tension 설정 → Web Speed 설정 → k
# =============================================================================
t_block = time.time()

df_all = pd.concat(run_frames, ignore_index=True)
df_all = df_all.sort_values(
    [COL_TENSION_SET, COL_REWINDER_SET, COL_SPEED_SET, "k"], kind="mergesort"
).reset_index(drop=True)

# 컬럼 순서: 원본 6개 → 추가 2개
df_all = df_all[KEEP_COLS + ["run_id", "k"]]

print(f"[블록 2] merged_data shape = {df_all.shape}")
print(f"         run 수 = {df_all['run_id'].nunique()}")
print(f"         Trigger 4 분포 = {df_all[COL_LABEL].value_counts().sort_index().to_dict()}")
print(f"         정렬/통합 완료 ({time.time() - t_block:.1f} s)")


# =============================================================================
# 블록 3. run 단위 요약 (01_file_summary.csv)
#   e = Film Tension #1 (kg) - Film-Tension Setting Value (kg)  : 설정값 대비 장력 오차
#   라벨은 요약 통계 산출 목적으로만 사용
# =============================================================================
t_block = time.time()

df_all["_e"] = df_all[COL_TENSION] - df_all[COL_TENSION_SET]   # 잔차 e (임시 컬럼, 저장 안 함)

summary_rows = []
for run_id, g in df_all.groupby("run_id", sort=False):
    label = g[COL_LABEL].to_numpy()
    e = g["_e"].to_numpy()
    k = g["k"].to_numpy()

    is_steady = label == LABEL_STEADY
    is_unstable = label == LABEL_UNSTABLE

    # steady state 진입 시점: Trigger 4가 처음 1이 되는 k
    if is_steady.any():
        first_steady_k = int(k[is_steady][0])
        # 첫 steady 이후 다시 0으로 표시된 샘플 수 (steady 진입 후 재불안정)
        unstable_after_steady = int((is_unstable & (k > first_steady_k)).sum())
        first_steady_sec = first_steady_k * SAMPLE_DT_SEC
    else:
        first_steady_k = np.nan
        unstable_after_steady = np.nan
        first_steady_sec = np.nan

    summary_rows.append(
        {
            "run_id": run_id,
            COL_TENSION_SET: int(g[COL_TENSION_SET].iloc[0]),
            COL_REWINDER_SET: int(g[COL_REWINDER_SET].iloc[0]),
            COL_SPEED_SET: int(g[COL_SPEED_SET].iloc[0]),
            "n": len(g),
            "n_label0": int(is_unstable.sum()),
            "n_label1": int(is_steady.sum()),
            "n_label2": int((label == LABEL_AMBIGUOUS).sum()),
            "first_steady_k": first_steady_k,
            "first_steady_sec": first_steady_sec,
            "unstable_after_steady": unstable_after_steady,
            # steady 구간 e: 설정값 대비 편향과 변동 크기
            "e_mean_steady": e[is_steady].mean() if is_steady.any() else np.nan,
            "e_std_steady": e[is_steady].std(ddof=1) if is_steady.sum() > 1 else np.nan,
            # 미안정 구간 e: 설정값 대비 최대 처짐 / 최대 overshoot
            "e_min_unstable": e[is_unstable].min() if is_unstable.any() else np.nan,
            "e_max_unstable": e[is_unstable].max() if is_unstable.any() else np.nan,
        }
    )

df_summary = pd.DataFrame(summary_rows).sort_values(
    [COL_TENSION_SET, COL_REWINDER_SET, COL_SPEED_SET], kind="mergesort"
).reset_index(drop=True)

df_all = df_all.drop(columns=["_e"])

print(f"[블록 3] file_summary shape = {df_summary.shape}")
print(f"         n 범위 = {df_summary['n'].min()}–{df_summary['n'].max()}")
print(f"         라벨 0이 없는 run = {(df_summary['n_label0'] == 0).sum()}개")
print(f"         라벨 2가 없는 run = {(df_summary['n_label2'] == 0).sum()}개")
_re = df_summary.loc[df_summary["unstable_after_steady"] > 0, "run_id"].tolist()
print(f"         steady 진입 후 재불안정 run = {_re}")
print(f"         요약 산출 완료 ({time.time() - t_block:.1f} s)")


# =============================================================================
# 블록 4. 샘플 간격 요약 출력 (등간격 0.109 s 가정의 근거)
# =============================================================================
df_interval = pd.DataFrame(interval_rows).sort_values("run_id").reset_index(drop=True)
print("[블록 4] 샘플 간격 (210 run 전체)")
print(f"         run별 평균 dt: {df_interval['dt_mean_sec'].min():.4f}–{df_interval['dt_mean_sec'].max():.4f} s "
      f"(전체 평균 {df_interval['dt_mean_sec'].mean():.4f} s)")
print(f"         전체 최솟값 dt = {df_interval['dt_min_sec'].min():.4f} s, "
      f"최댓값 dt = {df_interval['dt_max_sec'].max():.4f} s")


# =============================================================================
# 블록 5. 저장 (csv). 하이퍼파라미터도 결과와 함께 저장
# =============================================================================
t_block = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_all.to_csv(OUT_DIR / "merged_data.csv", index=False, encoding="utf-8")
df_summary.to_csv(OUT_DIR / "01_file_summary.csv", index=False, encoding="utf-8")
df_interval.to_csv(OUT_DIR / "01_sample_interval.csv", index=False, encoding="utf-8")

df_hp = pd.DataFrame(
    [
        {"name": "RANDOM_SEED", "value": RANDOM_SEED},
        {"name": "DATA_DIR", "value": str(DATA_DIR)},
        {"name": "OUT_DIR", "value": str(OUT_DIR)},
        {"name": "TENSION_SETTINGS", "value": str(TENSION_SETTINGS)},
        {"name": "SAMPLE_DT_SEC", "value": SAMPLE_DT_SEC},
        {"name": "LABEL_UNSTABLE", "value": LABEL_UNSTABLE},
        {"name": "LABEL_STEADY", "value": LABEL_STEADY},
        {"name": "LABEL_AMBIGUOUS", "value": LABEL_AMBIGUOUS},
    ]
)
df_hp.to_csv(OUT_DIR / "hyperparameters.csv", index=False, encoding="utf-8")

print(f"[블록 5] 저장 완료 → {OUT_DIR} ({time.time() - t_block:.1f} s)")
print(f"[전체] 실행 시간 {time.time() - t_start:.1f} s")
