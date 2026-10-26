"""
00_diagnostics.py

역할: 특징 분석 설계를 확정하기 전에, 설계가 성립하는지 확인하는 진단.
      아래 5개 질문에 수치로 답하고 결과를 csv와 plot으로 남긴다.

  D1. 두 상태(transient vs steady)의 차이 크기가 운전 조건에 따라 변하는가
      → 외삽 방향(상향/하향)에 따라 난이도가 달라지는지 확인
  D2. window 분류 과제가 포화되어 있는가 (조건 외삽 포함)
      → 수준 특징 mean|e| 하나의 미학습 조건 성능과 '남은 개선 여지'
  D3. steady 진입 시점(onset) 추정 과제에는 여유가 있는가 (조건 외삽 포함)
  D4. 샘플 간격 불균일이 상태에 따라 치우쳐 있는가 (등간격 가정의 타당성)
  D5. 롤 회전 주파수 f_rot = v/(pi*D)의 범위와 조건 의존성 (차수축 정규화 근거)

입력: outputs/01_merge/merged_data.csv, outputs/01_merge/01_file_summary.csv,
      data/data_tensionSetting_{3..7}.zip (D5에서 Rewinder roll diameter 읽기)
출력: outputs/00_diagnostics/ (csv 7개, plot 3종 × PNG/PDF, hyperparameters.csv)

라벨 사용: 본 파일은 `cost_modification_ideas.md` 0.1절의 '사전 특징 분석' 단계에 해당하므로
          `Trigger 4`를 사용한다. 제안 방법(CoST 변형) 학습에는 사용하지 않는다.
"""

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Hyperparameter (사용자 지정 영역)
# =============================================================================
RANDOM_SEED = 42                        # 랜덤 시드 (재현성)

IN_DIR = Path("outputs/01_merge")       # 01_merge.py 결과 위치
DATA_DIR = Path("data")                 # 원본 zip 위치 (D5)
OUT_DIR = Path("outputs/00_diagnostics")

WINDOW_L = 64                           # window 길이 (샘플). feature_analysis_plan 2절 주 분석값
WINDOW_HOP = 32                         # hop = L/2
WINDOW_L_LIST = [32, 64, 128]           # D2에서 포화 여부를 길이별로 비교
SAMPLE_DT_SEC = 0.109                   # 샘플 간격 가정값 (s). 근사값 — 임의의 값
THR_GRID_N = 400                        # 임계값 탐색 격자 수 — 임의의 값
ONSET_THR_QMAX = 0.60                   # onset 임계값 탐색 상한 분위수 — 임의의 값

# 원본 컬럼명 (변경 금지)
COL_LABEL = "Trigger 4"
COL_TENSION = "Film Tension #1 (kg)"
COL_TENSION_SET = "Film-Tension Setting Value (kg)"
COL_SPEED_SET = "Web Speed setting value (mm/sec)"
COL_SPEED_CUR = "Web Current Speed (mm/sec)"
COL_DIAM = "Rewinder roll diameter (mm)"

# 조건 외삽 분할 정의 (사용자 승인: 장력 양방향, 속도 양방향, 이중 외삽)
SPLITS = [
    # (ID, 설명, 축, 학습 조건, 검증 조건)
    ("X0a", "tension interpolation (control)", "T", [3, 5, 7], [4, 6]),
    ("X0b", "speed interpolation (control)", "V", [50, 100, 150, 200], [75, 125, 175]),
    ("X1", "tension extrapolation upward", "T", [3, 4, 5], [6, 7]),
    ("X2", "tension extrapolation downward", "T", [5, 6, 7], [3, 4]),
    ("X3", "speed extrapolation upward", "V", [50, 75, 100], [175, 200]),
    ("X4", "speed extrapolation downward", "V", [125, 150, 175, 200], [50, 75]),
]
# 이중 외삽 X6은 두 축을 동시에 제한하므로 별도 처리
SPLIT_X6_TRAIN = {"T": [3, 4, 5], "V": [50, 75, 100]}
SPLIT_X6_TEST = {"T": [6, 7], "V": [175, 200]}

# plot 공통 설정 (논문 스타일). 특징(방법)마다 고정 색
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 100, "savefig.bbox": "tight",
})
FEATURE_COLOR = {
    "mean_abs_e": "#1f77b4",        # 수준 특징 (level)
    "neg_min_e": "#4c9ac4",         # 수준 특징
    "mean_abs_e_ratio": "#7fb8d6",  # 수준 특징 (설정값 정규화)
    "std_e": "#d62728",             # 형태 특징 (shape)
    "ptp_centered": "#e8743b",      # 형태 특징
    "mad_diff": "#ff9896",          # 형태 특징
    "slope": "#9467bd",             # 형태 특징
}
LEVEL_FEATS = ["mean_abs_e", "neg_min_e", "mean_abs_e_ratio"]
SHAPE_FEATS = ["std_e", "ptp_centered", "mad_diff", "slope"]
ALL_FEATS = LEVEL_FEATS + SHAPE_FEATS

np.random.seed(RANDOM_SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 함수 (반복 사용되는 것만)
# =============================================================================
def save_fig(fig, stem):
    """plot을 PNG(300 dpi)와 PDF로 함께 저장. 입력: figure, 파일명 어간. 출력: 없음(파일 저장)"""
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def best_threshold(values, labels):
    """균형정확도를 최대화하는 임계값을 분위수 격자에서 탐색.
    입력: 특징값 1차원 배열, 이진 라벨(1 = transient). 출력: 임계값 (float)"""
    grid = np.quantile(values, np.linspace(0.001, 0.999, THR_GRID_N))
    scores = [balanced_accuracy_score(labels, (values >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def onset_index(stat, thr):
    """causal 통계가 처음 thr 아래로 내려간 index (= steady 진입 판정 시점).
    입력: causal 통계 배열, 임계값. 출력: index (없으면 마지막 index)"""
    below = np.where(stat < thr)[0]
    return int(below[0]) if len(below) else len(stat) - 1


t_start = time.time()

# =============================================================================
# 블록 0. 데이터 적재 및 잔차 e 계산
#   e = Film Tension #1 (kg) - Film-Tension Setting Value (kg)
# =============================================================================
t_block = time.time()
d = pd.read_csv(IN_DIR / "merged_data.csv")
d["e"] = d[COL_TENSION] - d[COL_TENSION_SET]
summary = pd.read_csv(IN_DIR / "01_file_summary.csv")
print(f"[블록 0] merged_data {d.shape}, file_summary {summary.shape} ({time.time() - t_block:.1f} s)")


# =============================================================================
# 블록 1. D1 — 상태 차이 크기의 조건 의존성
#   분리 크기 = |transient 구간 e 최솟값| / steady 구간 e 표준편차 (신호 대 잡음 비 성격)
#   외삽을 '쉬운 쪽'으로 하는지 '어려운 쪽'으로 하는지 판정하는 근거
# =============================================================================
t_block = time.time()
sub = summary.dropna(subset=["e_min_unstable"]).copy()
sub["separation"] = sub["e_min_unstable"].abs() / sub["e_std_steady"]

d1_rows = []
for axis, col in [("T", COL_TENSION_SET), ("V", COL_SPEED_SET)]:
    for val, g in sub.groupby(col):
        gs = summary[summary[col] == val]
        d1_rows.append({
            "axis": axis, "condition_value": val, "n_run": len(gs),
            "sag_median_kg": g["e_min_unstable"].median(),
            "steady_std_median_kg": gs["e_std_steady"].median(),
            "separation_median": g["separation"].median(),
            "first_steady_k_median": gs["first_steady_k"].median(),
        })
d1 = pd.DataFrame(d1_rows)
d1.to_csv(OUT_DIR / "D1_condition_dependence.csv", index=False, encoding="utf-8")
print(f"[블록 1] D1 조건 의존성")
for axis in ["T", "V"]:
    r = d1[d1.axis == axis]
    print(f"         {axis}: 분리 크기 " +
          "  ".join(f"{int(v)}:{s:.2f}" for v, s in zip(r.condition_value, r.separation_median)))
print(f"         (단조 증가하면 고장력·고속이 '쉬운 쪽') ({time.time() - t_block:.1f} s)")

# plot 1: 조건별 분리 크기
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, axis, xlabel in [(axes[0], "T", "Film-tension setting (kgf)"),
                         (axes[1], "V", "Web speed setting (mm/s)")]:
    r = d1[d1.axis == axis]
    ax.plot(r.condition_value, r.separation_median, "o-", color="#d62728",
            label="Separation = |sag| / steady std")
    ax.set_xlabel(xlabel); ax.set_ylabel("State separation (-)")
    ax.grid(True, alpha=0.4); ax.legend()
axes[0].set_title("(a) By tension setting"); axes[1].set_title("(b) By speed setting")
save_fig(fig, "D1_state_separation_by_condition")


# =============================================================================
# 블록 2. window 생성 (라벨이 모두 동일한 window만 사용 — 문서 2절 규칙)
#   수준(level) 특징과 형태(shape) 특징을 분리해 저장
# =============================================================================
t_block = time.time()
win_rows = []
for L in WINDOW_L_LIST:
    hop = L // 2
    for rid, g in d.groupby("run_id", sort=False):
        g = g.sort_values("k")
        e = g["e"].to_numpy(); lab = g[COL_LABEL].to_numpy()
        t_set, v_set = int(g[COL_TENSION_SET].iloc[0]), int(g[COL_SPEED_SET].iloc[0])
        for st in range(0, len(e) - L + 1, hop):
            wl = lab[st:st + L]
            if (wl == wl[0]).all() and wl[0] in (0, 1):
                w = e[st:st + L]
                wc = w - w.mean()                     # 평균 제거 (수준 정보 차단)
                win_rows.append({
                    "L": L, "run_id": rid, "T": t_set, "V": v_set,
                    "y": int(wl[0] == 0),             # 1 = transient (양성 = 라벨 0)
                    "mean_abs_e": np.abs(w).mean(),
                    "neg_min_e": -w.min(),
                    "mean_abs_e_ratio": np.abs(w).mean() / t_set,
                    "std_e": w.std(ddof=1),
                    "ptp_centered": wc.max() - wc.min(),
                    "mad_diff": np.abs(np.diff(w)).mean(),
                    "slope": np.polyfit(np.arange(L), w, 1)[0],
                })
WIN = pd.DataFrame(win_rows)
WIN.to_csv(OUT_DIR / "D2_window_features.csv", index=False, encoding="utf-8")

cnt_rows = []
for L in WINDOW_L_LIST:
    wl = WIN[WIN.L == L]
    cnt_rows.append({"L": L, "hop": L // 2,
                     "n_window_transient": int(wl.y.sum()),
                     "n_window_steady": int((1 - wl.y).sum()),
                     "imbalance_ratio": (1 - wl.y).sum() / max(wl.y.sum(), 1),
                     "n_run_with_transient": wl[wl.y == 1].run_id.nunique(),
                     "window_sec": L * SAMPLE_DT_SEC})
d2cnt = pd.DataFrame(cnt_rows)
d2cnt.to_csv(OUT_DIR / "D2_window_counts.csv", index=False, encoding="utf-8")
print(f"[블록 2] window 생성 ({time.time() - t_block:.1f} s)")
print(d2cnt.to_string(index=False))

# 조건 격자(장력 × 속도)별 transient window 수
grid = WIN[(WIN.L == WINDOW_L) & (WIN.y == 1)].pivot_table(
    index="T", columns="V", values="y", aggfunc="sum").fillna(0).astype(int)
grid.to_csv(OUT_DIR / "D2_transient_window_grid.csv", encoding="utf-8")
print(f"         L={WINDOW_L} 조건 격자별 transient window 수 (빈 셀 = 분석 불가):")
print(grid.to_string())


# =============================================================================
# 블록 3. D2 — 조건 외삽에서 window 분류의 포화 여부
#   임계값은 학습 조건에서만 정하고 검증 조건에 그대로 적용 (보정 전이 확인)
#   '남은 개선 여지'를 window 개수 단위로 환산해 측정 해상도를 함께 보고
# =============================================================================
t_block = time.time()
W = WIN[WIN.L == WINDOW_L]
d2_rows = []

split_specs = [(sid, desc, axis, tr, te) for sid, desc, axis, tr, te in SPLITS]
for sid, desc, axis, tr_vals, te_vals in split_specs:
    tr, te = W[W[axis].isin(tr_vals)], W[W[axis].isin(te_vals)]
    for f in ALL_FEATS:
        thr = best_threshold(tr[f].to_numpy(), tr.y.to_numpy())
        pred_te = (te[f].to_numpy() >= thr).astype(int)
        bacc_tr = balanced_accuracy_score(tr.y, (tr[f].to_numpy() >= thr).astype(int))
        bacc_te = balanced_accuracy_score(te.y, pred_te)
        cm = confusion_matrix(te.y, pred_te, labels=[1, 0])
        n_pos = int(te.y.sum())
        d2_rows.append({
            "split_id": sid, "split_desc": desc, "axis": axis,
            "train_values": str(tr_vals), "test_values": str(te_vals),
            "feature": f, "group": "level" if f in LEVEL_FEATS else "shape",
            "n_train_transient": int(tr.y.sum()), "n_test_transient": n_pos,
            "n_test_steady": int((1 - te.y).sum()),
            "auroc_train": roc_auc_score(tr.y, tr[f]),
            "auroc_test": roc_auc_score(te.y, te[f]),
            "balacc_train": bacc_tr, "balacc_test": bacc_te,
            "balacc_gap": bacc_te - bacc_tr,
            "threshold": thr,
            "test_transient_correct": int(cm[0, 0]), "test_transient_wrong": int(cm[0, 1]),
            "test_steady_correct": int(cm[1, 1]), "test_steady_wrong": int(cm[1, 0]),
            # 측정 해상도: transient window 1개 = 균형정확도 몇 %p
            "resolution_pp_per_window": 0.5 / n_pos * 100,
            "headroom_pp": (1 - bacc_te) * 100,
            "headroom_in_windows": (1 - bacc_te) / (0.5 / n_pos),
        })
    # 다변량 로지스틱 회귀 (수준 포함 / 수준 제외)
    for nm, cols in [("LR_all", ALL_FEATS), ("LR_shape_only", SHAPE_FEATS),
                     ("LR_level_only", LEVEL_FEATS)]:
        sc = StandardScaler().fit(tr[cols])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(tr[cols]), tr.y)
        bacc_tr = balanced_accuracy_score(tr.y, clf.predict(sc.transform(tr[cols])))
        bacc_te = balanced_accuracy_score(te.y, clf.predict(sc.transform(te[cols])))
        d2_rows.append({
            "split_id": sid, "split_desc": desc, "axis": axis,
            "train_values": str(tr_vals), "test_values": str(te_vals),
            "feature": nm, "group": "multivariate",
            "n_train_transient": int(tr.y.sum()), "n_test_transient": int(te.y.sum()),
            "n_test_steady": int((1 - te.y).sum()),
            "auroc_train": roc_auc_score(tr.y, clf.decision_function(sc.transform(tr[cols]))),
            "auroc_test": roc_auc_score(te.y, clf.decision_function(sc.transform(te[cols]))),
            "balacc_train": bacc_tr, "balacc_test": bacc_te, "balacc_gap": bacc_te - bacc_tr,
            "threshold": np.nan,
            "test_transient_correct": np.nan, "test_transient_wrong": np.nan,
            "test_steady_correct": np.nan, "test_steady_wrong": np.nan,
            "resolution_pp_per_window": 0.5 / int(te.y.sum()) * 100,
            "headroom_pp": (1 - bacc_te) * 100,
            "headroom_in_windows": (1 - bacc_te) / (0.5 / int(te.y.sum())),
        })

# 이중 외삽 X6
tr6 = W[W["T"].isin(SPLIT_X6_TRAIN["T"]) & W["V"].isin(SPLIT_X6_TRAIN["V"])]
te6 = W[W["T"].isin(SPLIT_X6_TEST["T"]) & W["V"].isin(SPLIT_X6_TEST["V"])]
for f in ALL_FEATS:
    thr = best_threshold(tr6[f].to_numpy(), tr6.y.to_numpy())
    pred_te = (te6[f].to_numpy() >= thr).astype(int)
    bacc_te = balanced_accuracy_score(te6.y, pred_te)
    cm = confusion_matrix(te6.y, pred_te, labels=[1, 0]); n_pos = int(te6.y.sum())
    d2_rows.append({
        "split_id": "X6", "split_desc": "double extrapolation (tension & speed)", "axis": "T+V",
        "train_values": str(SPLIT_X6_TRAIN), "test_values": str(SPLIT_X6_TEST),
        "feature": f, "group": "level" if f in LEVEL_FEATS else "shape",
        "n_train_transient": int(tr6.y.sum()), "n_test_transient": n_pos,
        "n_test_steady": int((1 - te6.y).sum()),
        "auroc_train": roc_auc_score(tr6.y, tr6[f]), "auroc_test": roc_auc_score(te6.y, te6[f]),
        "balacc_train": balanced_accuracy_score(tr6.y, (tr6[f].to_numpy() >= thr).astype(int)),
        "balacc_test": bacc_te,
        "balacc_gap": bacc_te - balanced_accuracy_score(tr6.y, (tr6[f].to_numpy() >= thr).astype(int)),
        "threshold": thr,
        "test_transient_correct": int(cm[0, 0]), "test_transient_wrong": int(cm[0, 1]),
        "test_steady_correct": int(cm[1, 1]), "test_steady_wrong": int(cm[1, 0]),
        "resolution_pp_per_window": 0.5 / n_pos * 100, "headroom_pp": (1 - bacc_te) * 100,
        "headroom_in_windows": (1 - bacc_te) / (0.5 / n_pos),
    })

d2 = pd.DataFrame(d2_rows)
d2.to_csv(OUT_DIR / "D2_window_classification_extrapolation.csv", index=False, encoding="utf-8")
print(f"[블록 3] D2 window 분류 조건 외삽 ({time.time() - t_block:.1f} s)")
show = d2[d2.feature.isin(["mean_abs_e", "std_e", "LR_shape_only"])]
for sid in ["X0a", "X1", "X2", "X3", "X4", "X6"]:
    r = show[show.split_id == sid]
    if len(r) == 0:
        continue
    print(f"         [{sid}] " + " | ".join(
        f"{row.feature}: AUROC {row.auroc_test:.4f}, BalAcc {row.balacc_test:.4f}"
        for row in r.itertuples()))
lv = d2[(d2.feature == "mean_abs_e") & (d2.split_id.isin(["X1", "X3"]))]
for row in lv.itertuples():
    print(f"         [{row.split_id}] mean|e| 한 줄: transient {row.test_transient_correct}/{row.n_test_transient} 맞춤, "
          f"steady 오경보 {row.test_steady_wrong}/{row.n_test_steady}, "
          f"남은 여지 {row.headroom_pp:.2f}%p = {row.headroom_in_windows:.1f} window분 "
          f"(1 window = {row.resolution_pp_per_window:.2f}%p)")

# plot 2: 조건별 window 분류 AUROC — 수준 특징은 평평, 형태 특징은 속도 의존
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, axis, xlabel in [(axes[0], "T", "Film-tension setting (kgf)"),
                         (axes[1], "V", "Web speed setting (mm/s)")]:
    for f in ["mean_abs_e", "std_e", "mad_diff"]:
        xs, ys = [], []
        for val, g in W.groupby(axis):
            if g.y.nunique() < 2:
                continue
            xs.append(val); ys.append(roc_auc_score(g.y, g[f]))
        ax.plot(xs, ys, "o-", color=FEATURE_COLOR[f], label=f)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel("AUROC within condition (-)")
    ax.set_ylim(0.3, 1.03); ax.grid(True, alpha=0.4); ax.legend(loc="lower right")
axes[0].set_title("(a) By tension setting"); axes[1].set_title("(b) By speed setting")
save_fig(fig, "D2_auroc_within_condition")


# =============================================================================
# 블록 4. D3 — onset(steady 진입 시점) 추정 오차와 조건 외삽 격차
#   causal rolling 통계 2종으로 판정: 수준 특징 mean|e|, 형태 특징 std(e)
#   임계값은 학습 조건에서 중앙절대오차를 최소화하도록 정하고 검증 조건에 그대로 적용
# =============================================================================
t_block = time.time()
s_idx = summary.set_index("run_id")
run_stats = []
for rid, g in d.groupby("run_id", sort=False):
    g = g.sort_values("k")
    truth = s_idx.loc[rid, "first_steady_k"]
    if np.isnan(truth):
        continue
    ser = pd.Series(g["e"].to_numpy())
    run_stats.append({
        "run_id": rid, "T": int(g[COL_TENSION_SET].iloc[0]), "V": int(g[COL_SPEED_SET].iloc[0]),
        "truth": int(truth),
        "level": ser.abs().rolling(WINDOW_L).mean().to_numpy(),
        "shape": ser.rolling(WINDOW_L).std(ddof=1).to_numpy(),
    })

d3_rows = []
onset_specs = [(sid, desc, axis, tr, te) for sid, desc, axis, tr, te in SPLITS]
for sid, desc, axis, tr_vals, te_vals in onset_specs + [("X6", "double extrapolation (tension & speed)", "T+V", None, None)]:
    if sid == "X6":
        tr_runs = [r for r in run_stats if r["T"] in SPLIT_X6_TRAIN["T"] and r["V"] in SPLIT_X6_TRAIN["V"]]
        te_runs = [r for r in run_stats if r["T"] in SPLIT_X6_TEST["T"] and r["V"] in SPLIT_X6_TEST["V"]]
    else:
        tr_runs = [r for r in run_stats if r[axis] in tr_vals]
        te_runs = [r for r in run_stats if r[axis] in te_vals]
    for feat in ["level", "shape"]:
        pool = np.concatenate([r[feat][~np.isnan(r[feat])] for r in tr_runs])
        cand = np.quantile(pool, np.linspace(0.005, ONSET_THR_QMAX, 200))
        err_by_thr = [np.median([abs(onset_index(r[feat], t) - r["truth"]) for r in tr_runs]) for t in cand]
        thr = float(cand[int(np.argmin(err_by_thr))])
        e_tr = np.median([abs(onset_index(r[feat], thr) - r["truth"]) for r in tr_runs])
        e_te = np.median([abs(onset_index(r[feat], thr) - r["truth"]) for r in te_runs])
        signed_te = np.median([onset_index(r[feat], thr) - r["truth"] for r in te_runs])
        d3_rows.append({
            "split_id": sid, "split_desc": desc, "axis": axis,
            "feature_group": feat, "threshold": thr,
            "n_train_run": len(tr_runs), "n_test_run": len(te_runs),
            "mae_train_sample": e_tr, "mae_test_sample": e_te,
            "mae_train_sec": e_tr * SAMPLE_DT_SEC, "mae_test_sec": e_te * SAMPLE_DT_SEC,
            "extrapolation_gap_sample": e_te - e_tr,
            "signed_err_test_sample": signed_te,
            "signed_err_test_sec": signed_te * SAMPLE_DT_SEC,
        })
d3 = pd.DataFrame(d3_rows)
d3.to_csv(OUT_DIR / "D3_onset_extrapolation.csv", index=False, encoding="utf-8")
print(f"[블록 4] D3 onset 추정 ({time.time() - t_block:.1f} s)")
for row in d3.itertuples():
    print(f"         [{row.split_id}] {row.feature_group:5s} | run {row.n_train_run:3d}->{row.n_test_run:3d} | "
          f"중앙절대오차 {row.mae_train_sample:6.1f}->{row.mae_test_sample:6.1f} 샘플 "
          f"({row.mae_test_sec:5.1f} s) | 외삽 격차 {row.extrapolation_gap_sample:+6.1f} | "
          f"부호오차 {row.signed_err_test_sample:+6.1f}")

# plot 3: onset 외삽 격차
fig, ax = plt.subplots(figsize=(8.2, 3.6))
order = ["X0a", "X0b", "X1", "X2", "X3", "X4", "X6"]
x = np.arange(len(order)); width = 0.38
for i, (feat, color, lbl) in enumerate([("level", "#1f77b4", "Level feature: mean|e|"),
                                        ("shape", "#d62728", "Shape feature: std(e)")]):
    vals = [d3[(d3.split_id == s) & (d3.feature_group == feat)].mae_test_sec.values[0] for s in order]
    ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=lbl)
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_xlabel("Condition split (X0 = interpolation control)")
ax.set_ylabel("Onset detection error (s)")
ax.grid(True, axis="y", alpha=0.4); ax.legend()
ax.set_title("Onset detection error under condition extrapolation")
save_fig(fig, "D3_onset_error_by_split")


# =============================================================================
# 블록 5. D4 — 샘플 간격의 상태별 분포 (등간격 가정의 상태 의존 편향)
#           D5 — 롤 회전 주파수 f_rot = v / (pi * D) 범위 (차수축 정규화 근거)
# =============================================================================
t_block = time.time()
dt_rows, frot_rows = [], []
for tension in sorted(d[COL_TENSION_SET].unique()):
    zf = zipfile.ZipFile(DATA_DIR / f"data_tensionSetting_{int(tension)}.zip")
    for name in sorted(n for n in zf.namelist() if n.lower().endswith(".xlsx")):
        df = pd.read_excel(zf.open(name), usecols=["Date", COL_LABEL, COL_DIAM, COL_SPEED_CUR, COL_SPEED_SET])
        dt = pd.to_datetime(df["Date"]).diff().dt.total_seconds().to_numpy()[1:]
        lab = df[COL_LABEL].to_numpy()[1:]
        dt_rows.append(pd.DataFrame({"dt_sec": dt, "label": lab}))
        D = df[COL_DIAM].to_numpy(); v = df[COL_SPEED_CUR].to_numpy()
        frot = v / (np.pi * np.where(D > 0, D, np.nan))
        lab_all = df[COL_LABEL].to_numpy()
        frot_rows.append({
            "run_id": f"T{int(df[COL_TENSION_SET].iloc[0]) if COL_TENSION_SET in df else int(tension)}",
            "V_set": int(df[COL_SPEED_SET].iloc[0]),
            "D_min_mm": np.nanmin(D), "D_max_mm": np.nanmax(D), "D_n_unique": len(np.unique(D)),
            "frot_min_hz": np.nanmin(frot), "frot_max_hz": np.nanmax(frot),
            "frot_median_hz": np.nanmedian(frot),
            "frot_median_transient_hz": np.nanmedian(frot[lab_all == 0]) if (lab_all == 0).any() else np.nan,
            "frot_median_steady_hz": np.nanmedian(frot[lab_all == 1]) if (lab_all == 1).any() else np.nan,
        })
DT = pd.concat(dt_rows, ignore_index=True)
d4 = DT.groupby("label").dt_sec.agg(
    n="size", mean="mean", median="median", std="std", min="min", max="max").reset_index()
d4["frac_gt_0.15s_pct"] = [(DT.loc[DT.label == l, "dt_sec"] > 0.15).mean() * 100 for l in d4.label]
d4.to_csv(OUT_DIR / "D4_sample_interval_by_state.csv", index=False, encoding="utf-8")
print(f"[블록 5] D4 샘플 간격 (상태별)")
print(d4.to_string(index=False))

FROT = pd.DataFrame(frot_rows)
d5 = FROT.groupby("V_set").agg(
    n_run=("V_set", "size"), frot_median_hz=("frot_median_hz", "median"),
    frot_min_hz=("frot_min_hz", "min"), frot_max_hz=("frot_max_hz", "max"),
    frot_transient_hz=("frot_median_transient_hz", "median"),
    frot_steady_hz=("frot_median_steady_hz", "median"),
    D_min_mm=("D_min_mm", "min"), D_max_mm=("D_max_mm", "max")).reset_index()
d5["nyquist_hz"] = 1.0 / SAMPLE_DT_SEC / 2
d5.to_csv(OUT_DIR / "D5_rotation_frequency.csv", index=False, encoding="utf-8")
print(f"         D5 회전 주파수 f_rot = v/(pi*D)")
print(d5.round(4).to_string(index=False))
print(f"         속도 {int(d5.V_set.min())}->{int(d5.V_set.max())} mm/s 에서 f_rot 중앙값 "
      f"{d5.frot_median_hz.min():.4f}->{d5.frot_median_hz.max():.4f} Hz "
      f"({d5.frot_median_hz.max()/d5.frot_median_hz.min():.1f}배 이동), "
      f"Nyquist {1/SAMPLE_DT_SEC/2:.2f} Hz ({time.time() - t_block:.1f} s)")


# =============================================================================
# 블록 6. hyperparameter 저장
# =============================================================================
pd.DataFrame([
    {"name": "RANDOM_SEED", "value": RANDOM_SEED},
    {"name": "WINDOW_L", "value": WINDOW_L},
    {"name": "WINDOW_HOP", "value": WINDOW_HOP},
    {"name": "WINDOW_L_LIST", "value": str(WINDOW_L_LIST)},
    {"name": "SAMPLE_DT_SEC", "value": SAMPLE_DT_SEC},
    {"name": "THR_GRID_N", "value": THR_GRID_N},
    {"name": "ONSET_THR_QMAX", "value": ONSET_THR_QMAX},
    {"name": "SPLITS", "value": str(SPLITS)},
    {"name": "SPLIT_X6_TRAIN", "value": str(SPLIT_X6_TRAIN)},
    {"name": "SPLIT_X6_TEST", "value": str(SPLIT_X6_TEST)},
]).to_csv(OUT_DIR / "hyperparameters.csv", index=False, encoding="utf-8")

print(f"[전체] 실행 시간 {time.time() - t_start:.1f} s → {OUT_DIR}")
