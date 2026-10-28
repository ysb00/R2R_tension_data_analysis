# =============================================================================
# code/03_screening.py : feature 선별 — D(식별력) · T(전이성) · 증분 기여도 · 진폭 정규화
#                        (docs/research_plan.md 4.2 / 4.3 / 4.5 절)
#
# 역할 : 02_features.py 가 만든 시점별 feature 를 **B 와 같은 평가 후단**에 넣어
#        "어떤 feature 를 CoST 변형의 입력으로 넣을지" 를 수치로 고른다.
#
#   A-2  D = 조건 내 식별력. run 안에서 `Trigger 4` 0 vs 1 의 AUROC (ambiguous 제외) 를 run 평균.
#        T = 조건 간 전이성. 학습 fold 에서 표준화+2군집 GMM 적합 -> 검증 fold AUPRC(ns).
#        -> D·T 산점도가 본 분석의 핵심 그림이다.
#   A-3  F0(movstd, |movslope|) 기준점에 후보를 1개씩 더해 ΔD·ΔT 를 보고, ΔT > 0 인 것만
#        전진 선택으로 키운다. **1단계는 승계 검증** — F0 수치가 B 의 MovStd-Slope 와 맞아야 한다.
#   A-5  진폭 정규화 3종(e / e÷설정값 / e÷MAD(run)). feature 는 다시 산출하지 않고
#        feature_scaling.csv 의 배율 지수로 환산한다 (run 내 상수배이므로 정확히 같다).
#
# 외부 코드 승계 (code/external_versions.csv)
#   evaluation.py · splits.py 는 B 저장소에서 **수정 없이 복사**한 것이다. 본 저장소의 경로만
#   다르므로 파일을 고치지 않고 아래 '어댑터' 블록에서 모듈 상수를 런타임에 바꿔 쓴다.
#
# [라벨 사용 경계]
#   라벨은 evaluation.py 안(_labels / score_fold)에서만 읽힌다. feature 의 **정의**에는
#   라벨이 들어가지 않고(02_features.py 는 라벨을 읽지 않는다) **순위와 선택**에만 쓴다.
#   전진 선택 기준은 장력축 5-fold CV 로 둔다 — 속도축 final_test 로 고르면 B 의 최종
#   테스트 집합에 선택이 새어 들어가므로, final_test 는 **고른 뒤 확인용**으로만 본다.
#
# 입력 : outputs/02_features/{계열}/features_W{W}.csv.gz, _cross/feature_scaling.csv, _cross/run_scales.csv
#        outputs/01_merge/merged_data.csv (e 와 조건 컬럼. `Trigger 4` 는 읽지 않는다)
#        outputs/00_folds/folds.csv, folds_c.csv (B 의 fold 배정 그대로)
# 출력 : outputs/03_screening/  — **feature 계열마다 폴더를 나눈다** (02_features 와 같은 이름)
#          P0_baseline/ P1_spectral/ P2_phase/ P3_nonstationarity/ P4_physical/ P5_additional/
#              D_summary.csv        식별력 D
#              DT_single.csv        단일 feature 의 D·T
#              increment.csv        F0 에 1개씩 더했을 때의 ΔD·ΔT (P0_baseline 에는 없다 — 기준점이라 후보가 아니다)
#              state_mean_diff.csv  상태별 feature 평균과 차이 부호
#              DT_scatter_W144.*    그 계열을 강조한 D·T 산점도 (나머지 계열은 회색)
#            P0_baseline/ 에는 승계 검증 결과(inheritance_check.csv, F0_W*_fold/run_metrics.csv)가 추가된다
#          _cross/                  계열을 넘나드는 결과 (계열 하나에 속하지 않는 것)
#              전체 D·T 산점도, 전체 증분 막대그림, 전진 선택, 창 민감도, A-5 정규화,
#              feature 상관행렬·heatmap, 전 계열 통합 csv, hyperparameters.csv
# 실행 : python code/03_screening.py
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev
import splits

# --- 어댑터 : 복사한 외부 코드의 경로 상수만 런타임에 바꾼다 (원본 파일 미수정) -----------
ev.MERGED_PATH = 'outputs/01_merge/merged_data.csv'
splits.SPEC['tension']['path'] = 'outputs/00_folds/folds.csv'
splits.SPEC['c']['path'] = 'outputs/00_folds/folds_c.csv'

# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                         # 난수 시드 (seed 반복의 시작값). B 승계값
N_SEEDS = 5                      # GMM 후단의 seed 반복 횟수. B 승계값
WINDOWS = [64, 144, 256]         # 이동창 길이 (samples). 승인 #1
W_MAIN = 144                     # 주 창 길이 — 증분 기여도·전진 선택·A-5 는 이 창에서 수행한다

# 평가할 fold : 계획 4.1절 그대로 (장력축 leave-one-level-out 5 fold + 속도축 final_test)
#   'tension' 5 fold = 미학습 장력 수준 외삽, 'c'/final_test = 저속 학습 -> 고속 테스트
FOLD_SPEC = [('tension', None), ('c', ['final_test'])]      # None = 그 분할의 모든 fold
T_PRIMARY = 'pooled_auprc'       # T 주 지표 = AUPRC(non-steady). 승인 #9 (B 승계)
T_SELECT_GROUP = 'tension'       # 전진 선택 기준 fold 묶음 (final_test 누출 방지). 결정
FORWARD_MIN_GAIN = 0.005         # 전진 선택 중단 기준: ΔT 가 이 값 미만이면 중단. **임의의 값** (승인 #10)
FORWARD_MAX_DIM = 6              # 전진 선택 최대 차원 수. **임의의 값** (승인 #10)
                                 # 후단이 diag GMM 이라 차원이 늘면 과적합한다
SLOPE_ABS = True                 # F0 의 기울기는 크기만 쓴다 (B `01_stat_ssd.py:280` 과 동일해야 한다)

FEAT_DIR = 'outputs/02_features'
OUT_DIR = 'outputs/03_screening'
CROSS_DIR = '_cross'             # 계열에 속하지 않는 산출물을 모으는 폴더 이름 (02_features 와 같다)
T1 = 'Film Tension #1 (kg)'      # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS
ID_COLS = ['run_id', 'k', TS, RS, VS]

F0_COLS = ['movstd', 'movslope']                 # 기준점 (B 의 MovStd-Slope 와 같은 2차원)

# 계열별 고정 색 (논문 스타일 — 그림 전체에서 계열마다 같은 색).
# evaluation.py 의 METHOD_STYLE 과 같은 Okabe-Ito 계열 색을 쓴다.
# P0(기준점)은 검정, 제안 4계열은 유채색, P5(추가)는 회색 — 제안분이 눈에 먼저 들어오게 한다
FAMILY_STYLE = {'P0': ('#000000', 'o'), 'P1': ('#d55e00', 'v'), 'P2': ('#662aa2', 'X'),
                'P3': ('#009e73', '^'), 'P4': ('#e69f00', 'P'), 'P5': ('#8c8c8c', 's')}

# B 의 통계 baseline 수치 (승계 검증 대조표). 출처: B outputs/03_baselines/01_stat_ssd/ ·
# outputs/04_speed_extrap/01_stat_ssd/ 의 *_fold_metrics.csv (커밋 caaac99)
# 창 64 = MovStd-Slope-fixed, 창 144 = MovStd-Slope-fixed-rule1b
B_REFERENCE = {(64, 'tension'): 0.7181, (144, 'tension'): 0.6081,
               (64, 'final_test'): 0.8467, (144, 'final_test'): 0.5414}

# =============================================================================
# 함수
# =============================================================================


def eval_z(z, method, folds):
    # 역할 : 시점별 표현 z 를 B 의 평가 후단에 넣어 fold×seed 지표를 만든다.
    #        학습 fold 로 표준화+2군집 GMM 적합 -> |e 평균| 작은 군집 = steady -> 검증 fold 채점.
    #        채점은 전부 evaluation.score_fold 안에서 일어난다 (라벨 접근 지점 단일화).
    # 입력 : z (n_point, d) df 행 순서와 같은 표현 / method 기록용 이름
    #        folds [(split, fold, is_train, is_val)] 리스트
    # 출력 : (fold_metrics DataFrame, run_metrics DataFrame)
    fm, rm = [], []
    for split, fold, itr, iva in folds:
        for seed in range(SEED, SEED + N_SEEDS):
            model, sidx, _ = ev.fit_steady_gmm(z[itr], E[itr], seed)
            pred, score = ev.predict_steady(model, sidx, z[iva])
            r, f = ev.score_fold(ID.loc[iva], pred, score, method, fold, seed)
            f['split'], r['split'] = split, split
            fm.append(f), rm.append(r)
    fm, rm = pd.concat(fm, ignore_index=True), pd.concat(rm, ignore_index=True)
    fm['group'] = [splits.fold_group(f) or 'tension' for f in fm.fold]      # A 분할은 묶음이 하나다
    rm['group'] = [splits.fold_group(f) or 'tension' for f in rm.fold]
    return fm, rm


def t_scalar(fold_metrics, group):
    # 역할 : fold×seed 지표표에서 fold 묶음 하나의 T 대표값(주 지표 평균)을 뽑는다.
    # 입력 : fold_metrics eval_z 결과 / group 'tension' 또는 'final_test'
    # 출력 : float (fold 와 seed 전체 평균)
    return float(fold_metrics.loc[fold_metrics.group == group, T_PRIMARY].mean())


def build_z(cols, divisor=None):
    # 역할 : feature 이름 목록으로 표현 행렬을 만든다. F0 의 기울기는 크기만 쓰고(B 와 동일),
    #        divisor 가 주어지면 A-5 의 진폭 정규화를 배율 지수로 환산해 적용한다.
    #        (e -> e/c, c 는 run 내 상수 => 지수 1 feature 는 1/c 배, 지수 0 feature 는 불변)
    # 입력 : cols feature 이름 리스트 / divisor (n_point,) run별 분모 또는 None
    # 출력 : z (n_point, len(cols))
    z = np.column_stack([np.abs(FEAT[c].to_numpy()) if (SLOPE_ABS and c == 'movslope')
                         else FEAT[c].to_numpy() for c in cols])
    if divisor is not None:
        expo = np.array([SCALE_EXPO[c] for c in cols], dtype=float)
        z = z / divisor[:, None] ** expo[None, :]
    return z


def save_plot(fig, stem):
    # 역할 : 그림을 PNG(300 dpi) + PDF 로 저장한다. 같은 이름은 덮어쓰지 않는다.
    # 입력 : fig matplotlib figure / stem 확장자 없는 경로
    # 출력 : 없음
    for ext in ['png', 'pdf']:
        p = f'{stem}.{ext}'
        assert not os.path.exists(p), f'이미 그림 파일이 있다 (덮어쓰기 금지): {p}'
        fig.savefig(p, dpi=ev.DPI)
    plt.close(fig)
    print(f'[save] plot -> {stem}.png / .pdf')


def split_by_family(frame, filename):
    # 역할 : feature 단위 결과표를 계열 폴더로 나눠 저장하고, 통합본은 _cross/ 에 둔다.
    #        결과물을 "어떤 방식(feature 계열)에 대한 결과인지" 로 구분하기 위한 단일 지점이다.
    # 입력 : frame  'family' 컬럼을 가진 결과표 / filename 저장할 파일 이름 (예: 'D_summary.csv')
    # 출력 : 없음 (계열 폴더마다 1개 + _cross/ 에 통합본 1개)
    for fam, d in FAMILY_DIR.items():
        g = frame[frame.family == fam]
        if len(g):                                           # F0 는 증분 후보가 아니라 행이 없을 수 있다
            g.to_csv(f'{OUT_DIR}/{d}/{filename}', index=False)
    frame.to_csv(f'{OUT_DIR}/{CROSS_DIR}/{filename.replace(".csv", "_all.csv")}', index=False)


def load_features(w):
    # 역할 : 계열 폴더에 나뉘어 저장된 창 w 의 feature 를 한 표로 합친다.
    #        모든 계열 파일이 같은 (run_id, k) 순서로 저장돼 있으므로 이어 붙이기만 하면 된다.
    # 입력 : w 창 길이 (samples)
    # 출력 : DataFrame [run_id, k, 16 feature]
    parts = [pd.read_csv(f'{FEAT_DIR}/{d}/features_W{w}.csv.gz') for d in FAMILY_DIR.values()]
    for p in parts[1:]:
        assert (p.run_id.to_numpy() == parts[0].run_id.to_numpy()).all() and \
               (p.k.to_numpy() == parts[0].k.to_numpy()).all(), f'창 {w}: 계열 파일 간 행 순서가 다르다'
    return pd.concat([parts[0]] + [p.drop(columns=['run_id', 'k']) for p in parts[1:]], axis=1)


# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 마스크
# =============================================================================
t_all = time.time()
t0 = time.time()
base = pd.read_csv(ev.MERGED_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
base = base.sort_values(['run_id', 'k']).reset_index(drop=True)
E = (base[T1] - base[TS]).to_numpy()                       # 잔차 e — GMM 의 steady 군집 지정에 쓰인다
ID = base[ID_COLS]

spec = pd.read_csv(f'{FEAT_DIR}/{CROSS_DIR}/feature_scaling.csv')
SCALE_EXPO = dict(zip(spec.feature, spec.amplitude_scale_exponent))
FAMILY = dict(zip(spec.feature, spec.family))
FAMILY_DIR = dict(zip(spec.family, spec.family_dir))        # 02_features 가 정한 계열 폴더 이름을 그대로 쓴다
ALL_FEATURES = spec.feature.tolist()
CANDIDATES = [c for c in ALL_FEATURES if c not in F0_COLS]  # F0 에 더해 볼 후보 14개
for d in list(FAMILY_DIR.values()) + [CROSS_DIR]:
    os.makedirs(f'{OUT_DIR}/{d}', exist_ok=True)

run_scales = pd.read_csv(f'{FEAT_DIR}/{CROSS_DIR}/run_scales.csv').set_index('run_id')

FOLDS = {}
for split, keep in FOLD_SPEC:
    sp = splits.load(split)
    for fold in (sp.folds if keep is None else keep):
        tr_run, va_run = sp.run_role(fold)
        itr = base.run_id.map(dict(zip(sp.run_ids, tr_run))).to_numpy(dtype=bool)
        iva = base.run_id.map(dict(zip(sp.run_ids, va_run))).to_numpy(dtype=bool)
        FOLDS[(split, fold)] = (itr, iva)
        print(f'[Step 1] 분할 {split:7s} fold {str(fold):10s} 학습 {itr.sum():6d} / 검증 {iva.sum():6d} 시점')
FOLD_LIST = [(s, f, itr, iva) for (s, f), (itr, iva) in FOLDS.items()]
print(f'[Step 1] rows = {len(base)}, feature {len(ALL_FEATURES)}개, fold {len(FOLD_LIST)}개 '
      f'({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 창별 D(식별력) · 상태별 feature 평균 (A-2 전반 · 계획 4.2절의 가설 방향 재확인)
#
#   D 는 분류기가 없는 지표다 — feature 값을 그대로 non-steady 점수로 넣고 run 내 AUROC 를 읽는다.
#   AUROC 는 부호에 무관하므로 방향을 모르는 feature 는 1-AUROC 가 될 수 있다.
#   그래서 raw 와 방향 보정값(0.5 기준 반사) 둘 다 기록한다.
#   `ev.score_fold` 는 pred 기반 지표도 함께 계산하지만 여기서는 auroc 만 쓴다 (pred 는 더미).
# =============================================================================
d_rows, state_rows, corr_store = [], [], {}
FEAT_BY_W = {}
for w in WINDOWS:
    t0 = time.time()
    FEAT = load_features(w)
    assert (FEAT.run_id.to_numpy() == base.run_id.to_numpy()).all() and \
           (FEAT.k.to_numpy() == base.k.to_numpy()).all(), f'창 {w}: feature 행 순서가 merged 와 다르다'
    FEAT_BY_W[w] = FEAT

    # --- D : feature 1개를 점수로 쓴 run별 AUROC ---
    dummy = np.zeros(len(FEAT), dtype=int)
    for c in ALL_FEATURES:
        x = np.abs(FEAT[c].to_numpy()) if (SLOPE_ABS and c == 'movslope') else FEAT[c].to_numpy()
        rm, _ = ev.score_fold(ID, dummy, x, c, 'all', SEED)
        d_rows.append({'W': w, 'feature': c, 'family': FAMILY[c],
                       'D_raw': rm.auroc.mean(), 'D_raw_std': rm.auroc.std(),
                       'D': rm.auroc.map(lambda v: max(v, 1 - v)).mean(),
                       'n_runs': int(rm.auroc.notna().sum())})

    # --- 상태별 평균 (채점 블록 — 라벨을 evaluation._labels 로 읽는다) ---
    lab = FEAT[['run_id', 'k']].merge(ev._labels(), on=['run_id', 'k'], how='left')[ev.LABEL_COL].to_numpy()
    m0, m1 = lab == 0, lab == 1                                # 0 = transient, 1 = steady (2 는 제외)
    for c in ALL_FEATURES:
        x = FEAT[c].to_numpy()
        state_rows.append({'W': w, 'feature': c, 'family': FAMILY[c],
                           'mean_transient': x[m0].mean(), 'mean_steady': x[m1].mean(),
                           'diff_transient_minus_steady': x[m0].mean() - x[m1].mean(),
                           'sign': '+' if x[m0].mean() > x[m1].mean() else '-'})
    corr_store[w] = FEAT[ALL_FEATURES].corr()
    print(f'[Step 2] 창 {w:3d}: D 산출 완료 ({time.time() - t0:.1f} s)')

d_metric = pd.DataFrame(d_rows)
state_mean = pd.DataFrame(state_rows)
split_by_family(d_metric, 'D_summary.csv')
split_by_family(state_mean, 'state_mean_diff.csv')
for w in WINDOWS:                                            # 상관행렬은 계열을 넘나드는 표다
    corr_store[w].to_csv(f'{OUT_DIR}/{CROSS_DIR}/corr_matrix_W{w}.csv')
print('[Step 2] D 상위 5개 (창 144):')
print(d_metric[d_metric.W == W_MAIN].nlargest(5, 'D')[['feature', 'family', 'D', 'D_raw']].to_string(index=False))

# =============================================================================
# Step 3. 승계 검증 — F0 수치가 B 의 MovStd-Slope 와 맞는지 먼저 확인한다 (계획 4.3절 1단계)
#         맞지 않으면 데이터 처리가 B 와 다르다는 뜻이므로 다음 단계로 가지 않는다.
# =============================================================================
f0_t, inherit = {}, []
for w in WINDOWS:
    t0 = time.time()
    FEAT = FEAT_BY_W[w]
    fm, rm = eval_z(build_z(F0_COLS), f'F0-W{w}', FOLD_LIST)
    fm.to_csv(f'{OUT_DIR}/{FAMILY_DIR["P0"]}/F0_W{w}_fold_metrics.csv', index=False)
    rm.to_csv(f'{OUT_DIR}/{FAMILY_DIR["P0"]}/F0_W{w}_run_metrics.csv', index=False)
    for grp in ['tension', 'final_test']:
        got = t_scalar(fm, grp)
        f0_t[(w, grp)] = got
        ref = B_REFERENCE.get((w, grp))
        inherit.append({'W': w, 'group': grp, 'A_T': got, 'B_T': ref,
                        'diff': None if ref is None else got - ref,
                        'B_method': {64: 'MovStd-Slope-fixed', 144: 'MovStd-Slope-fixed-rule1b'}.get(w)})
    print(f'[Step 3] 창 {w:3d} F0: T(tension) {f0_t[(w, "tension")]:.4f}, '
          f'T(final_test) {f0_t[(w, "final_test")]:.4f} ({time.time() - t0:.1f} s)')
inherit = pd.DataFrame(inherit)
inherit.to_csv(f'{OUT_DIR}/{FAMILY_DIR["P0"]}/inheritance_check.csv', index=False)
print('[Step 3] 승계 검증 (A 의 F0 vs B 의 MovStd-Slope, AUPRC(ns)):')
print(inherit.to_string(index=False))

# =============================================================================
# Step 4. A-3 증분 기여도 — F0 에 후보를 1개씩 더해 ΔT 를 본다 (주 창 W_MAIN)
# =============================================================================
t0 = time.time()
FEAT = FEAT_BY_W[W_MAIN]
inc = []
for c in CANDIDATES:
    fm, _ = eval_z(build_z(F0_COLS + [c]), f'F0+{c}', FOLD_LIST)
    row = {'W': W_MAIN, 'feature': c, 'family': FAMILY[c], 'dim': 3}
    for grp in ['tension', 'final_test']:
        row[f'T_{grp}'] = t_scalar(fm, grp)
        row[f'dT_{grp}'] = row[f'T_{grp}'] - f0_t[(W_MAIN, grp)]
    row['D'] = float(d_metric.query('W == @W_MAIN and feature == @c').D.iloc[0])
    inc.append(row)
    print(f'[Step 4] F0+{c:18s} T(tension) {row["T_tension"]:.4f} (dT {row["dT_tension"]:+.4f}), '
          f'T(final_test) {row["T_final_test"]:.4f} (dT {row["dT_final_test"]:+.4f})')
inc = pd.DataFrame(inc).sort_values('dT_tension', ascending=False)
split_by_family(inc, 'increment.csv')
print(f'[Step 4] 증분 기여도 완료 ({time.time() - t0:.1f} s). dT_tension > 0 인 feature '
      f'{int((inc.dT_tension > 0).sum())}/{len(inc)}개')

# =============================================================================
# Step 5. A-2 단일 feature T — D·T 산점도용 (feature 1차원만 후단에 넣는다, 주 창)
# =============================================================================
t0 = time.time()
single = []
for c in ALL_FEATURES:
    fm, _ = eval_z(build_z([c]), f'single-{c}', FOLD_LIST)
    row = {'W': W_MAIN, 'feature': c, 'family': FAMILY[c], 'dim': 1,
           'D': float(d_metric.query('W == @W_MAIN and feature == @c').D.iloc[0])}
    for grp in ['tension', 'final_test']:
        row[f'T_{grp}'] = t_scalar(fm, grp)
        row[f'entry_mae_{grp}'] = float(fm.loc[fm.group == grp, 'runmae_entry_err_s'].mean())
        row[f'contam_{grp}'] = float(fm.loc[fm.group == grp, 'pooled_contamination_rate'].mean())
    single.append(row)
    print(f'[Step 5] single {c:18s} D {row["D"]:.4f}, T(tension) {row["T_tension"]:.4f}, '
          f'T(final_test) {row["T_final_test"]:.4f}')
single = pd.DataFrame(single)
split_by_family(single, 'DT_single.csv')
print(f'[Step 5] 단일 feature D·T 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 6. A-3 전진 선택 — ΔT > 0 인 후보만 대상으로 집합을 키운다 (기준: 장력축 CV)
# =============================================================================
t0 = time.time()
pool = inc.loc[inc.dT_tension > 0, 'feature'].tolist()
chosen, steps = list(F0_COLS), []
cur = f0_t[(W_MAIN, T_SELECT_GROUP)]
steps.append({'step': 0, 'added': '+'.join(F0_COLS), 'dim': len(chosen),
              'T_tension': f0_t[(W_MAIN, 'tension')], 'T_final_test': f0_t[(W_MAIN, 'final_test')],
              'dT': np.nan, 'n_candidates': len(pool), 'stop': False})
while len(chosen) < FORWARD_MAX_DIM and pool:
    trials = []
    for c in pool:
        fm, _ = eval_z(build_z(chosen + [c]), f'fwd{len(chosen)}+{c}', FOLD_LIST)
        trials.append((t_scalar(fm, T_SELECT_GROUP), t_scalar(fm, 'final_test'), c))
    trials.sort(reverse=True)
    best_t, best_ft, best_c = trials[0]
    gain = best_t - cur
    stop = gain < FORWARD_MIN_GAIN
    steps.append({'step': len(chosen) - 1, 'added': best_c, 'dim': len(chosen) + 1,
                  'T_tension': best_t, 'T_final_test': best_ft, 'dT': gain,
                  'n_candidates': len(pool), 'stop': stop})
    print(f'[Step 6] 차원 {len(chosen) + 1}: 최선 후보 {best_c} -> T {best_t:.4f} (dT {gain:+.4f})'
          + (' => 중단 기준 미달, 선택 종료' if stop else ''))
    if stop:
        break
    chosen.append(best_c)
    pool.remove(best_c)
    cur = best_t
steps = pd.DataFrame(steps)
steps.to_csv(f'{OUT_DIR}/{CROSS_DIR}/forward_selection.csv', index=False)
print(f'[Step 6] 선택된 집합 ({len(chosen)}차원): {chosen} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 7. 창 민감도 — F0 와 선택 집합을 창 3종에서 비교한다 (계획 한계 L2 분리)
# =============================================================================
t0 = time.time()
sens = []
for w in WINDOWS:
    FEAT = FEAT_BY_W[w]
    for name, cols in [('F0', F0_COLS), ('selected', chosen)]:
        fm, _ = eval_z(build_z(cols), f'{name}-W{w}', FOLD_LIST)
        row = {'W': w, 'feature_set': name, 'dim': len(cols), 'cols': '+'.join(cols)}
        for grp in ['tension', 'final_test']:
            row[f'T_{grp}'] = t_scalar(fm, grp)
            row[f'entry_mae_{grp}'] = float(fm.loc[fm.group == grp, 'runmae_entry_err_s'].mean())
            row[f'contam_{grp}'] = float(fm.loc[fm.group == grp, 'pooled_contamination_rate'].mean())
        sens.append(row)
        print(f'[Step 7] 창 {w:3d} {name:8s}: T(tension) {row["T_tension"]:.4f}, '
              f'T(final_test) {row["T_final_test"]:.4f}, 진입 MAE(final_test) {row["entry_mae_final_test"]:.2f} s')
sens = pd.DataFrame(sens)
sens.to_csv(f'{OUT_DIR}/{CROSS_DIR}/window_sensitivity.csv', index=False)
print(f'[Step 7] 창 민감도 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 8. A-5 진폭 정규화 3종 — 좌표를 바꿔 F0 와 선택 집합을 다시 평가한다 (주 창)
#         e ÷ MAD(run) 은 run 전체를 보므로 causal 하지 않다 (한계로 기록한다)
# =============================================================================
t0 = time.time()
FEAT = FEAT_BY_W[W_MAIN]
div_set = base.run_id.map(run_scales[TS]).to_numpy(dtype=float)
div_mad = base.run_id.map(run_scales['mad_run']).to_numpy(dtype=float)
a5 = []
for coord, divisor, causal in [('e', None, True), ('e_div_setting', div_set, True),
                               ('e_div_mad_run', div_mad, False)]:
    for name, cols in [('F0', F0_COLS), ('selected', chosen)]:
        fm, _ = eval_z(build_z(cols, divisor), f'{name}-{coord}', FOLD_LIST)
        row = {'coordinate': coord, 'causal': causal, 'feature_set': name, 'W': W_MAIN,
               'dim': len(cols), 'cols': '+'.join(cols)}
        for grp in ['tension', 'final_test']:
            row[f'T_{grp}'] = t_scalar(fm, grp)
            row[f'entry_mae_{grp}'] = float(fm.loc[fm.group == grp, 'runmae_entry_err_s'].mean())
        a5.append(row)
        print(f'[Step 8] {coord:14s} {name:8s}: T(tension) {row["T_tension"]:.4f}, '
              f'T(final_test) {row["T_final_test"]:.4f}')
a5 = pd.DataFrame(a5)
a5.to_csv(f'{OUT_DIR}/{CROSS_DIR}/a5_normalization.csv', index=False)
print(f'[Step 8] A-5 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 9. 그림 (논문 스타일 — 축·범례 영어, PNG 300 dpi + PDF, 계열마다 고정 색)
# =============================================================================
t0 = time.time()

# --- 9.1 D·T 산점도 (본 분석의 핵심 그림). fold 묶음 2개를 나란히 둔다 ---
#     전체판은 _cross/ 에, 계열마다 그 계열만 강조하고 나머지를 회색으로 둔 판은 계열 폴더에 둔다
for focus in [None] + list(FAMILY_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(ev.PAGE_WIDTH, 3.1), layout='constrained', sharex=True)
    for ax, grp, ttl in zip(axes, ['tension', 'final_test'],
                            ['Unseen tension level (5-fold)', 'Unseen high speed (final test)']):
        for fam, g in single.groupby('family'):
            col, mk = FAMILY_STYLE[fam]
            dim = focus is not None and fam != focus          # 강조 대상이 아니면 회색으로 뒤에 깐다
            ax.scatter(g.D, g[f'T_{grp}'], c='#c9c9c9' if dim else col, marker=mk,
                       s=20 if dim else 30, label=fam, zorder=3 if dim else 4)
        for _, r in single.iterrows():
            dim = focus is not None and r.family != focus
            ax.annotate(r.feature, (r.D, r[f'T_{grp}']), textcoords='offset points', xytext=(3, 3),
                        fontsize=ev.FONT_SIZE - 4, color='#bbbbbb' if dim else '#333333')
        ax.axhline(f0_t[(W_MAIN, grp)], color='#000000', linewidth=0.8, linestyle='--', zorder=2)
        ax.set_title(ttl)
        ax.set_xlabel('D: within-run AUROC (orientation-corrected)')
    axes[0].set_ylabel(f'T: AUPRC (non-steady), W = {W_MAIN}')
    # 범례는 그림 밖에 둔다 — 안쪽에 두면 왼쪽 아래 점(movkurt 등)을 가린다
    fig.legend(*axes[0].get_legend_handles_labels(), title='Family', ncols=8,
               loc='outside lower center', frameon=False)
    save_plot(fig, f'{OUT_DIR}/{CROSS_DIR if focus is None else FAMILY_DIR[focus]}/DT_scatter_W{W_MAIN}')

# --- 9.2 증분 기여도 (F0 + 1 feature 의 ΔT) ---
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH, 2.9), layout='constrained')
x = np.arange(len(inc))
for off, grp, col, hatch, lbl in [(-0.2, 'tension', '#0072b2', None, 'Unseen tension level'),
                                  (0.2, 'final_test', '#d55e00', '//', 'Unseen high speed')]:
    ax.bar(x + off, inc[f'dT_{grp}'], width=0.4, color=col, hatch=hatch,
           edgecolor='#333333', linewidth=0.5, label=lbl)
ax.axhline(0, color='#000000', linewidth=0.8)
ax.set_xticks(x, inc.feature, rotation=60, ha='right', fontsize=ev.FONT_SIZE - 2)
ax.set_ylabel(r'$\Delta$T: AUPRC gain over F0')
ax.legend()
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/increment_W{W_MAIN}')

# --- 9.3 전진 선택 경로 ---
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH * 0.6, 2.6), layout='constrained')
for grp, col, mk, lbl in [('tension', '#0072b2', 'o', 'Unseen tension level'),
                          ('final_test', '#d55e00', 's', 'Unseen high speed')]:
    ax.plot(steps.dim, steps[f'T_{grp}'], color=col, marker=mk, linewidth=1.2, label=lbl)
ax.set_xlabel('Number of feature dimensions')
ax.set_ylabel('T: AUPRC (non-steady)')
ax.set_xticks(steps.dim)
ax.legend()
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/forward_selection')

# --- 9.4 feature 상관행렬 (중복 명시) ---
c = corr_store[W_MAIN].loc[ALL_FEATURES, ALL_FEATURES]
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH, ev.PAGE_WIDTH * 0.92), layout='constrained')
im = ax.imshow(c.to_numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(ALL_FEATURES)), ALL_FEATURES, rotation=60, ha='right', fontsize=ev.FONT_SIZE - 2)
ax.set_yticks(range(len(ALL_FEATURES)), ALL_FEATURES, fontsize=ev.FONT_SIZE - 2)
ax.grid(False)
for i in range(len(ALL_FEATURES)):
    for j in range(len(ALL_FEATURES)):
        v = c.to_numpy()[i, j]
        ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center',
                fontsize=ev.FONT_SIZE - 4, color='#ffffff' if abs(v) > 0.6 else '#0b0b0b')
fig.colorbar(im, ax=ax, label=f'Pearson correlation (W = {W_MAIN})', fraction=0.045, pad=0.02)
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/corr_heatmap_W{W_MAIN}')

# --- 9.5 창 민감도 ---
fig, axes = plt.subplots(1, 2, figsize=(ev.PAGE_WIDTH, 2.6), layout='constrained', sharex=True)
for ax, grp, ttl in zip(axes, ['tension', 'final_test'],
                        ['Unseen tension level', 'Unseen high speed']):
    for name, col, mk in [('F0', '#000000', 'o'), ('selected', '#009e73', 's')]:
        g = sens[sens.feature_set == name]
        ax.plot(g.W, g[f'T_{grp}'], color=col, marker=mk, linewidth=1.2, label=name)
    ax.set_xticks(WINDOWS)
    ax.set_xlabel('Window length (samples)')
    ax.set_title(ttl)
axes[0].set_ylabel('T: AUPRC (non-steady)')
axes[1].legend(title='Feature set')
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/window_sensitivity')

# --- 9.6 A-5 진폭 정규화 ---
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH * 0.75, 2.7), layout='constrained')
coords = ['e', 'e_div_setting', 'e_div_mad_run']
xx = np.arange(len(coords))
BARS = [('F0', 'tension', '#000000'), ('F0', 'final_test', '#8c8c8c'),
        ('selected', 'tension', '#009e73'), ('selected', 'final_test', '#8ed6bd')]
for i, (name, grp, col) in enumerate(BARS):
    g = a5[a5.feature_set == name].set_index('coordinate').loc[coords]
    ax.bar(xx + (i - 1.5) * 0.2, g[f'T_{grp}'], width=0.2, color=col,
           edgecolor='#333333', linewidth=0.5,
           label=f'{name}, {"unseen tension" if grp == "tension" else "unseen high speed"}')
ax.set_xticks(xx, ['e', r'e $\div$ setting', r'e $\div$ MAD(run)'])
ax.set_ylabel('T: AUPRC (non-steady)')
fig.legend(*ax.get_legend_handles_labels(), ncols=2, loc='outside lower center', frameon=False)
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/a5_normalization')
print(f'[Step 9] 그림 6종 저장 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 10. hyperparameter 기록
# =============================================================================
cfg = {**ev.EVAL_CONFIG, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOWS': WINDOWS, 'W_MAIN': W_MAIN,
       'FOLD_SPEC': str(FOLD_SPEC), 'T_PRIMARY': T_PRIMARY, 'T_SELECT_GROUP': T_SELECT_GROUP,
       'FORWARD_MIN_GAIN': FORWARD_MIN_GAIN, 'FORWARD_MAX_DIM': FORWARD_MAX_DIM, 'SLOPE_ABS': SLOPE_ABS,
       'F0_COLS': '+'.join(F0_COLS), 'SELECTED_COLS': '+'.join(chosen),
       'LABEL_USE': 'feature 순위·선택에만 사용 (정의에는 미사용). 선택 기준은 장력축 CV',
       'B_COMMIT': 'caaac99317e2a9b04039a369f3bb8059d4440bd7'}
pd.DataFrame({'name': list(cfg), 'value': [str(v) for v in cfg.values()]}).to_csv(
    f'{OUT_DIR}/{CROSS_DIR}/hyperparameters.csv', index=False)
print(f'[done] feature 선별 완료 (총 {time.time() - t_all:.1f} s)')
