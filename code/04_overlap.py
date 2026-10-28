# =============================================================================
# code/04_overlap.py : A-4 상태 정보와 조건 정보의 중첩 (docs/research_plan.md 4.4절)
#
# 역할 : CoST 변형 설계의 분기점을 수치로 정한다.
#        transient 의 처짐이 속도와 상관되면(저속 -0.23 -> 고속 -0.46 kg) **조건 정보를 지우면
#        상태 정보도 같이 지워진다.** 그래서 feature 마다 두 가지를 재고 산점도로 비교한다.
#          1) 상태 식별력 : `Trigger 4` 0 vs 1 (ambiguous 제외) AUROC  <- 03_screening 의 D 를 쓴다
#          2) 조건 식별력 : 장력(T) 또는 속도(V) 수준을 맞히는 정도
#        -> 상태는 맞히고 조건은 못 맞히는 feature 가 이상적이다.
#           그런 feature 가 없으면 조건 불변화(적대적 제거) 대신 조건 조건화(FiLM 등 설정값 주입)로 간다.
#
# 조건 식별력의 측정 방식 (2종을 함께 낸다)
#   (a) 평균 쌍별 AUROC — 수준 쌍마다 AUROC 를 구해 평균한다. 학습이 필요 없는 순위 통계이고
#       우연 수준이 0.5 여서 **상태 AUROC 와 같은 잣대로 산점도에 놓을 수 있다**. 주 지표.
#   (b) 다중분류 정확도 — 계획 4.4절이 적은 "정확도". feature 1차원을 분위 구간으로 나눈
#       비모수 Bayes 분류기로 재고, run 단위로 나눈 2-fold 로 채점한다 (같은 run 이 학습·검증에
#       동시에 들어가지 않게). 우연 수준은 T 1/5 = 0.200, V 1/7 = 0.143 이라 (a) 와 잣대가 다르다.
#
# 보조 표 — 계획 4.4절의 전제 자체를 확인한다 : 속도 수준마다 '상태 평균차'(transient - steady)를
#   내서, 상태 신호의 크기가 조건에 따라 달라지는지(= 두 정보가 겹치는지)를 직접 본다.
#
# [라벨 사용 경계] 라벨은 evaluation._labels() 로만 읽는다 (채점·분석 블록). feature 정의에는 안 쓴다.
#
# 입력 : outputs/02_features/{계열}/features_W{W}.csv.gz, _cross/feature_scaling.csv
#        outputs/03_screening/_cross/D_summary_all.csv  (상태 식별력 D — 같은 코드로 산출된 값을 재사용)
#        outputs/01_merge/merged_data.csv (조건 컬럼)
# 출력 : outputs/04_overlap/  — **feature 계열마다 폴더를 나눈다** (02·03 과 같은 이름)
#          P0_baseline/ ~ P5_additional/  그 계열의 state_vs_condition.csv, state_diff_by_condition.csv,
#                                     state_diff_variation.csv, 그 계열을 강조한 산점도
#          _cross/                    전 계열 통합 csv, 전체 산점도, 속도별 상태 평균차 그림,
#                                     hyperparameters.csv
# 실행 : python code/04_overlap.py
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev

# --- 어댑터 : 복사한 외부 코드의 경로 상수만 런타임에 바꾼다 (원본 파일 미수정) -----------
ev.MERGED_PATH = 'outputs/01_merge/merged_data.csv'

# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                         # 난수 시드 (이 단계는 결정론적이지만 규칙상 1개로 고정해 둔다)
W_MAIN = 144                     # 주 창 길이 (samples). 03_screening 과 같은 값
N_BINS = 20                      # (b) Bayes 분류기의 분위 구간 수. **임의의 값**
N_COND_FOLDS = 2                 # (b) 의 run 단위 교차검증 fold 수. **임의의 값**

FEAT_DIR = 'outputs/02_features'
SCREEN_DIR = 'outputs/03_screening'
OUT_DIR = 'outputs/04_overlap'
CROSS_DIR = '_cross'             # 계열에 속하지 않는 산출물을 모으는 폴더 이름 (02·03 과 같다)
TS, RS, VS = ev.TS, ev.RS, ev.VS
COND_AXES = [('tension', TS), ('speed', VS)]        # 조건 축 2개 (계획 0.1절: 둘 다 검토)

# 계열별 고정 색 — 03_screening 과 같은 배정 (P0 검정 / 제안 4계열 유채색 / P5 회색)
FAMILY_STYLE = {'P0': ('#000000', 'o'), 'P1': ('#d55e00', 'v'), 'P2': ('#662aa2', 'X'),
                'P3': ('#009e73', '^'), 'P4': ('#e69f00', 'P'), 'P5': ('#8c8c8c', 's')}

# =============================================================================
# 함수
# =============================================================================


def mean_pairwise_auroc(x, lv):
    # 역할 : 수준이 3개 이상인 조건 축에 대해 '수준 쌍마다 AUROC' 를 구해 평균한다.
    #        AUROC 는 부호에 무관하므로 쌍마다 0.5 기준으로 방향을 보정한다(max(a, 1-a)).
    #        학습이 필요 없고 우연 수준이 0.5 라 상태 AUROC 와 같은 잣대가 된다.
    # 입력 : x (n,) feature 값 / lv (n,) 조건 수준 값
    # 출력 : (평균 쌍별 AUROC, 최대 쌍별 AUROC)
    levels = np.unique(lv)
    vals = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            m = (lv == levels[i]) | (lv == levels[j])
            a = roc_auc_score((lv[m] == levels[j]).astype(int), x[m])
            vals.append(max(a, 1 - a))
    return float(np.mean(vals)), float(np.max(vals))


def binned_bayes_accuracy(x, lv, run, n_bins, n_folds):
    # 역할 : feature 1차원을 분위 구간으로 나눈 비모수 Bayes 분류기로 조건 수준 다중분류 정확도를 낸다.
    #        학습 fold 에서 구간 경계와 P(수준 | 구간) 을 추정하고 검증 fold 에서 argmax 로 예측한다.
    #        fold 는 **run 단위**로 나눈다 — 같은 run 의 시점이 학습과 검증에 함께 들어가면
    #        조건 식별력이 과대평가된다. 수준마다 run 을 정렬해 번갈아 배정하므로 수준 비율이 유지된다.
    # 입력 : x (n,) feature 값 / lv (n,) 조건 수준 / run (n,) run_id / n_bins 구간 수 / n_folds fold 수
    # 출력 : (정확도, 우연 수준 정확도)
    runs = pd.DataFrame({'run': run, 'lv': lv}).drop_duplicates().sort_values(['lv', 'run'])
    runs['fold'] = runs.groupby('lv').cumcount() % n_folds
    fold_of = run.map(dict(zip(runs.run, runs.fold))).to_numpy()
    levels = np.unique(lv)
    hit = 0
    for f in range(n_folds):
        tr, va = fold_of != f, fold_of == f
        edges = np.unique(np.quantile(x[tr], np.linspace(0, 1, n_bins + 1)[1:-1]))
        b_tr, b_va = np.digitize(x[tr], edges), np.digitize(x[va], edges)
        tab = np.zeros((len(edges) + 1, len(levels)))
        for li, l in enumerate(levels):                       # 구간 x 수준 빈도 -> P(수준 | 구간)
            tab[:, li] = np.bincount(b_tr[lv[tr] == l], minlength=len(edges) + 1)
        pred = levels[tab.argmax(axis=1)][b_va]
        hit += (pred == lv[va]).sum()
    return float(hit / len(x)), float(1.0 / len(levels))


def split_by_family(frame, filename):
    # 역할 : feature 단위 결과표를 계열 폴더로 나눠 저장하고, 통합본은 _cross/ 에 둔다.
    #        결과물을 "어떤 방식(feature 계열)에 대한 결과인지" 로 구분하기 위한 단일 지점이다.
    # 입력 : frame 'family' 컬럼을 가진 결과표 / filename 저장할 파일 이름
    # 출력 : 없음 (계열 폴더마다 1개 + _cross/ 에 통합본 1개)
    for fam, d in FAMILY_DIR.items():
        g = frame[frame.family == fam]
        if len(g):
            g.to_csv(f'{OUT_DIR}/{d}/{filename}', index=False)
    frame.to_csv(f'{OUT_DIR}/{CROSS_DIR}/{filename.replace(".csv", "_all.csv")}', index=False)


def load_features(w):
    # 역할 : 계열 폴더에 나뉘어 저장된 창 w 의 feature 를 한 표로 합친다 (행 순서는 모든 계열에서 같다).
    # 입력 : w 창 길이 (samples)
    # 출력 : DataFrame [run_id, k, 16 feature]
    parts = [pd.read_csv(f'{FEAT_DIR}/{d}/features_W{w}.csv.gz') for d in FAMILY_DIR.values()]
    for q in parts[1:]:
        assert (q.run_id.to_numpy() == parts[0].run_id.to_numpy()).all(), '계열 파일 간 행 순서가 다르다'
    return pd.concat([parts[0]] + [q.drop(columns=['run_id', 'k']) for q in parts[1:]], axis=1)


def save_plot(fig, stem):
    # 역할 : 그림을 PNG(300 dpi) + PDF 로 저장한다. 같은 이름은 덮어쓰지 않는다.
    for ext in ['png', 'pdf']:
        p = f'{stem}.{ext}'
        assert not os.path.exists(p), f'이미 그림 파일이 있다 (덮어쓰기 금지): {p}'
        fig.savefig(p, dpi=ev.DPI)
    plt.close(fig)
    print(f'[save] plot -> {stem}.png / .pdf')


# =============================================================================
# Step 1. 데이터 로드 + 라벨 결합 (채점·분석 블록 — 라벨은 evaluation._labels 로만 읽는다)
# =============================================================================
t_all = time.time()
t0 = time.time()
spec = pd.read_csv(f'{FEAT_DIR}/{CROSS_DIR}/feature_scaling.csv')
FEATURES, FAMILY = spec.feature.tolist(), dict(zip(spec.feature, spec.family))
FAMILY_DIR = dict(zip(spec.family, spec.family_dir))       # 02_features 가 정한 계열 폴더 이름을 그대로 쓴다
for d in list(FAMILY_DIR.values()) + [CROSS_DIR]:
    os.makedirs(f'{OUT_DIR}/{d}', exist_ok=True)
base = pd.read_csv(ev.MERGED_PATH, usecols=['run_id', 'k', TS, RS, VS])
base = base.sort_values(['run_id', 'k']).reset_index(drop=True)
FEAT = load_features(W_MAIN)
assert (FEAT.run_id.to_numpy() == base.run_id.to_numpy()).all() and \
       (FEAT.k.to_numpy() == base.k.to_numpy()).all(), 'feature 행 순서가 merged 와 다르다'
d_summary = pd.read_csv(f'{SCREEN_DIR}/{CROSS_DIR}/D_summary_all.csv').query('W == @W_MAIN').set_index('feature')

lab = base[['run_id', 'k']].merge(ev._labels(), on=['run_id', 'k'], how='left')[ev.LABEL_COL].to_numpy()
scored = lab != 2                                              # ambiguous(2) 는 전 분석에서 제외
steady = lab == 1
print(f'[Step 1] rows = {len(base)}, 채점 대상 {scored.sum()}, steady {steady.sum()}, '
      f'transient {(lab == 0).sum()} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 조건 식별력 — (a) 평균 쌍별 AUROC, (b) 분위구간 Bayes 다중분류 정확도
#         (a) 는 '채점 대상 전체' 와 'steady 만' 두 경우를 낸다.
#             steady 만 보는 쪽이 순수한 조건 정보다 (상태 변화가 섞이지 않는다).
# =============================================================================
rows = []
for name, col in COND_AXES:
    lv_all = base[col].to_numpy()
    for c in FEATURES:
        t0 = time.time()
        x = FEAT[c].to_numpy()
        a_all, mx_all = mean_pairwise_auroc(x[scored], lv_all[scored])
        a_st, mx_st = mean_pairwise_auroc(x[steady], lv_all[steady])
        acc, chance = binned_bayes_accuracy(x[scored], lv_all[scored],
                                            base.run_id[scored].reset_index(drop=True),
                                            N_BINS, N_COND_FOLDS)
        rows.append({'cond_axis': name, 'cond_col': col, 'feature': c, 'family': FAMILY[c],
                     'D_state': float(d_summary.loc[c, 'D']),
                     'cond_auroc_scored': a_all, 'cond_auroc_scored_max': mx_all,
                     'cond_auroc_steady': a_st, 'cond_auroc_steady_max': mx_st,
                     'cond_accuracy': acc, 'cond_accuracy_chance': chance,
                     'n_levels': base[col].nunique()})
        print(f'[Step 2] {name:7s} {c:18s} D {rows[-1]["D_state"]:.4f} | 조건 AUROC '
              f'{a_all:.4f} (steady {a_st:.4f}) | 정확도 {acc:.4f} (우연 {chance:.3f}) '
              f'({time.time() - t0:.1f} s)')
overlap = pd.DataFrame(rows)
split_by_family(overlap, 'state_vs_condition.csv')

# =============================================================================
# Step 3. 보조 표 — 속도 수준별 '상태 평균차'. 계획 4.4절의 전제(두 정보가 겹친다)를 직접 본다.
#         |차이| 가 속도에 따라 크게 달라지면 조건 정보를 지울 때 상태 정보도 함께 지워진다.
# =============================================================================
t0 = time.time()
diff_rows = []
for name, col in COND_AXES:
    for level in sorted(base[col].unique()):
        m = (base[col] == level).to_numpy()
        mt, ms, msc = m & (lab == 0), m & steady, m & scored
        y_ns = (lab[msc] == 0).astype(int)                     # non-steady = positive (B 규약과 동일)
        for c in FEATURES:
            x = FEAT[c].to_numpy()
            a = roc_auc_score(y_ns, x[msc])
            diff_rows.append({'cond_axis': name, 'level': level, 'feature': c, 'family': FAMILY[c],
                              'mean_transient': x[mt].mean(), 'mean_steady': x[ms].mean(),
                              'diff': x[mt].mean() - x[ms].mean(),
                              'auroc_within_level': max(a, 1 - a)})
state_diff = pd.DataFrame(diff_rows)
split_by_family(state_diff, 'state_diff_by_condition.csv')

# 수준 간 변동 요약 — 상태 평균차의 변동계수(CV)가 크면 조건 의존이 강하다
cv = (state_diff.groupby(['cond_axis', 'feature'])['diff']
      .agg(mean_diff='mean', std_diff='std').reset_index())
cv['abs_cv'] = (cv.std_diff / cv.mean_diff.abs()).replace([np.inf, -np.inf], np.nan)
cv = cv.merge(overlap[['cond_axis', 'feature', 'family', 'D_state', 'cond_auroc_scored']],
              on=['cond_axis', 'feature'], how='left')
split_by_family(cv, 'state_diff_variation.csv')
print(f'[Step 3] 조건 수준별 상태 평균차 완료 ({time.time() - t0:.1f} s)')
print('[Step 3] 속도축 상태 평균차의 변동계수 상위/하위 3개:')
sp = cv[cv.cond_axis == 'speed'].dropna(subset=['abs_cv']).sort_values('abs_cv')
print(pd.concat([sp.head(3), sp.tail(3)])[['feature', 'mean_diff', 'std_diff', 'abs_cv']].to_string(index=False))

# =============================================================================
# Step 4. 그림 (논문 스타일 — 축·범례 영어, PNG 300 dpi + PDF, 계열마다 고정 색)
# =============================================================================
t0 = time.time()

# --- 4.1 상태 vs 조건 산점도 (A-4 의 핵심 그림). 왼쪽 위가 이상적 ---
#     전체판은 _cross/ 에, 계열마다 그 계열만 강조한 판은 계열 폴더에 둔다
for focus in [None] + list(FAMILY_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(ev.PAGE_WIDTH, 3.1), layout='constrained', sharey=True)
    for ax, (name, col) in zip(axes, COND_AXES):
        g_all = overlap[overlap.cond_axis == name]
        for fam, g in g_all.groupby('family'):
            c_, mk = FAMILY_STYLE[fam]
            dim = focus is not None and fam != focus          # 강조 대상이 아니면 회색으로 뒤에 깐다
            ax.scatter(g.cond_auroc_scored, g.D_state, c='#c9c9c9' if dim else c_, marker=mk,
                       s=20 if dim else 30, label=fam, zorder=3 if dim else 4)
        for _, r in g_all.iterrows():
            dim = focus is not None and r.family != focus
            ax.annotate(r.feature, (r.cond_auroc_scored, r.D_state), textcoords='offset points',
                        xytext=(3, 3), fontsize=ev.FONT_SIZE - 4,
                        color='#bbbbbb' if dim else '#333333')
        ax.axvline(0.5, color='#888888', linewidth=0.8, linestyle=':')
        ax.set_xlabel(f'Condition AUROC ({name} level)')
        ax.set_title(f'{name.capitalize()} axis ({g_all.n_levels.iloc[0]} levels)')
    axes[0].set_ylabel(f'State AUROC (within run), W = {W_MAIN}')
    # 범례는 그림 밖에 둔다 — 안쪽에 두면 왼쪽 아래 점을 가린다
    fig.legend(*axes[0].get_legend_handles_labels(), title='Family', ncols=8,
               loc='outside lower center', frameon=False)
    save_plot(fig, f'{OUT_DIR}/{CROSS_DIR if focus is None else FAMILY_DIR[focus]}'
                   f'/state_vs_condition_W{W_MAIN}')

# --- 4.2 속도 수준별 상태 평균차 (겹침의 직접 증거). 진폭 계열만 단위가 같으므로 kg 계열로 제한 ---
kg_feats = spec.loc[(spec.unit == 'kg') & (spec.amplitude_scale_exponent == 1), 'feature'].tolist()
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH * 0.75, 2.9), layout='constrained')
sd = state_diff[(state_diff.cond_axis == 'speed') & (state_diff.feature.isin(kg_feats))]
for i, (c, g) in enumerate(sd.groupby('feature')):
    col, mk = FAMILY_STYLE[FAMILY[c]]
    ax.plot(g.level, g['diff'], color=col, marker=mk, linewidth=1.2, label=c,
            linestyle=['-', '--', '-.', ':'][i % 4])
ax.axhline(0, color='#000000', linewidth=0.8)
ax.set_xlabel('Web speed setting (mm/s)')
ax.set_ylabel('State mean difference (transient - steady, kg)')
ax.legend(fontsize=ev.FONT_SIZE - 2, ncols=2)
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/state_diff_by_speed_W{W_MAIN}')
print(f'[Step 4] 그림 2종 저장 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 5. hyperparameter 기록
# =============================================================================
cfg = {'SEED': SEED, 'W_MAIN': W_MAIN, 'N_BINS': N_BINS, 'N_COND_FOLDS': N_COND_FOLDS,
       'COND_AXES': str([a for a, _ in COND_AXES]),
       'COND_PRIMARY': '평균 쌍별 AUROC (우연 0.5, 상태 AUROC 와 같은 잣대)',
       'COND_SECONDARY': f'분위구간({N_BINS}) Bayes 다중분류 정확도, run 단위 {N_COND_FOLDS}-fold',
       'D_SOURCE': f'{SCREEN_DIR}/{CROSS_DIR}/D_summary_all.csv (03_screening 과 같은 값 재사용)',
       'LABEL_USE': 'evaluation._labels() 로만 읽음 (분석 전용). feature 정의에는 미사용'}
pd.DataFrame({'name': list(cfg), 'value': [str(v) for v in cfg.values()]}).to_csv(
    f'{OUT_DIR}/{CROSS_DIR}/hyperparameters.csv', index=False)
print(f'[done] A-4 상태/조건 중첩 분석 완료 (총 {time.time() - t_all:.1f} s)')
