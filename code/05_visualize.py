# =============================================================================
# code/05_visualize.py : feature 계열별 시각화 + A-6 보조 시각화
#                        (docs/research_plan.md 4.6절 + 2026-10-28 사용자 보강 지시)
#
# 역할 : 판정용이 아니라 **설명용 그림**이다. 두 질문에 그림으로 답한다.
#          Q1. steady state 와 transient 사이에 차이가 있는가?
#          Q2. 각 feature 를 실제로 추출하면 결과가 어떻게 나오는가?
#
#        **feature 계열(= 사용자 제안 4계열)마다 폴더를 나누고, 계열마다 그림 3종을 넣는다.**
#          1) state_distribution_W{W}  : 상태별 분포 비교        -> Q1
#             transient / steady 의 feature 분포를 겹쳐 그리고, 분리도(D)와 효과크기(Cohen's d)를
#             패널 제목에 적는다. 분포가 겹치면 그 feature 로는 두 상태를 못 가른다.
#          2) timeline_W{W}            : 추출 결과 시계열        -> Q2
#             대표 run 2개(최저속 50 / 최고속 200)에서 원신호 e 와 그 계열 feature 를 같은
#             시간축에 쌓아 그리고, 라벨 구간을 배경색으로 깐다. 전이 순간에 feature 가 실제로
#             어떻게 움직이는지(또는 움직이지 않는지)가 그대로 보인다.
#          3) window_separation        : 창 길이별 분리도        -> Q1 + Q2
#             같은 feature 를 창 64/144/256 에서 뽑았을 때 상태 분리도가 어떻게 변하는지.
#             창 144 + backfill 이 과도 구간 분해능을 0 으로 만드는 효과(계획 L2)를 계열별로 본다.
#
#        A-6 (계획 4.6절) 의 그림 2종은 해당 계열 폴더에 함께 들어간다.
#          P1_spectral/ spectrogram_W64         과도 -> 정상 전이에서 스펙트럼이 어떻게 바뀌는지
#          P2_phase/    phase_portrait_lag1     steady 의 limit cycle vs transient 의 이탈
#
#        대표 run : 각 속도 1개, 장력·ReWinder 는 중앙값 수준 (승인 #13). 7개 run.
#        Δt = 1 샘플 (0.109 s) (승인 #7, **임의의 값**).
#
# [라벨 사용 경계] 라벨은 evaluation._labels() 로만 읽고 **그림 주석(색·경계선·상태별 분포)에만**
#   쓴다 (계획 5절 표: 05_visualize 는 '그림 주석용만'). 수치 판정에는 쓰지 않는다.
#   분리도 D 는 03_screening 이 B 의 평가 코드로 이미 산출한 값을 그대로 가져다 적는다.
#
# 입력 : outputs/01_merge/merged_data.csv            (`Film Tension #1 (kg)` 와 설정값)
#        outputs/02_features/{계열}/features_W{W}.csv.gz, _cross/feature_scaling.csv
#        outputs/03_screening/_cross/D_summary_all.csv  (분리도 D — 같은 코드로 산출된 값 재사용)
# 출력 : outputs/05_visualize/  — feature 계열마다 폴더 (02~04 와 같은 이름)
#          P0_baseline/ ~ P5_additional/  그림 3종 (+ P1 spectrogram, P2 phase portrait)
#          _cross/                    전 계열 분리도 heatmap, 대표 run 목록, hyperparameters.csv
#        마지막 단계에서 `docs/methods/{계열}.md` 를 전 단계의 같은 계열 폴더로 복사한다 (Step 8)
# 실행 : python code/05_visualize.py      (03_screening.py 다음에 실행한다)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev

# --- 어댑터 : 복사한 외부 코드의 경로 상수만 런타임에 바꾼다 (원본 파일 미수정) -----------
ev.MERGED_PATH = 'outputs/01_merge/merged_data.csv'

# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                         # 난수 시드 (이 단계는 결정론적이지만 규칙상 1개로 고정해 둔다)
DT = ev.DT                       # 샘플 간격 (s). B 승계값
WINDOWS = [64, 144, 256]         # 이동창 길이 (samples). 02_features 와 같은 값
W_MAIN = 144                     # 주 창 길이 (samples). 분포·시계열 그림이 쓰는 창

TIMELINE_SPEEDS = [50, 200]      # 시계열 그림에 쓸 대표 run 의 속도 (mm/s). **임의의 값**
                                 # 조건 격자의 양 끝 — 저속과 고속에서 feature 거동이 다른지 함께 본다
TIMELINE_SECONDS = 60            # 시계열 그림의 가로축 길이 (s). **임의의 값**
                                 # 과도 구간(중앙값 7.0 s)과 진입 직후 정상 구간이 함께 보이는 길이
HIST_BINS = 60                   # 상태별 분포 히스토그램의 구간 수. **임의의 값**
HIST_CLIP_PCT = (0.5, 99.5)      # 분포 그림의 가로 범위를 이 분위로 자른다. **임의의 값**
                                 # 꼬리를 그대로 두면 분포 본체가 한 칸으로 눌린다

SPEC_NPERSEG = 64                # spectrogram 분석창 (samples = 7.0 s). **임의의 값**
                                 # 주 창 144(15.7 s)를 쓰면 과도 구간이 창 하나에 통째로 들어가
                                 # 전이 자체가 보이지 않는다 (설명용 그림의 목적이 성립하지 않는다)
SPEC_OVERLAP = 56                # spectrogram 창 겹침 (samples, 87.5 %). **임의의 값**
SPEC_CLIP_PCT = (2, 98)          # spectrogram 색 범위를 전체 패널의 이 분위로 자른다. **임의의 값**
SPEC_FMAX = 2.0                  # spectrogram 세로축 상한 (Hz). **임의의 값**
PHASE_LAG = 1                    # phase portrait 의 Δt (samples = 0.109 s). 승인 #7, **임의의 값**
PHASE_MAX_POINTS = 400           # phase portrait 에 찍을 시점 수 상한 (앞부분부터). **임의의 값**

ROLL_CIRCUMFERENCE_MM = 783.5    # B 측정 회전 둘레 (docs/research_plan.md 1.3절)

IN_PATH = 'outputs/01_merge/merged_data.csv'
FEAT_DIR = 'outputs/02_features'
SCREEN_DIR = 'outputs/03_screening'
OUT_DIR = 'outputs/05_visualize'
CROSS_DIR = '_cross'             # 계열에 속하지 않는 산출물을 모으는 폴더 이름 (02~04 와 같다)
SPECTROGRAM_FAMILY = 'P1'        # spectrogram 은 제안 P1(스펙트럼) 계열의 설명 그림이다
PHASE_FAMILY = 'P2'              # phase portrait 은 제안 P2(위상) 계열의 설명 그림이다

T1 = 'Film Tension #1 (kg)'      # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS

# 계열별 고정 색 — 03·04 와 같은 배정 (P0 검정 / 제안 4계열 유채색 / P5 회색)
FAMILY_STYLE = {'P0': '#000000', 'P1': '#d55e00', 'P2': '#662aa2',
                'P3': '#009e73', 'P4': '#e69f00', 'P5': '#8c8c8c'}

STATE_COLOR = {0: '#d55e00', 1: '#0072b2', 2: '#999999'}       # transient / steady / ambiguous
STATE_LABEL = {0: 'Transient', 1: 'Steady', 2: 'Ambiguous'}

# =============================================================================
# 함수
# =============================================================================


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


def load_features(w):
    # 역할 : 계열 폴더에 나뉘어 저장된 창 w 의 feature 를 한 표로 합친다 (행 순서는 모든 계열에서 같다).
    # 입력 : w 창 길이 (samples)
    # 출력 : DataFrame [run_id, k, 16 feature]
    parts = [pd.read_csv(f'{FEAT_DIR}/{d}/features_W{w}.csv.gz') for d in FAMILY_DIR.values()]
    for q in parts[1:]:
        assert (q.run_id.to_numpy() == parts[0].run_id.to_numpy()).all(), '계열 파일 간 행 순서가 다르다'
    return pd.concat([parts[0]] + [q.drop(columns=['run_id', 'k']) for q in parts[1:]], axis=1)


def shade_states(ax, t, state):
    # 역할 : 시계열 패널의 배경에 라벨 구간을 색으로 깐다 (연속된 같은 라벨 구간 = 색 띠 하나).
    #        feature 가 전이 순간에 움직이는지 눈으로 바로 맞춰 볼 수 있게 하는 장치다.
    # 입력 : ax 패널 / t (n,) 시간(s) / state (n,) 라벨 (0/1/2)
    # 출력 : 없음
    edge = np.flatnonzero(np.diff(state) != 0) + 1
    for a, b in zip(np.r_[0, edge], np.r_[edge, len(state)]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=STATE_COLOR[int(state[a])], alpha=0.13, lw=0)


def cohens_d(x0, x1):
    # 역할 : 두 집단 평균차를 합동 표준편차로 나눈 효과크기. 분포 그림의 제목에 적는다.
    #        AUROC 가 "순위로 얼마나 갈리나" 라면 이 값은 "평균이 표준편차 몇 배만큼 떨어져 있나" 다.
    # 입력 : x0 (n0,) transient 값 / x1 (n1,) steady 값
    # 출력 : float (부호 포함. + 면 transient 쪽이 크다)
    n0, n1 = len(x0), len(x1)
    sp = np.sqrt(((n0 - 1) * x0.var(ddof=1) + (n1 - 1) * x1.var(ddof=1)) / (n0 + n1 - 2))
    return float((x0.mean() - x1.mean()) / sp) if sp > 0 else np.nan


# =============================================================================
# Step 1. 데이터 로드 + 라벨 결합 + 대표 run 선정
# =============================================================================
t_all = time.time()
t0 = time.time()
spec = pd.read_csv(f'{FEAT_DIR}/{CROSS_DIR}/feature_scaling.csv')
FAMILY_DIR = dict(zip(spec.family, spec.family_dir))       # 02_features 가 정한 계열 폴더 이름을 그대로 쓴다
FAMILY_COLS = {fam: spec.loc[spec.family == fam, 'feature'].tolist() for fam in FAMILY_DIR}
UNIT = dict(zip(spec.feature, spec.unit))
for d in list(FAMILY_DIR.values()) + [CROSS_DIR]:
    os.makedirs(f'{OUT_DIR}/{d}', exist_ok=True)

df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
df['e'] = df[T1] - df[TS]
# 라벨 결합 (그림 주석용. evaluation._labels 가 유일한 라벨 접근 지점이다)
df['state'] = df[['run_id', 'k']].merge(ev._labels(), on=['run_id', 'k'], how='left')[ev.LABEL_COL].to_numpy()

FEAT_BY_W = {}
for w in WINDOWS:
    f = load_features(w)
    assert (f.run_id.to_numpy() == df.run_id.to_numpy()).all() and \
           (f.k.to_numpy() == df.k.to_numpy()).all(), f'창 {w}: feature 행 순서가 merged 와 다르다'
    FEAT_BY_W[w] = f
FEAT = FEAT_BY_W[W_MAIN]

# 분리도 D — 03_screening 이 B 의 평가 코드로 산출한 값을 그대로 쓴다 (새로 계산하지 않는다)
d_path = f'{SCREEN_DIR}/{CROSS_DIR}/D_summary_all.csv'
assert os.path.exists(d_path), f'{d_path} 가 없다. 먼저 python code/03_screening.py 를 실행한다'
D_TABLE = pd.read_csv(d_path).set_index(['W', 'feature']).D

# 중앙값 수준 : 정렬한 수준 목록의 가운데 값 (수준 수가 짝수면 위쪽 값)
t_med = sorted(df[TS].unique())[df[TS].nunique() // 2]
r_med = sorted(df[RS].unique())[df[RS].nunique() // 2]
speeds = sorted(df[VS].unique())
reps = (df[(df[TS] == t_med) & (df[RS] == r_med)].groupby([VS, 'run_id']).size()
        .reset_index().rename(columns={0: 'n'}).sort_values(VS))
assert len(reps) == len(speeds), f'대표 run 선정 실패: 속도 {len(speeds)}개 vs run {len(reps)}개'
reps.to_csv(f'{OUT_DIR}/{CROSS_DIR}/representative_runs.csv', index=False)
TL_RUNS = [(v, reps.loc[reps[VS] == v, 'run_id'].iloc[0]) for v in TIMELINE_SPEEDS]
print(f'[Step 1] 중앙값 수준 장력 {t_med} kg / ReWinder {r_med} kg, 대표 run {len(reps)}개. '
      f'시계열 그림 run = {[r for _, r in TL_RUNS]} ({time.time() - t0:.1f} s)')

# 상태 마스크 (ambiguous 는 분포 비교에서 제외한다 — 채점 규약과 같다)
m_tr, m_st = (df.state == 0).to_numpy(), (df.state == 1).to_numpy()

# =============================================================================
# Step 2. 계열별 그림 1 — 상태별 분포 비교 (Q1: steady 와 transient 가 다른가)
#         transient / steady 의 feature 분포를 겹쳐 그린다. 많이 겹칠수록 못 가르는 feature 다.
#         제목에 D(run 내 AUROC, 03_screening 산출)와 Cohen's d 를 함께 적는다.
# =============================================================================
t0 = time.time()
sep_rows = []
for fam, dirname in FAMILY_DIR.items():
    cols = FAMILY_COLS[fam]
    ncols = min(3, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout='constrained',
                             figsize=(ev.PAGE_WIDTH, 2.3 * nrows + 0.5))
    for ax, c in zip(axes.ravel(), cols):
        x = FEAT[c].to_numpy()
        lo, hi = np.percentile(x[m_tr | m_st], HIST_CLIP_PCT)
        bins = np.linspace(lo, hi, HIST_BINS + 1)
        for m, sv in [(m_st, 1), (m_tr, 0)]:                   # steady 를 먼저 깔고 transient 를 위에
            ax.hist(np.clip(x[m], lo, hi), bins=bins, density=True, color=STATE_COLOR[sv],
                    alpha=0.55, label=STATE_LABEL[sv])
        d_auroc = float(D_TABLE.loc[(W_MAIN, c)])
        d_eff = cohens_d(x[m_tr], x[m_st])
        sep_rows.append({'family': fam, 'feature': c, 'W': W_MAIN, 'D': d_auroc, 'cohens_d': d_eff})
        ax.set_title(f'{c}\nD = {d_auroc:.3f},  d = {d_eff:+.2f}', fontsize=ev.FONT_SIZE - 1)
        ax.set_xlabel(f'{c} ({UNIT[c]})' if UNIT[c] != '-' else c)
    for ax in axes.ravel()[len(cols):]:
        ax.axis('off')
    axes.ravel()[0].set_ylabel('Probability density')
    fig.legend(*axes.ravel()[0].get_legend_handles_labels(), ncols=2,
               loc='outside lower center', frameon=False)
    save_plot(fig, f'{OUT_DIR}/{dirname}/state_distribution_W{W_MAIN}')
print(f'[Step 2] 계열별 상태 분포 그림 {len(FAMILY_DIR)}종 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 3. 계열별 그림 2 — 추출 결과 시계열 (Q2: 실제로 뽑으면 어떻게 나오는가)
#         맨 위 = 원신호 e, 아래 = 그 계열 feature. 배경색 = 라벨 구간.
#         열 = 대표 run 2개 (최저속 50 / 최고속 200).
# =============================================================================
t0 = time.time()
for fam, dirname in FAMILY_DIR.items():
    cols = FAMILY_COLS[fam]
    nrows = 1 + len(cols)
    fig, axes = plt.subplots(nrows, len(TL_RUNS), squeeze=False, layout='constrained',
                             sharex=True, figsize=(ev.PAGE_WIDTH, 1.15 * nrows + 0.8))
    for j, (v, run_id) in enumerate(TL_RUNS):
        m = (df.run_id == run_id).to_numpy()
        n = min(int(TIMELINE_SECONDS / DT), int(m.sum()))
        idx = np.flatnonzero(m)[:n]
        t = df.k.to_numpy()[idx] * DT
        st = df.state.to_numpy()[idx]
        axes[0, j].plot(t, df.e.to_numpy()[idx], color='#333333', linewidth=0.8)
        shade_states(axes[0, j], t, st)
        axes[0, j].set_title(f'{run_id} (web speed {int(v)} mm/s)', fontsize=ev.FONT_SIZE - 1)
        if j == 0:
            axes[0, j].set_ylabel('e (kg)', fontsize=ev.FONT_SIZE - 1)
        for i, c in enumerate(cols, start=1):
            axes[i, j].plot(t, FEAT[c].to_numpy()[idx], color=FAMILY_STYLE[fam], linewidth=1.0)
            shade_states(axes[i, j], t, st)
            if j == 0:
                axes[i, j].set_ylabel(f'{c}\n({UNIT[c]})' if UNIT[c] != '-' else c,
                                      fontsize=ev.FONT_SIZE - 2)
        axes[-1, j].set_xlabel('Time (s)')
    # 배경색 범례 (라벨 구간) — 선 범례가 아니라 색 띠 설명이다
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLOR[s], alpha=0.35) for s in [0, 1, 2]]
    fig.legend(handles, [STATE_LABEL[s] for s in [0, 1, 2]], ncols=3,
               loc='outside lower center', frameon=False)
    save_plot(fig, f'{OUT_DIR}/{dirname}/timeline_W{W_MAIN}')
print(f'[Step 3] 계열별 추출 결과 시계열 {len(FAMILY_DIR)}종 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 4. 계열별 그림 3 — 창 길이별 상태 분리도
#         같은 feature 를 창 64/144/256 에서 뽑았을 때 D 가 어떻게 변하는지.
#         창 144 + backfill 이 과도 구간 분해능을 0 으로 만드는 효과(계획 L2)를 계열별로 본다.
# =============================================================================
t0 = time.time()
W_COLOR = {64: '#cde2fb', 144: '#3987e5', 256: '#104281'}      # 창 길이 = 밝기 (sequential)
for fam, dirname in FAMILY_DIR.items():
    cols = FAMILY_COLS[fam]
    fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH * 0.72, 2.7), layout='constrained')
    x = np.arange(len(cols))
    for i, w in enumerate(WINDOWS):
        vals = [float(D_TABLE.loc[(w, c)]) for c in cols]
        ax.bar(x + (i - 1) * 0.26, vals, width=0.26, color=W_COLOR[w], edgecolor='#333333',
               linewidth=0.5, label=f'W = {w} ({w * DT:.1f} s)')
    ax.axhline(0.5, color='#888888', linewidth=0.8, linestyle=':')   # 우연 수준
    ax.set_xticks(x, cols, rotation=20, ha='right', fontsize=ev.FONT_SIZE - 1)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('D: within-run state AUROC')
    fig.legend(*ax.get_legend_handles_labels(), ncols=3, loc='outside lower center', frameon=False)
    save_plot(fig, f'{OUT_DIR}/{dirname}/window_separation')
print(f'[Step 4] 계열별 창 민감도 그림 {len(FAMILY_DIR)}종 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 5. F3 — Time-frequency spectrogram (계획 4.6절)
#         창 평균 제거(detrend='constant') + Hann. 평균을 두면 DC 가 전체를 덮는다 (계획 3.1절)
#         회전 외란 주파수 f_rot = V / 783.5 mm 를 가로선으로 함께 표시한다
# =============================================================================
t0 = time.time()
specs = {}
for _, r in reps.iterrows():                                   # 색 범위를 공유값으로 쓰려고 먼저 전부 계산한다
    g = df[df.run_id == r.run_id]
    f, t, s = spectrogram(g.e.to_numpy(), fs=1.0 / DT, window='hann', nperseg=SPEC_NPERSEG,
                          noverlap=SPEC_OVERLAP, detrend='constant', scaling='density')
    centroid = (s * f[:, None]).sum(axis=0) / s.sum(axis=0)    # 열마다 전력가중 스펙트럼 중심
    specs[r.run_id] = (f, t, 10 * np.log10(s + 1e-12), centroid)
allv = np.concatenate([v[2].ravel() for v in specs.values()])
vmin, vmax = np.percentile(allv, SPEC_CLIP_PCT)

fig, axes = plt.subplots(len(reps), 1, figsize=(ev.PAGE_WIDTH, 1.45 * len(reps) + 0.6),
                         layout='constrained', sharex=True, sharey=True)
for ax, (_, r) in zip(np.atleast_1d(axes), reps.iterrows()):
    g = df[df.run_id == r.run_id]
    f, t, db, centroid = specs[r.run_id]
    im = ax.pcolormesh(t, f, db, cmap=ev.SEQ_CMAP, vmin=vmin, vmax=vmax, shading='nearest')
    ax.plot(t, centroid, color='#ffd100', linewidth=1.0, label='Spectral centroid')
    entry = g.loc[g.state == 1, 'k']
    if len(entry):
        ax.axvline(int(entry.min()) * DT, color='#d55e00', linewidth=1.2,
                   label='First labelled steady sample')
    ax.axhline(r[VS] / ROLL_CIRCUMFERENCE_MM, color='#ffffff', linewidth=0.8, linestyle='--',
               label='Rotational disturbance $f_{rot}$')
    ax.set_title(f'{r.run_id} (web speed {int(r[VS])} mm/s)', fontsize=ev.FONT_SIZE - 1)
    ax.set_ylim(0, SPEC_FMAX)
    ax.grid(False)
fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), ncols=3,
           loc='outside upper center', frameon=False)
fig.supxlabel('Time (s)', fontsize=ev.FONT_SIZE)
fig.supylabel('Frequency (Hz)', fontsize=ev.FONT_SIZE)
fig.colorbar(im, ax=axes, label=r'Power spectral density (dB re 1 kg$^2$/Hz)', fraction=0.03, pad=0.02)
save_plot(fig, f'{OUT_DIR}/{FAMILY_DIR[SPECTROGRAM_FAMILY]}/spectrogram_W{SPEC_NPERSEG}')
print(f'[Step 5] spectrogram 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 6. F6 — Phase portrait e(t) vs e(t+Δt) (계획 4.6절)
#         run 앞부분 PHASE_MAX_POINTS 시점만 그린다 (과도 구간이 run 앞쪽에 몰려 있다)
# =============================================================================
t0 = time.time()
ncols = 4
nrows = int(np.ceil(len(reps) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ev.PAGE_WIDTH, 1.75 * nrows + 0.5),
                         layout='constrained', sharex=True, sharey=True)
for ax, (_, r) in zip(axes.ravel(), reps.iterrows()):
    g = df[df.run_id == r.run_id].head(PHASE_MAX_POINTS)
    e, st = g.e.to_numpy(), g.state.to_numpy()
    x, y, sx = e[:-PHASE_LAG], e[PHASE_LAG:], st[:-PHASE_LAG]
    for sv in [2, 1, 0]:                                       # transient 를 맨 위에 그린다
        m = sx == sv
        ax.scatter(x[m], y[m], s=3, c=STATE_COLOR[sv], label=STATE_LABEL[sv], zorder=3 + (sv == 0))
    ax.plot(x, y, color='#bbbbbb', linewidth=0.3, zorder=2)    # 시간 순서를 잇는 궤적
    ax.set_title(f'{int(r[VS])} mm/s', fontsize=ev.FONT_SIZE - 1)
    ax.set_aspect('equal', adjustable='box')                   # 축 공유 중이라 box 로 맞춘다
for ax in axes.ravel()[len(reps):]:
    ax.axis('off')
h, l = axes.ravel()[0].get_legend_handles_labels()
axes.ravel()[-1].legend(h, l, loc='center', fontsize=ev.FONT_SIZE - 2, markerscale=3, frameon=False)
fig.supxlabel('e(t) (kg)', fontsize=ev.FONT_SIZE)
fig.supylabel(rf'e(t + {PHASE_LAG * DT:.3f} s) (kg)', fontsize=ev.FONT_SIZE)
save_plot(fig, f'{OUT_DIR}/{FAMILY_DIR[PHASE_FAMILY]}/phase_portrait_lag{PHASE_LAG}')
print(f'[Step 6] phase portrait 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 7. _cross — 전 계열 한눈에 보기 (feature x 창 분리도 heatmap + 효과크기 표)
# =============================================================================
t0 = time.time()
sep = pd.DataFrame(sep_rows)
sep.to_csv(f'{OUT_DIR}/{CROSS_DIR}/state_separation.csv', index=False)
order = [c for fam in FAMILY_DIR for c in FAMILY_COLS[fam]]
grid = np.array([[float(D_TABLE.loc[(w, c)]) for w in WINDOWS] for c in order])
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH * 0.62, 4.6), layout='constrained')
im = ax.imshow(grid, cmap=ev.SEQ_CMAP, vmin=0.5, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(WINDOWS)), [f'{w}' for w in WINDOWS])
ax.set_yticks(range(len(order)), order, fontsize=ev.FONT_SIZE - 2)
ax.set_xlabel('Window length (samples)')
ax.grid(False)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f'{grid[i, j]:.3f}'.lstrip('0'), ha='center', va='center',
                fontsize=ev.FONT_SIZE - 3, color='#ffffff' if grid[i, j] > 0.78 else '#0b0b0b')
fig.colorbar(im, ax=ax, label='D: within-run state AUROC', fraction=0.06, pad=0.03)
save_plot(fig, f'{OUT_DIR}/{CROSS_DIR}/state_separation_heatmap')
print(f'[Step 7] 전 계열 분리도 heatmap 완료 ({time.time() - t0:.1f} s)')
print('[Step 7] 창 144 기준 효과크기 |d| 상위 5개:')
print(sep.reindex(sep.cohens_d.abs().sort_values(ascending=False).index)
      .head(5)[['family', 'feature', 'D', 'cohens_d']].to_string(index=False))

# =============================================================================
# Step 8. 방법론 문서 배포 — docs/methods/{계열}.md 를 전 단계의 같은 계열 폴더로 복사한다
#
#   문서의 원본은 `docs/` 에 둔다. outputs/ 는 재실행할 때 통째로 지우는 생성물 폴더라
#   손으로 쓴 문서를 거기에만 두면 다음 실행에서 사라지기 때문이다.
#   대신 마지막 단계에서 복사본을 뿌려, 결과를 보는 자리에서 바로 문서를 열 수 있게 한다.
#   (파이프라인을 다시 돌리면 복사본도 자동으로 최신화된다)
# =============================================================================
t0 = time.time()
DOC_SRC = 'docs/methods'
STAGE_DIRS = ['outputs/02_features', 'outputs/03_screening', 'outputs/04_overlap', OUT_DIR]
n_doc = 0
for dirname in FAMILY_DIR.values():
    src = f'{DOC_SRC}/{dirname}.md'
    if not os.path.exists(src):                                # 제안 계열 4개에만 문서가 있다
        continue
    text = open(src, encoding='utf-8').read()
    for stage in STAGE_DIRS:
        dst = f'{stage}/{dirname}/{dirname}.md'
        if os.path.isdir(f'{stage}/{dirname}'):
            open(dst, 'w', encoding='utf-8', newline='\n').write(text)
            n_doc += 1
print(f'[Step 8] 방법론 문서 {n_doc}개 배포 완료 (원본 {DOC_SRC}/) ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 9. hyperparameter 기록
# =============================================================================
cfg = {'SEED': SEED, 'DT': DT, 'WINDOWS': WINDOWS, 'W_MAIN': W_MAIN,
       'TIMELINE_SPEEDS': TIMELINE_SPEEDS, 'TIMELINE_SECONDS': TIMELINE_SECONDS,
       'HIST_BINS': HIST_BINS, 'HIST_CLIP_PCT': HIST_CLIP_PCT,
       'SPEC_NPERSEG': SPEC_NPERSEG, 'SPEC_OVERLAP': SPEC_OVERLAP,
       'SPEC_CLIP_PCT': SPEC_CLIP_PCT, 'SPEC_FMAX': SPEC_FMAX,
       'SPEC_WINDOW_FUNCTION': 'hann', 'SPEC_DETREND': 'constant (창 평균 제거)',
       'PHASE_LAG': PHASE_LAG, 'PHASE_LAG_SECONDS': PHASE_LAG * DT,
       'PHASE_MAX_POINTS': PHASE_MAX_POINTS,
       'REPRESENTATIVE_TENSION': t_med, 'REPRESENTATIVE_REWINDER': r_med,
       'REPRESENTATIVE_RUNS': '+'.join(reps.run_id.tolist()),
       'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
       'D_SOURCE': f'{d_path} (03_screening 과 같은 값 재사용)',
       'LABEL_USE': '그림 주석(색·경계선)과 상태별 분포에만 사용. 수치 판정에는 미사용'}
pd.DataFrame({'name': list(cfg), 'value': [str(v) for v in cfg.values()]}).to_csv(
    f'{OUT_DIR}/{CROSS_DIR}/hyperparameters.csv', index=False)
print(f'[done] 계열별 시각화 + A-6 완료 (총 {time.time() - t_all:.1f} s)')
