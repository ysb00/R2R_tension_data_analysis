# =============================================================================
# code/02_features.py : 장력 신호 1채널에서 시점별 feature 산출 (docs/research_plan.md 3.2절)
#
# 역할 : `Film Tension #1 (kg)` 단독 신호의 잔차 e 에 causal 이동창을 씌워 F0~F7 계열
#        16개 feature 를 **시점마다** 산출한다. 평가 후단(evaluation.py)이 시점별 표현을
#        받으므로 feature 도 시점별이어야 한다.
#
# 승계 규약 (B 저장소 = R2R_steadyState_identification_for_changing_condition)
#   - 잔차 e = `Film Tension #1 (kg)` - `Film-Tension Setting Value (kg)`
#   - 창 미완성 구간은 **버리지 않는다** — 앞 W-1 시점은 첫 완전창의 값으로 backfill.
#     (창 길이가 달라도 채점 대상 시점 수가 같아야 한다. B `01_stat_ssd.py:117-118`)
#   - 주 창 길이 144 샘플 (회전 외란 1주기, V=50 기준). 민감도용 64·256 을 함께 산출한다.
#     64 는 B 통계 baseline `MovStd-Slope-fixed` 의 창이라 승계 검증 기준점이 된다.
#   - F0(`movstd`, `movslope`)는 B 의 `rolling_std_slope` 를 **그대로 복사**해 쓴다.
#
# [라벨 누출 차단 — 코드 구조]
#   이 파일은 `Trigger 4` 를 usecols 에 넣지 않으며 feature csv 에도 저장하지 않는다.
#   라벨은 채점 단계(evaluation.py 의 _labels)에서 run_id·k 로 조인해 가져온다. B 와 동일 구조.
#
# 입력 : outputs/01_merge/merged_data.csv  (`Trigger 4` 는 읽지 않는다)
# 출력 : outputs/02_features/  — **feature 계열(= 사용자 제안 4계열)마다 폴더를 나눈다**
#          F0_baseline/   movstd, movslope                       (B 통계 baseline 과 같은 기준점)
#          F1_amplitude/  movmean_abs_e, movrange, movmad
#          F2_shape/      movkurt, movskew                       <- 제안 P3 (비정상성) 일부
#          F3_spectral/   spec_centroid, spec_rolloff85, spec_bandwidth   <- 제안 P1 (스펙트럼)
#          F4_trend/      detrend_resid_std, half_mean_diff
#          F5_envelope/   env_mean, env_std                      <- 제안 P4 (물리기반) 채택분
#          F6_phase/      phase_vel_std                          <- 제안 P2 (위상) 채택분
#          F7_memory/     hurst                                  <- 제안 P3 (비정상성) 일부
#          _cross/        계열에 속하지 않는 공통 산출물
#        계열 폴더 안 : features_W{W}.csv.gz (run_id, k, 그 계열 feature), feature_summary.csv
#        `_cross/` 안 : feature_scaling.csv (정의표 + 진폭 배율 지수, A-5 용),
#                       run_scales.csv (run별 진폭 정규화 분모), hyperparameters.csv
#        F7_memory/ 에는 hurst_fit_quality.csv (R/S log-log 적합 품질) 가 추가된다
# 실행 : python code/02_features.py
# =============================================================================
import os, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                         # 난수 시드 (이 단계는 결정론적이지만 규칙상 1개로 고정해 둔다)
DT = 0.109                       # 샘플 간격 (s). B `evaluation.py:40` 승계값
WINDOWS = [64, 144, 256]         # 이동창 길이 (samples). 승인 #1 — 주 설정 144, 민감도 64·256
W_MAIN = 144                     # 주 창 길이 (회전 외란 1주기, V=50). B 본문 표와 동일

ROLLOFF_FRACTION = 0.85          # spectral roll-off 누적 에너지 비율. 승인 #4 (제안문 승계)
SPEC_DETREND_MEAN = True         # 스펙트럼 산출 전 창 내 평균 제거. 승인 — 없으면 roll-off 가 0 Hz 로 고정된다
KURTOSIS_FISHER = True           # 첨도 규약: excess (정규 = 0). 승인 #6 (scipy 기본)
KURTOSIS_BIAS = True             # scipy 기본값 (표본 편향 보정 없음) — 승인 #6 의 "scipy 기본" 에 포함
HURST_MIN_BLOCK = 8              # R/S 추정의 최소 블록 길이 (samples). **임의의 값**
                                 # 블록 척도는 창 길이를 2로 계속 나눠 만든다 (W, W/2, ... >= 이 값).
                                 # 나누어떨어지는 척도만 쓰므로 블록 나머지가 생기지 않는다

IN_PATH = 'outputs/01_merge/merged_data.csv'
OUT_DIR = 'outputs/02_features'

T1 = 'Film Tension #1 (kg)'      # 원본 컬럼명 (변경하지 않음)
TS = 'Film-Tension Setting Value (kg)'
RS = 'ReWinder-Tension Setting Value (kg)'
VS = 'Web Speed setting value (mm/sec)'

# feature 정의표 — 이름 / 계열 / 진폭 배율 지수 / 단위 / 설명
#   배율 지수 : e -> c*e (c > 0, run 내 상수) 일 때 feature -> c^지수 * feature.
#   1 = 진폭에 비례 (std 계열), 0 = 진폭 불변 (형상·주파수·위상·R/S 계열).
#   A-5(진폭 정규화 3종)는 이 지수로 재계산한다 — feature 를 다시 산출할 필요가 없다.
FEATURE_SPEC = [
    ('movstd',            'F0', 1, 'kg',         '이동 표준편차 (ddof=1). B rolling_std_slope 그대로'),
    ('movslope',          'F0', 1, 'kg/s',       '이동 최소제곱 1차 기울기. B rolling_std_slope 그대로 (부호 보존)'),
    ('movmean_abs_e',     'F1', 1, 'kg',         '|e| 의 창 평균'),
    ('movrange',          'F1', 1, 'kg',         '창 내 최댓값 - 최솟값'),
    ('movmad',            'F1', 1, 'kg',         '창 평균에 대한 평균절대편차 mean(|x - mean(x)|)'),
    ('movkurt',           'F2', 0, '-',          '창 첨도 (excess, 정규 = 0)'),
    ('movskew',           'F2', 0, '-',          '창 왜도'),
    ('spec_centroid',     'F3', 0, 'Hz',         '창 평균 제거 + Hann 후 진폭스펙트럼의 1차 모멘트'),
    ('spec_rolloff85',    'F3', 0, 'Hz',         f'누적 진폭이 전체의 {ROLLOFF_FRACTION:.0%} 가 되는 최저 주파수'),
    ('spec_bandwidth',    'F3', 0, 'Hz',         'centroid 기준 진폭스펙트럼의 2차 모멘트'),
    ('detrend_resid_std', 'F4', 1, 'kg',         '창 내 1차 추세 제거 후 잔차 표준편차 (분모 W-1)'),
    ('half_mean_diff',    'F4', 1, 'kg',         '창 후반 평균 - 전반 평균'),
    ('env_mean',          'F5', 1, 'kg',         '창 평균 제거 후 Hilbert 포락선의 창 평균'),
    ('env_std',           'F5', 1, 'kg',         'Hilbert 포락선의 창 표준편차'),
    ('phase_vel_std',     'F6', 0, 'rad/sample', 'std(diff(unwrap(angle(hilbert(x - mean)))))'),
    ('hurst',             'F7', 0, '-',          'R/S 기반 Hurst 지수 (블록 척도 log-log 회귀 기울기)'),
]
FEATURES = [f[0] for f in FEATURE_SPEC]

# 계열 -> 폴더 이름. 결과물을 "어떤 방식(feature 계열)에 대한 결과인지" 로 나누기 위한 단일 출처다.
# 03·04·05 단계도 같은 이름을 쓴다 (사용자 지시 2026-10-28).
FAMILY_DIR = {'F0': 'F0_baseline', 'F1': 'F1_amplitude', 'F2': 'F2_shape', 'F3': 'F3_spectral',
              'F4': 'F4_trend', 'F5': 'F5_envelope', 'F6': 'F6_phase', 'F7': 'F7_memory'}
CROSS_DIR = '_cross'                                                   # 계열에 속하지 않는 공통 산출물
FAMILY_COLS = {fam: [n for n, f, *_ in FEATURE_SPEC if f == fam] for fam in FAMILY_DIR}

# =============================================================================
# 함수
# =============================================================================


def rolling_std_slope(e, w):
    # 역할 : 길이 w 의 뒤쪽(causal) 이동창에서 표준편차와 선형추세 기울기를 계산한다.
    #        **B 저장소 `01_stat_ssd.py:114-125` 를 그대로 복사한 것이다** (F0 수치 일치가 승계 검증 조건).
    # 입력 : e (n,) run 1개의 오차 신호 / w 창 길이 (samples)
    # 출력 : (std (n,) kg, slope (n,) kg/s). 창이 아직 다 차지 않은 앞 w-1 개 시점은
    #        첫 완전창의 값으로 채운다 (창 길이가 달라도 채점 대상 시점 수가 같도록).
    std = np.array(pd.Series(e).rolling(w).std())
    j = np.arange(w) - (w - 1) / 2.0
    kern = j / (j ** 2).sum() / DT                                     # 최소제곱 기울기 가중치, kg/sample -> kg/s
    slope = np.empty(len(e))
    slope[w - 1:] = np.convolve(e, kern[::-1], mode='valid')
    std[:w - 1], slope[:w - 1] = std[w - 1], slope[w - 1]
    return std, slope


def hurst_rs(xw, scales):
    # 역할 : 창 행렬의 각 행에 대해 R/S 기반 Hurst 지수를 구한다.
    #        척도 n 마다 창을 길이 n 의 비겹침 블록으로 나눠 R/S 를 평균하고,
    #        log(R/S) 를 log(n) 에 회귀한 기울기를 Hurst 지수로 쓴다 (고전 R/S 추정).
    #        블록 표준편차가 0 인 블록(양자화로 완전 평탄한 구간)은 R/S 가 정의되지 않아 평균에서 제외한다.
    # 입력 : xw (m, W) 창 행렬 (행 = 시점) / scales 블록 길이 리스트 (W 를 나누어떨어지는 값)
    # 출력 : (hurst (m,), r2 (m,)) 지수와 log-log 적합의 결정계수(추정 안정성 보고용)
    m = xw.shape[0]
    log_rs = np.full((m, len(scales)), np.nan)
    for c, n in enumerate(scales):
        nb = xw.shape[1] // n
        y = xw[:, :nb * n].reshape(m, nb, n)
        z = np.cumsum(y - y.mean(axis=2, keepdims=True), axis=2)       # 평균 제거 후 누적합
        r = z.max(axis=2) - z.min(axis=2)                              # (m, nb) range
        s = y.std(axis=2, ddof=1)                                      # (m, nb) 블록 표준편차
        rs = np.where(s > 0, r / np.where(s > 0, s, 1.0), np.nan)
        with np.errstate(invalid='ignore'):
            mean_rs = np.nanmean(rs, axis=1)                           # 유효 블록만 평균
        log_rs[:, c] = np.where(mean_rs > 0, np.log(mean_rs), np.nan)
    # 척도별로 유효한 점만 써서 가중 없는 최소제곱 기울기와 결정계수를 구한다
    x = np.log(np.asarray(scales, dtype=float))
    ok = np.isfinite(log_rs)
    n_ok = ok.sum(axis=1)
    xm = np.where(ok, x, 0.0).sum(axis=1) / n_ok
    ym = np.where(ok, log_rs, 0.0).sum(axis=1) / n_ok
    dx = np.where(ok, x - xm[:, None], 0.0)
    dy = np.where(ok, log_rs - ym[:, None], 0.0)
    num, den = (dx * dy).sum(axis=1), (dx ** 2).sum(axis=1)
    sst = (dy ** 2).sum(axis=1)
    return num / den, np.where(sst > 0, num ** 2 / (den * sst), np.nan)


def window_features(e, w):
    # 역할 : run 1개의 e 에 길이 w 의 causal 이동창을 씌워 16개 feature 를 시점마다 산출한다.
    #        창이 다 차지 않은 앞 w-1 시점은 첫 완전창의 값으로 backfill 한다 (B 규약).
    # 입력 : e (n,) run 1개의 잔차 / w 창 길이 (samples)
    # 출력 : (out dict {feature 이름: (n,) 배열}, r2 (n,) F7 적합 결정계수)
    n = len(e)
    xw = sliding_window_view(e, w)                                     # (n-w+1, w) 행 i = e[i:i+w]
    mu = xw.mean(axis=1, keepdims=True)
    xc = xw - mu                                                       # 창 내 평균 제거 (DC 제거)
    out = {}

    # --- F0 진폭·추세 (B 함수 그대로. 자체 backfill 포함) ---
    out['movstd'], out['movslope'] = rolling_std_slope(e, w)

    # --- F1 진폭 ---
    out['movmean_abs_e'] = np.abs(xw).mean(axis=1)
    out['movrange'] = xw.max(axis=1) - xw.min(axis=1)
    out['movmad'] = np.abs(xc).mean(axis=1)

    # --- F2 분포형상 (excess 첨도 · 왜도) ---
    out['movkurt'] = kurtosis(xw, axis=1, fisher=KURTOSIS_FISHER, bias=KURTOSIS_BIAS)
    out['movskew'] = skew(xw, axis=1, bias=KURTOSIS_BIAS)

    # --- F3 스펙트럼 (평균 제거 + Hann -> rfft 진폭) ---
    # 평균을 제거하지 않으면 DC bin 이 전체 진폭의 85% 를 넘어 roll-off 가 전 구간 0 Hz 로 고정된다
    taper = np.hanning(w)
    mag = np.abs(np.fft.rfft((xc if SPEC_DETREND_MEAN else xw) * taper, axis=1))
    freq = np.fft.rfftfreq(w, d=DT)
    tot = mag.sum(axis=1)
    cen = (mag * freq).sum(axis=1) / tot
    out['spec_centroid'] = cen
    out['spec_rolloff85'] = freq[np.argmax(mag.cumsum(axis=1) >= ROLLOFF_FRACTION * tot[:, None], axis=1)]
    out['spec_bandwidth'] = np.sqrt((mag * (freq[None, :] - cen[:, None]) ** 2).sum(axis=1) / tot)

    # --- F4 추세/정상성 ---
    j = np.arange(w) - (w - 1) / 2.0
    slope_px = (xc * j).sum(axis=1) / (j ** 2).sum()                   # 창 내 최소제곱 기울기 (kg/sample)
    resid = xc - slope_px[:, None] * j
    out['detrend_resid_std'] = np.sqrt((resid ** 2).sum(axis=1) / (w - 1))
    out['half_mean_diff'] = xw[:, w // 2:].mean(axis=1) - xw[:, :w // 2].mean(axis=1)

    # --- F5 포락선 · F6 위상 (창 단위 Hilbert — causal 유지. 테이퍼는 쓰지 않는다) ---
    a = hilbert(xc, axis=1)
    env = np.abs(a)
    out['env_mean'] = env.mean(axis=1)
    out['env_std'] = env.std(axis=1, ddof=1)
    out['phase_vel_std'] = np.diff(np.unwrap(np.angle(a), axis=1), axis=1).std(axis=1, ddof=1)

    # --- F7 장기기억 ---
    scales = []
    s = w
    while s >= HURST_MIN_BLOCK:                                        # W, W/2, W/4, ... (나누어떨어지는 척도만)
        scales.append(s)
        s //= 2
    scales = sorted(scales)
    out['hurst'], r2_w = hurst_rs(xw, scales)

    # --- backfill : 앞 w-1 시점을 첫 완전창의 값으로 채운다 (F0 는 함수 안에서 이미 처리됨) ---
    for key in FEATURES:
        if key in ('movstd', 'movslope'):
            continue
        full = np.empty(n)
        full[w - 1:] = out[key]
        full[:w - 1] = out[key][0]
        out[key] = full
    r2 = np.empty(n)
    r2[w - 1:], r2[:w - 1] = r2_w, r2_w[0]
    return out, r2


# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + 잔차 e
# =============================================================================
t_all = time.time()
t0 = time.time()
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
df['e'] = df[T1] - df[TS]
lengths = df.groupby('run_id', sort=True).size()
print(f'[Step 1] rows = {len(df)}, runs = {df.run_id.nunique()}, '
      f'run 길이 {lengths.min()}~{lengths.max()}, e 범위 {df.e.min():.3f}~{df.e.max():.3f} kg '
      f'({time.time() - t0:.1f} s)')
assert lengths.min() > max(WINDOWS), 'run 길이가 최대 창 길이보다 짧다'

# =============================================================================
# Step 2. run별 · 창별 feature 산출
# =============================================================================
for d in list(FAMILY_DIR.values()) + [CROSS_DIR]:
    os.makedirs(f'{OUT_DIR}/{d}', exist_ok=True)
hurst_rows = []
for w in WINDOWS:
    t0 = time.time()
    parts = []
    for run_id, g in df.groupby('run_id', sort=True):
        e = g.e.to_numpy()
        feat, r2 = window_features(e, w)
        part = pd.DataFrame({'run_id': run_id, 'k': g.k.to_numpy(), **feat})
        parts.append(part)
        hurst_rows.append({'W': w, 'run_id': run_id, VS: g[VS].iloc[0],
                           'hurst_mean': feat['hurst'].mean(), 'hurst_std': feat['hurst'].std(ddof=1),
                           'hurst_min': feat['hurst'].min(), 'hurst_max': feat['hurst'].max(),
                           'fit_r2_mean': np.nanmean(r2), 'fit_r2_p05': np.nanpercentile(r2, 5)})
    feats = pd.concat(parts, ignore_index=True)
    n_bad = int(feats[FEATURES].isna().sum().sum())
    assert n_bad == 0, f'창 {w}: 결측 feature {n_bad} 개'
    # 계열마다 따로 저장한다 — 행 순서(run_id, k)는 모든 계열 파일에서 같다 (03 단계가 그대로 이어 붙인다)
    total_mb = 0.0
    for fam, dirname in FAMILY_DIR.items():
        out_path = f'{OUT_DIR}/{dirname}/features_W{w}.csv.gz'   # gzip csv. pandas 가 확장자로 알아서 처리한다
        assert not os.path.exists(out_path), f'이미 결과 파일이 있다 (덮어쓰기 금지): {out_path}'
        feats[['run_id', 'k'] + FAMILY_COLS[fam]].to_csv(out_path, index=False, float_format='%.6g')
        total_mb += os.path.getsize(out_path) / 1e6
    print(f'[Step 2] 창 {w:3d} ({w * DT:5.1f} s): {len(feats)} 시점 x {len(FEATURES)} feature '
          f'-> 계열 {len(FAMILY_DIR)}개 폴더 ({total_mb:.1f} MB, {time.time() - t0:.1f} s)')

# =============================================================================
# Step 3. 기술통계 · 정의표 · run별 진폭 분모 저장 (전부 라벨 미사용)
# =============================================================================
t0 = time.time()
rows = []
for fam, dirname in FAMILY_DIR.items():
    frows = []
    for w in WINDOWS:
        f = pd.read_csv(f'{OUT_DIR}/{dirname}/features_W{w}.csv.gz')
        for name in FAMILY_COLS[fam]:
            s = f[name]
            unit = next(u for n, _, _, u, _ in FEATURE_SPEC if n == name)
            frows.append({'W': w, 'feature': name, 'family': fam, 'unit': unit,
                          'mean': s.mean(), 'std': s.std(ddof=1), 'min': s.min(),
                          'q05': s.quantile(0.05), 'median': s.median(), 'q95': s.quantile(0.95),
                          'max': s.max(), 'n_unique': s.nunique()})
    pd.DataFrame(frows).to_csv(f'{OUT_DIR}/{dirname}/feature_summary.csv', index=False)
    rows += frows
pd.DataFrame(rows).to_csv(f'{OUT_DIR}/{CROSS_DIR}/feature_summary_all.csv', index=False)

# feature 정의표 — 계열 폴더 이름까지 적어 둔다 (03·04·05 가 이 표로 폴더를 찾는다)
pd.DataFrame([{'feature': n, 'family': fam, 'family_dir': FAMILY_DIR[fam],
               'amplitude_scale_exponent': e, 'unit': u, 'definition': d}
              for n, fam, e, u, d in FEATURE_SPEC]).to_csv(
    f'{OUT_DIR}/{CROSS_DIR}/feature_scaling.csv', index=False)

# run별 진폭 정규화 분모 — A-5 의 좌표 2개가 쓴다.
#   setting_value : `Film-Tension Setting Value (kg)` (run 내 상수)
#   mad_run       : run 전체 e 의 중위절대편차. **run 전체를 보므로 causal 하지 않다** (한계로 기록)
g = df.groupby('run_id', sort=True)
scales = g.agg(**{TS: (TS, 'first'), RS: (RS, 'first'), VS: (VS, 'first'), 'n': ('e', 'size')})
scales['mad_run'] = g.e.apply(lambda s: float(np.median(np.abs(s - np.median(s)))))
scales.reset_index().to_csv(f'{OUT_DIR}/{CROSS_DIR}/run_scales.csv', index=False)

hq = pd.DataFrame(hurst_rows)
hq.to_csv(f'{OUT_DIR}/{FAMILY_DIR["F7"]}/hurst_fit_quality.csv', index=False)   # F7 계열의 추정 품질
print(f'[Step 3] 기술통계·정의표·분모·F7 품질 저장 완료 ({time.time() - t0:.1f} s)')
print('[Step 3] F7 적합 품질 (창별 run 평균):')
print(hq.groupby('W')[['hurst_mean', 'hurst_std', 'fit_r2_mean', 'fit_r2_p05']].mean().round(4).to_string())
print(f'[Step 3] mad_run 범위 {scales.mad_run.min():.4f}~{scales.mad_run.max():.4f} kg')

# =============================================================================
# Step 4. hyperparameter 기록
# =============================================================================
cfg = {'SEED': SEED, 'DT': DT, 'WINDOWS': WINDOWS, 'W_MAIN': W_MAIN,
       'ROLLOFF_FRACTION': ROLLOFF_FRACTION, 'SPEC_DETREND_MEAN': SPEC_DETREND_MEAN,
       'SPEC_WINDOW_FUNCTION': 'hann', 'KURTOSIS_FISHER': KURTOSIS_FISHER, 'KURTOSIS_BIAS': KURTOSIS_BIAS,
       'HURST_MIN_BLOCK': HURST_MIN_BLOCK, 'HURST_SCALE_RULE': 'W, W/2, W/4, ... >= HURST_MIN_BLOCK',
       'WINDOW_INCOMPLETE_RULE': 'backfill with first complete window (B 승계)',
       'RESIDUAL': 'e = Film Tension #1 (kg) - Film-Tension Setting Value (kg)',
       'LABEL_USED': False, 'IN_PATH': IN_PATH, 'N_FEATURES': len(FEATURES),
       'FAMILY_DIR': str(FAMILY_DIR)}
pd.DataFrame({'name': list(cfg), 'value': [str(v) for v in cfg.values()]}).to_csv(
    f'{OUT_DIR}/{CROSS_DIR}/hyperparameters.csv', index=False)
print(f'[done] feature 산출 완료 - 창 {WINDOWS}, feature {len(FEATURES)}개 (총 {time.time() - t_all:.1f} s)')
