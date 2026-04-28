# MSDM5058 Project II full workflow (output-folder version)
# ---------------------------------------------------------
# What this version changes:
# 1. It does NOT call plt.show(). All figures are saved into outputs/figures.
# 2. All important intermediate tables are saved into outputs/tables.
# 3. A text log is saved into outputs/logs.
# 4. More middle-step outputs are generated to help with report writing.
#
# Requirements:
#   pip install yfinance pandas numpy scipy matplotlib
# Optional but recommended:
#   pip install openpyxl

import warnings
warnings.filterwarnings('ignore')

import math
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, gaussian_kde

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance first: pip install yfinance")

# ============================================================
# 0. USER SETTINGS
# ============================================================
RISKY_CANDIDATE = "AMZN"   # candidate only; final risky/safe labels are determined by training volatility
SAFE_CANDIDATE = "KO"     # candidate only
START_DATE = "2000-01-01"
END_DATE = None            # None = today
INITIAL_CAPITAL = 100000.0
DAILY_RF = 0.00001         # 0.001% daily, as required by the project
H_WINDOWS = [30, 100, 300, np.inf]
MA_WINDOWS = [30, 100, 300]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EPS_SCALE = 0.25           # epsilon = EPS_SCALE * std(training returns of risky stock)
AGGRESSIVE_G = 0.50
CONSERVATIVE_G = 0.20
CONDITIONAL_DENSITY_KIND = 'auto_parametric'   # choose between normal and logistic for stability
PLOT_DPI = 180
ROLLING_SHARPE_WINDOW = 60
ROLLING_VOL_WINDOW = 60
OUTPUT_ROOT = "outputs"   # folder automatically created
SAVE_FIG_PDF = False       # True -> also save PDF copies of figures
PRICE_FIELD = 'Adj Close'  # use split-adjusted close to avoid spurious jumps from stock splits
EXTREME_RETURN_ABS_LOG_THRESHOLD = 0.40
EXTREME_RETURN_FAIL_ON_RAW_CLOSE = True

# ============================================================
# 1. OUTPUT FOLDERS / LOGGING / HELPERS
# ============================================================

def ensure_output_dirs(root: str = OUTPUT_ROOT) -> Dict[str, Path]:
    root_path = Path(root)
    figs = root_path / "figures"
    tabs = root_path / "tables"
    logs = root_path / "logs"
    for p in [root_path, figs, tabs, logs]:
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root_path, "figures": figs, "tables": tabs, "logs": logs}


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("msdm5058_project2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def nice_plot_style() -> None:
    plt.rcParams['figure.dpi'] = PLOT_DPI
    plt.rcParams['axes.grid'] = True
    plt.rcParams['savefig.dpi'] = PLOT_DPI


def save_figure(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, bbox_inches='tight')
    if SAVE_FIG_PDF:
        pdf_path = out_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


def save_table(df: pd.DataFrame, out_dir: Path, filename: str, index: bool = True) -> None:
    csv_path = out_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=index, encoding='utf-8-sig')


def save_series(s: pd.Series, out_dir: Path, filename: str, index: bool = True) -> None:
    save_table(s.to_frame(name=s.name if s.name is not None else 'value'), out_dir, filename, index=index)


def log_and_print(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)


def make_summary_table(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)

# ============================================================
# 2. DATA DOWNLOAD AND PREPROCESSING
# ============================================================

def download_two_stocks(ticker1: str, ticker2: str, start: str, end: Optional[str] = None,
                        price_field: str = PRICE_FIELD) -> pd.DataFrame:
    data = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=False, progress=False)
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError('Unexpected yfinance output format.')

    available_fields = set(data.columns.get_level_values(0))
    if price_field not in available_fields:
        raise ValueError(f"Could not find '{price_field}' in yfinance output. Available fields: {sorted(available_fields)}")

    close = data[price_field].copy()
    close = close.dropna(how='any').copy()
    close.columns = [ticker1, ticker2]

    if len(close) < 4001:
        raise ValueError(f'Common aligned price history has only {len(close)} rows; need > 4000 days.')

    return close

def make_returns(close: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(close / close.shift(1))
    return ret.dropna().copy()


def split_3_to_1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = int(round(0.75 * n))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def split_prices_and_returns_consistently(close: pd.DataFrame, ret: Optional[pd.DataFrame] = None):
    """
    Keep the cleaned price series as the *full* price history, and define returns from it.
    If the first test price date is split_date, then:
      - training prices: dates < split_date
      - testing prices : dates >= split_date
      - training returns: return dates < split_date
      - testing returns : return dates >= split_date
    so that prices and returns are temporally self-consistent and
    n_return_rows = n_price_rows - 1.
    """
    close = close.copy()
    ret = make_returns(close) if ret is None else ret.copy()

    if len(ret) != len(close) - 1:
        raise ValueError(
            f'Price/return inconsistency: len(ret)={len(ret)} but len(close)-1={len(close)-1}.'
        )

    train_close, test_close = split_3_to_1(close)
    split_date = test_close.index[0]
    train_ret = ret.loc[ret.index < split_date].copy()
    test_ret = ret.loc[ret.index >= split_date].copy()
    return train_close, test_close, train_ret, test_ret, split_date


def decide_risky_safe(ret: pd.DataFrame) -> Tuple[str, str, pd.Series]:
    train_ret, _ = split_3_to_1(ret)
    vols = train_ret.std()
    risky = vols.idxmax()
    safe = vols.idxmin()
    if risky == safe:
        raise ValueError("Could not distinguish risky and safe stocks.")
    return risky, safe, vols


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.columns)
    out['count'] = df.count()
    out['mean'] = df.mean()
    out['std'] = df.std(ddof=1)
    out['min'] = df.min()
    out['25%'] = df.quantile(0.25)
    out['50%'] = df.quantile(0.50)
    out['75%'] = df.quantile(0.75)
    out['max'] = df.max()
    out['skew'] = df.skew()
    out['kurtosis'] = df.kurtosis()
    return out


def summarize_extreme_returns(ret: pd.DataFrame, threshold: float = EXTREME_RETURN_ABS_LOG_THRESHOLD) -> pd.DataFrame:
    rows = []
    for col in ret.columns:
        s = ret[col].dropna()
        extreme = s[np.abs(s) >= threshold]
        rows.append({
            'ticker': col,
            'threshold_abs_log_return': threshold,
            'count_extreme': int(extreme.count()),
            'max_abs_return': float(np.abs(s).max()) if len(s) else np.nan,
            'min_return': float(s.min()) if len(s) else np.nan,
            'max_return': float(s.max()) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def list_extreme_return_events(ret: pd.DataFrame, threshold: float = EXTREME_RETURN_ABS_LOG_THRESHOLD) -> pd.DataFrame:
    rows = []
    for col in ret.columns:
        s = ret[col].dropna()
        extreme = s[np.abs(s) >= threshold]
        for dt, val in extreme.items():
            rows.append({'date': dt, 'ticker': col, 'log_return': float(val), 'abs_log_return': float(abs(val))})
    if not rows:
        return pd.DataFrame(columns=['date', 'ticker', 'log_return', 'abs_log_return'])
    return pd.DataFrame(rows).sort_values(['abs_log_return', 'date'], ascending=[False, True])


def validate_price_and_return_quality(close: pd.DataFrame, ret: pd.DataFrame, price_field: str,
                                      threshold: float = EXTREME_RETURN_ABS_LOG_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (close <= 0).any().any():
        raise ValueError('Found non-positive prices, so log returns are not well-defined.')

    summary = summarize_extreme_returns(ret, threshold=threshold)
    events = list_extreme_return_events(ret, threshold=threshold)

    if price_field == 'Close' and EXTREME_RETURN_FAIL_ON_RAW_CLOSE and not events.empty:
        raise ValueError(
            'Extreme log returns were detected while using raw Close prices. '
            'This strongly suggests stock-split contamination. '
            'Switch PRICE_FIELD to "Adj Close" and rerun.'
        )
    return summary, events


def clean_price_history_by_direct_date_drop(close_raw: pd.DataFrame,
                                            target_ticker: str = RISKY_CANDIDATE,
                                            threshold: float = EXTREME_RETURN_ABS_LOG_THRESHOLD,
                                            price_field: str = PRICE_FIELD):
    ret_raw_before = make_returns(close_raw)
    close_aligned_before = close_raw.loc[ret_raw_before.index].copy()
    raw_summary, raw_events = validate_price_and_return_quality(close_aligned_before, ret_raw_before, price_field, threshold)

    target_events = raw_events[raw_events['ticker'] == target_ticker].copy() if not raw_events.empty else pd.DataFrame(columns=['date', 'ticker', 'log_return', 'abs_log_return'])
    dates_to_drop = pd.Index(target_events['date']).unique() if not target_events.empty else pd.Index([])

    removed_rows = close_raw.loc[close_raw.index.intersection(dates_to_drop)].copy()
    if removed_rows.empty:
        removed_rows = pd.DataFrame(columns=['removed_reason'] + list(close_raw.columns))
    else:
        removed_rows.insert(0, 'removed_reason', f'direct_drop_due_to_{target_ticker}_extreme_log_return')

    close_clean = close_raw.drop(index=dates_to_drop, errors='ignore').copy()
    ret_clean = make_returns(close_clean)
    close_aligned_clean = close_clean.loc[ret_clean.index].copy()
    clean_summary, clean_events = validate_price_and_return_quality(close_aligned_clean, ret_clean, price_field, threshold)

    return {
        'close_clean': close_clean,
        'ret_clean': ret_clean,
        'raw_summary': raw_summary,
        'raw_events': raw_events,
        'target_events': target_events,
        'removed_rows': removed_rows,
        'clean_summary': clean_summary,
        'clean_events': clean_events,
    }

# ============================================================
# 3. MARKOWITZ MINIMUM-RISK PORTFOLIO
# ============================================================

def min_var_weight_two_assets(var_risky: float, var_safe: float, cov_rs: float) -> float:
    denom = var_risky + var_safe - 2.0 * cov_rs
    if abs(denom) < 1e-14:
        return 0.5
    p = (var_safe - cov_rs) / denom
    return float(np.clip(p, 0.0, 1.0))


def rolling_min_var_frontier_stats(ret2: pd.DataFrame, risky: str, safe: str, h) -> pd.DataFrame:
    rows = []
    idxs = []
    for i in range(len(ret2)):
        if h is np.inf:
            window = ret2.iloc[:i+1]
        else:
            if i + 1 < h:
                rows.append({'var_r': np.nan, 'var_s': np.nan, 'cov_rs': np.nan,
                             'pJ': np.nan, 'sigmaJ': np.nan, 'sigma_risky': np.nan})
                idxs.append(ret2.index[i])
                continue
            window = ret2.iloc[i-h+1:i+1]
        if len(window) < 2:
            rows.append({'var_r': np.nan, 'var_s': np.nan, 'cov_rs': np.nan,
                         'pJ': np.nan, 'sigmaJ': np.nan, 'sigma_risky': np.nan})
            idxs.append(ret2.index[i])
            continue
        var_r = float(window[risky].var(ddof=1))
        var_s = float(window[safe].var(ddof=1))
        cov_rs = float(window[[risky, safe]].cov().loc[risky, safe])
        p = min_var_weight_two_assets(var_r, var_s, cov_rs)
        sigmaJ = portfolio_sigma_from_weight(p, var_r, var_s, cov_rs)
        sigma_risky = float(np.sqrt(max(var_r, 0.0)))
        rows.append({'var_r': var_r, 'var_s': var_s, 'cov_rs': cov_rs,
                     'pJ': p, 'sigmaJ': sigmaJ, 'sigma_risky': sigma_risky})
        idxs.append(ret2.index[i])
    return pd.DataFrame(rows, index=idxs)


def rolling_min_var_weights(ret2: pd.DataFrame, risky: str, safe: str, h) -> pd.Series:
    stats = rolling_min_var_frontier_stats(ret2, risky, safe, h)
    return stats['pJ'].rename(f'pJ_h_{h}')

def portfolio_from_dynamic_weights(close2: pd.DataFrame, ret2: pd.DataFrame, w_risky: pd.Series, risky: str, safe: str,
                                   rf_daily: float = DAILY_RF) -> pd.DataFrame:
    """
    Section 2 object construction.

    The project statement asks for S_J(t,h) itself and its Sharpe ratio. Therefore the
    displayed Sharpe must be computed from the return series of S_J(t,h), i.e.
        R_J(t,h) = log(S_J(t,h) / S_J(t-1,h)),
    instead of mixing in a separate self-financing portfolio-return object.

    We still keep the previous-day-weight portfolio return as an auxiliary diagnostic,
    but the main Sharpe shown in figures/tables is the Sharpe of S_J_exact.
    """
    aligned_w_close = w_risky.reindex(close2.index).copy()
    s_j_exact = (aligned_w_close * close2[risky] + (1.0 - aligned_w_close) * close2[safe]).rename('S_J_exact')
    if not s_j_exact.dropna().empty:
        s_j_norm = (s_j_exact / s_j_exact.dropna().iloc[0]).rename('S_J_normalized')
    else:
        s_j_norm = s_j_exact.rename('S_J_normalized')

    # Return series attached to the actual displayed S_J(t,h) object.
    s_j_return = np.log(s_j_exact / s_j_exact.shift(1)).rename('S_J_return')
    sharpe_roll = (((s_j_return.rolling(ROLLING_SHARPE_WINDOW).mean() - rf_daily) /
                    s_j_return.rolling(ROLLING_SHARPE_WINDOW).std(ddof=1)) * np.sqrt(252)).rename(f'rolling_sharpe_{ROLLING_SHARPE_WINDOW}d')

    # Keep the self-financing return proxy only as an auxiliary diagnostic.
    aligned_w_ret = w_risky.reindex(ret2.index).copy()
    w_prev = aligned_w_ret.shift(1)
    port_ret = (w_prev * ret2[risky] + (1.0 - w_prev) * ret2[safe]).dropna().rename('portfolio_return_aux')
    wealth_aux = np.exp(port_ret.cumsum()).rename('portfolio_wealth_aux')

    out = pd.concat([
        s_j_exact,
        s_j_norm,
        s_j_return,
        sharpe_roll,
        port_ret,
        wealth_aux,
        aligned_w_close.rename('weight_risky_t'),
        w_prev.rename('weight_risky_prevday'),
    ], axis=1)
    return out

def portfolio_sigma_from_weight(p: float, var_r: float, var_s: float, cov_rs: float) -> float:
    val = p**2 * var_r + (1-p)**2 * var_s + 2*p*(1-p)*cov_rs
    return float(np.sqrt(max(val, 0.0)))


def weight_from_target_sigma(target_sigma: float, var_r: float, var_s: float, cov_rs: float,
                             p_min: float = 0.0, p_max: float = 1.0) -> float:
    grid = np.linspace(p_min, p_max, 5000)
    sigs = np.array([portfolio_sigma_from_weight(p, var_r, var_s, cov_rs) for p in grid])
    idx = np.argmin(np.abs(sigs - target_sigma))
    return float(grid[idx])

# ============================================================
# 4. MA / MACD
# ============================================================

def add_moving_averages(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    df = pd.DataFrame({'price': price.copy()})
    for w in windows:
        df[f'SMA_{w}'] = price.rolling(w).mean()
        df[f'EMA_{w}'] = price.ewm(span=w, adjust=False).mean()
    return df


def add_macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line

    out = pd.DataFrame({
        'price': price,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'macd': macd,
        'signal': signal_line,
        'hist': hist
    })
    out['macd_above_signal'] = (out['macd'] > out['signal']).astype(int)
    out['macd_above_zero'] = (out['macd'] > 0).astype(int)
    out['cross_signal'] = out['macd_above_signal'].diff()
    out['cross_zero'] = out['macd_above_zero'].diff()
    return out

# ============================================================
# 5. PDF / CDF FITTING
# ============================================================

def empirical_cdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_sorted = np.sort(np.asarray(x))
    y = np.arange(1, len(x_sorted)+1) / len(x_sorted)
    return x_sorted, y


def logistic_cdf(x, x0, b):
    return 1.0 / (1.0 + np.exp(-b * (x - x0)))


def logistic_pdf(x, x0, b):
    L = logistic_cdf(x, x0, b)
    return b * L * (1.0 - L)


def fit_normal_and_logistic(x: pd.Series) -> Dict[str, float]:
    x = x.dropna().values
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    xs, F = empirical_cdf(x)

    p0 = [float(np.median(x)), float(1.0 / max(np.std(x, ddof=1), 1e-6))]
    try:
        popt, _ = curve_fit(logistic_cdf, xs, F, p0=p0, maxfev=20000)
        x0, b = float(popt[0]), float(abs(popt[1]))
    except Exception:
        x0 = float(np.median(x))
        b = float(1.0 / max(np.std(x, ddof=1), 1e-6))

    return {'mu': mu, 'sigma': sigma, 'x0': x0, 'b': b}

# ============================================================
# 6. DIGITIZATION + CONDITIONAL PDFS
# ============================================================

def digitize_returns(x: pd.Series, eps: float) -> pd.Series:
    y = pd.Series(index=x.index, dtype=object)
    y[x < -eps] = 'D'
    y[x > eps] = 'U'
    y[(x >= -eps) & (x <= eps)] = 'H'
    return y


@dataclass
class ConditionalDensity:
    kind: str
    params: Dict[str, float]
    kde: Optional[gaussian_kde] = None

    def pdf(self, xgrid: np.ndarray) -> np.ndarray:
        if self.kind == 'normal':
            sigma = max(self.params['sigma'], 1e-8)
            return norm.pdf(xgrid, loc=self.params['mu'], scale=sigma)
        if self.kind == 'logistic':
            return logistic_pdf(xgrid, self.params['x0'], self.params['b'])
        if self.kind == 'kde':
            vals = self.kde.evaluate(xgrid)
            return np.maximum(vals, 1e-12)
        raise ValueError('Unknown density kind')

    def cdf(self, xgrid: np.ndarray) -> np.ndarray:
        if self.kind == 'normal':
            sigma = max(self.params['sigma'], 1e-8)
            return norm.cdf(xgrid, loc=self.params['mu'], scale=sigma)
        if self.kind == 'logistic':
            return logistic_cdf(xgrid, self.params['x0'], self.params['b'])
        if self.kind == 'kde':
            vals = np.array([self.kde.integrate_box_1d(-np.inf, xx) for xx in xgrid])
            return np.clip(vals, 0.0, 1.0)
        raise ValueError('Unknown density kind')


def cdf_fit_rmse(x: np.ndarray, dist_kind: str) -> float:
    xs, F = empirical_cdf(x)
    if dist_kind == 'normal':
        mu = float(np.mean(x))
        sigma = max(float(np.std(x, ddof=1)), 1e-8)
        fitted = norm.cdf(xs, loc=mu, scale=sigma)
    elif dist_kind == 'logistic':
        pars = fit_normal_and_logistic(pd.Series(x))
        fitted = logistic_cdf(xs, pars['x0'], pars['b'])
    else:
        raise ValueError('Unsupported dist kind for RMSE.')
    return float(np.sqrt(np.mean((F - fitted) ** 2)))


def fit_conditional_density(x: pd.Series, kind: str = 'auto_parametric') -> ConditionalDensity:
    x = pd.Series(x).dropna().astype(float).values
    if len(x) == 0:
        return ConditionalDensity(kind='normal', params={'mu': 0.0, 'sigma': 1e-8})
    if len(x) < 5:
        kind = 'normal'

    if kind == 'auto_parametric':
        normal_rmse = cdf_fit_rmse(x, 'normal')
        logistic_rmse = cdf_fit_rmse(x, 'logistic')
        kind = 'logistic' if logistic_rmse < normal_rmse else 'normal'

    if kind == 'kde':
        try:
            if len(x) < 5 or np.std(x, ddof=1) < 1e-12:
                raise ValueError('Degenerate sample for KDE')
            kde = gaussian_kde(x)
            return ConditionalDensity(kind='kde', params={}, kde=kde)
        except Exception:
            kind = 'normal'
    if kind == 'normal':
        sigma = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        if sigma < 1e-12:
            sigma = 1e-8
        mu = float(np.mean(x)) if len(x) else 0.0
        return ConditionalDensity(kind='normal', params={'mu': mu, 'sigma': sigma})
    if kind == 'logistic':
        pars = fit_normal_and_logistic(pd.Series(x))
        return ConditionalDensity(kind='logistic', params={'x0': pars['x0'], 'b': pars['b']})
    raise ValueError('Unsupported conditional density kind')


def bayes_detector_setup(x_t: pd.Series, y_next: pd.Series, density_kind: str = 'auto_parametric'):
    common_idx = x_t.index.intersection(y_next.index)
    x_t = x_t.loc[common_idx]
    y_next = y_next.loc[common_idx]

    priors = y_next.value_counts(normalize=True).reindex(['D', 'U', 'H']).fillna(0.0).to_dict()
    cond = {}
    for lab in ['D', 'U', 'H']:
        x_sub = x_t[y_next == lab]
        cond[lab] = fit_conditional_density(x_sub, density_kind)
    return priors, cond


def bayes_predict_one(x: float, priors: Dict[str, float], cond: Dict[str, ConditionalDensity]) -> str:
    scores = {}
    for lab in ['D', 'U', 'H']:
        score = priors[lab] * cond[lab].pdf(np.array([x]))[0]
        scores[lab] = score
    return max(scores, key=scores.get)


def critical_points_bayes(xmin: float, xmax: float, priors, cond, ngrid: int = 4000):
    grid = np.linspace(xmin, xmax, ngrid)
    labels = np.array([bayes_predict_one(xx, priors, cond) for xx in grid])
    crit = []
    for i in range(1, len(grid)):
        if labels[i] != labels[i-1]:
            crit.append(grid[i])
    return np.array(crit), grid, labels

# ============================================================
# 8. ASSOCIATION RULES
# ============================================================

def make_rules(y: pd.Series, k: int = 5) -> pd.DataFrame:
    vals = y.dropna().tolist()
    rows = []
    total_windows = max(len(vals) - k, 0)
    if total_windows <= 0:
        return pd.DataFrame(columns=['antecedent', 'consequent', 'count_xy', 'count_x', 'support', 'confidence'])

    antecedent_counts = {}
    rule_counts = {}
    for i in range(k, len(vals)):
        ant = tuple(vals[i-k:i])
        cons = vals[i]
        antecedent_counts[ant] = antecedent_counts.get(ant, 0) + 1
        rule_counts[(ant, cons)] = rule_counts.get((ant, cons), 0) + 1

    for ant in itertools.product(['D', 'U', 'H'], repeat=k):
        for cons in ['D', 'U', 'H']:
            cxy = rule_counts.get((ant, cons), 0)
            cx = antecedent_counts.get(ant, 0)
            support = cxy / total_windows
            confidence = cxy / cx if cx > 0 else 0.0
            rows.append({
                'antecedent': ''.join(ant),
                'consequent': cons,
                'count_xy': cxy,
                'count_x': cx,
                'support': support,
                'confidence': confidence,
                'geom_mean': math.sqrt(max(support * confidence, 0.0)),
                'RPF': support * (confidence ** 2),
            })
    return pd.DataFrame(rows)




def add_rule_usefulness_columns(rules: pd.DataFrame, lambdas: List[float]) -> pd.DataFrame:
    out = rules.copy()
    for lam in lambdas:
        col = f'usefulness_lambda_{lam:.2f}'
        out[col] = (out['support'].clip(lower=0.0) ** (1.0 - lam)) * (out['confidence'].clip(lower=0.0) ** lam)
    return out


def summarize_best_lambda_rules(rules: pd.DataFrame, lambdas: List[float], top_n: int = 10) -> pd.DataFrame:
    enriched = add_rule_usefulness_columns(rules, lambdas)
    rows = []
    for lam in lambdas:
        col = f'usefulness_lambda_{lam:.2f}'
        top = enriched.sort_values([col, 'support', 'confidence'], ascending=False).head(top_n)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append({'lambda': lam, 'rank': rank, 'antecedent': row['antecedent'], 'consequent': row['consequent'],
                         'support': row['support'], 'confidence': row['confidence'], 'usefulness': row[col]})
    return pd.DataFrame(rows)


def lambda_sweep_analysis(rules: pd.DataFrame, lambdas: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_rows = []
    util_rows = []
    for lam in lambdas:
        util = (rules['support'].clip(lower=0.0).values ** (1.0 - float(lam))) * (rules['confidence'].clip(lower=0.0).values ** float(lam))
        tmp = rules.copy()
        tmp['lambda'] = float(lam)
        tmp['utility'] = util
        util_rows.append(tmp[['antecedent', 'consequent', 'support', 'confidence', 'lambda', 'utility']])
        top = tmp.sort_values(['utility', 'confidence', 'support'], ascending=False).iloc[0]
        top_rows.append({
            'lambda': float(lam),
            'antecedent': top['antecedent'],
            'consequent': top['consequent'],
            'support': float(top['support']),
            'confidence': float(top['confidence']),
            'utility': float(top['utility']),
        })
    return pd.DataFrame(top_rows), pd.concat(util_rows, ignore_index=True)
def choose_best_rule_signal(history5: Tuple[str, ...], rules: pd.DataFrame) -> Optional[str]:
    sub = rules[rules['antecedent'] == ''.join(history5)]
    if sub.empty:
        return None
    return sub.sort_values(['confidence', 'support'], ascending=False).iloc[0]['consequent']

# ============================================================
# 9. SIGNALS
# ============================================================

def macd_signal_today(macd_df: pd.DataFrame, t) -> int:
    row = macd_df.loc[t]
    if row['macd'] > row['signal'] and row['macd'] > 0:
        return 1
    if row['macd'] < row['signal'] and row['macd'] < 0:
        return -1
    return 0


def bayes_signal_today(x_today: float, priors, cond) -> int:
    pred = bayes_predict_one(float(x_today), priors, cond)
    return {'U': 1, 'D': -1, 'H': 0}[pred]


def rule_signal_today(history5: Tuple[str, ...], rules: pd.DataFrame) -> int:
    pred = choose_best_rule_signal(history5, rules)
    if pred is None:
        return 0
    return {'U': 1, 'D': -1, 'H': 0}[pred]


def combined_signal(macd_s: int, bayes_s: int, rule_s: int) -> int:
    score = macd_s + bayes_s + rule_s
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def confidence_score(macd_s: int, bayes_s: int, rule_s: int) -> int:
    return abs(macd_s + bayes_s + rule_s)


def build_signal_table(dates: pd.Index, ret_series: pd.Series, macd_df: pd.DataFrame,
                       priors, cond, y_all: pd.Series, rules: pd.DataFrame) -> pd.DataFrame:
    """
    Build an information-date signal table.

    IMPORTANT: every row is the signal that becomes available only *after* that day's close,
    because it uses X(t), MACD(t) and Y(t-4:t). Therefore it must be used for trading on the
    next trading day, not on the same day.
    """
    rows = []
    for dt in dates:
        macd_s = bayes_s = rule_s = 0
        if dt in ret_series.index and dt in macd_df.index:
            x_today = float(ret_series.loc[dt])
            macd_s = macd_signal_today(macd_df, dt)
            bayes_s = bayes_signal_today(x_today, priors, cond)
            if dt in y_all.index:
                loc = y_all.index.get_loc(dt)
                if loc >= 4:
                    hist5 = tuple(y_all.iloc[loc-4:loc+1].tolist())
                    rule_s = rule_signal_today(hist5, rules)
        combined = combined_signal(macd_s, bayes_s, rule_s)
        conf = confidence_score(macd_s, bayes_s, rule_s)
        rows.append({
            'date': dt,
            'macd_signal': macd_s,
            'bayes_signal': bayes_s,
            'rule_signal': rule_s,
            'combined_signal': combined,
            'confidence_score': conf,
        })
    return pd.DataFrame(rows).set_index('date')


def build_cash_benchmark(dates: pd.Index, initial_capital: float = INITIAL_CAPITAL,
                         rf_daily: float = DAILY_RF) -> pd.Series:
    """
    Benchmark with explicit t=0 alignment:
    - first test date is treated as t=0, value = initial_capital
    - from the second test date onward, cash compounds once per day
    """
    out = pd.Series(index=dates, dtype=float, name='cash_benchmark')
    if len(dates) == 0:
        return out
    out.iloc[0] = initial_capital
    for i in range(1, len(dates)):
        out.iloc[i] = out.iloc[i-1] * (1.0 + rf_daily)
    return out


def build_dynamic_minrisk_benchmark(test_close: pd.DataFrame, risky: str, safe: str,
                                    pJ_available: pd.Series, initial_capital: float = INITIAL_CAPITAL,
                                    initial_pJ: Optional[float] = None) -> pd.Series:
    dates = test_close.index
    wealth = pd.Series(index=dates, dtype=float, name='base_minrisk')
    if len(dates) == 0:
        return wealth

    p0 = float(initial_pJ) if initial_pJ is not None else (float(pJ_available.iloc[0]) if len(pJ_available) else 0.5)
    if np.isnan(p0):
        p0 = 0.5

    pr0 = float(test_close[risky].iloc[0])
    ps0 = float(test_close[safe].iloc[0])
    Nr = p0 * initial_capital / pr0
    Ns = (1.0 - p0) * initial_capital / ps0
    wealth.iloc[0] = initial_capital

    for i in range(1, len(dates)):
        dt = dates[i]
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        V = Nr * pr + Ns * ps
        p_ref = float(pJ_available.loc[dt]) if dt in pJ_available.index else np.nan
        if not np.isnan(p_ref):
            Nr = (p_ref * V) / pr
            Ns = ((1.0 - p_ref) * V) / ps
            V = Nr * pr + Ns * ps
        wealth.iloc[i] = V
    return wealth

def backtest_one_stock_money(test_price: pd.Series, signal_table: pd.DataFrame,
                             g: float, initial_capital: float = INITIAL_CAPITAL,
                             rf_daily: float = DAILY_RF) -> pd.DataFrame:
    """
    Section 7 backtest with strict timing:
    - first test date is t=0: initial state only, no trade
    - for test date t_i (i>=1):
        1) cash accrues interest at the beginning of the day
        2) trade uses the signal generated from the previous test date t_{i-1}
        3) trade is executed at the current day's price
    """
    dates = test_price.index
    M = initial_capital
    N = 0.0
    out = []

    for i, dt in enumerate(dates):
        price = float(test_price.loc[dt])
        if i == 0:
            V = M + N * price
            out.append({
                'date': dt, 'signal_source_date': pd.NaT, 'money': M, 'shares': N,
                'price': price, 'wealth': V,
                'macd_signal': 0, 'bayes_signal': 0, 'rule_signal': 0,
                'action': 0, 'g': 0.0, 'trade_cash': 0.0, 'trade_shares': 0.0
            })
            continue

        M *= (1.0 + rf_daily)
        prev_dt = dates[i-1]
        sig_row = signal_table.loc[prev_dt] if prev_dt in signal_table.index else None
        macd_s = int(sig_row['macd_signal']) if sig_row is not None else 0
        bayes_s = int(sig_row['bayes_signal']) if sig_row is not None else 0
        rule_s = int(sig_row['rule_signal']) if sig_row is not None else 0
        action = int(sig_row['combined_signal']) if sig_row is not None else 0

        trade_cash = 0.0
        trade_shares = 0.0
        if action > 0 and M > 0:
            spend = g * M
            shares = spend / price
            M -= spend
            N += shares
            trade_cash = -spend
            trade_shares = shares
        elif action < 0 and N > 0:
            sell = g * N
            cash = sell * price
            M += cash
            N -= sell
            trade_cash = cash
            trade_shares = -sell

        V = M + N * price
        out.append({
            'date': dt, 'signal_source_date': prev_dt, 'money': M, 'shares': N, 'price': price, 'wealth': V,
            'macd_signal': macd_s, 'bayes_signal': bayes_s, 'rule_signal': rule_s,
            'action': action, 'g': g, 'trade_cash': trade_cash, 'trade_shares': trade_shares
        })
    return pd.DataFrame(out).set_index('date')


def backtest_bob_dynamic_greed(test_price: pd.Series, signal_table: pd.DataFrame,
                               g_aggr: float, g_cons: float,
                               initial_capital: float = INITIAL_CAPITAL,
                               rf_daily: float = DAILY_RF) -> pd.DataFrame:
    dates = test_price.index
    M = initial_capital
    N = 0.0
    out = []

    for i, dt in enumerate(dates):
        price = float(test_price.loc[dt])
        if i == 0:
            V = M + N * price
            out.append({
                'date': dt, 'signal_source_date': pd.NaT, 'money': M, 'shares': N,
                'price': price, 'wealth': V,
                'macd_signal': 0, 'bayes_signal': 0, 'rule_signal': 0,
                'action': 0, 'g_today': 0.0, 'g_reason': 'initial_no_trade',
                'trade_cash': 0.0, 'trade_shares': 0.0
            })
            continue

        M *= (1.0 + rf_daily)
        prev_dt = dates[i-1]
        sig_row = signal_table.loc[prev_dt] if prev_dt in signal_table.index else None
        macd_s = int(sig_row['macd_signal']) if sig_row is not None else 0
        bayes_s = int(sig_row['bayes_signal']) if sig_row is not None else 0
        rule_s = int(sig_row['rule_signal']) if sig_row is not None else 0
        action = int(sig_row['combined_signal']) if sig_row is not None else 0

        if action > 0:
            g_today = g_aggr
            g_reason = 'buy_use_aggressive_g'
        elif action < 0:
            g_today = g_cons
            g_reason = 'sell_use_conservative_g'
        else:
            g_today = 0.0
            g_reason = 'hold_no_trade'

        trade_cash = 0.0
        trade_shares = 0.0
        if action > 0 and M > 0:
            spend = g_today * M
            buy = spend / price
            N += buy
            M -= spend
            trade_cash = -spend
            trade_shares = buy
        elif action < 0 and N > 0:
            sell = g_today * N
            cash = sell * price
            M += cash
            N -= sell
            trade_cash = cash
            trade_shares = -sell

        V = M + N * price
        out.append({'date': dt, 'signal_source_date': prev_dt, 'money': M, 'shares': N, 'price': price, 'wealth': V,
                    'macd_signal': macd_s, 'bayes_signal': bayes_s, 'rule_signal': rule_s,
                    'action': action, 'g_today': g_today, 'g_reason': g_reason,
                    'trade_cash': trade_cash, 'trade_shares': trade_shares})
    return pd.DataFrame(out).set_index('date')


def summarize_bob_policy(bt_bob: pd.DataFrame, g_aggr: float, g_cons: float) -> pd.DataFrame:
    if bt_bob.empty:
        return pd.DataFrame(columns=['metric', 'value'])
    rows = [
        {'metric': 'aggressive_g_value', 'value': g_aggr},
        {'metric': 'conservative_g_value', 'value': g_cons},
        {'metric': 'n_buy_days_using_aggressive_g', 'value': int(((bt_bob['action'] > 0) & np.isclose(bt_bob['g_today'], g_aggr)).sum())},
        {'metric': 'n_sell_days_using_conservative_g', 'value': int(((bt_bob['action'] < 0) & np.isclose(bt_bob['g_today'], g_cons)).sum())},
        {'metric': 'n_hold_days', 'value': int((bt_bob['action'] == 0).sum())},
        {'metric': 'final_wealth', 'value': float(bt_bob['wealth'].iloc[-1])},
    ]
    return pd.DataFrame(rows)


def summarize_section41(eps: float, y_train: pd.Series, priors: Dict[str, float], crit: np.ndarray) -> pd.DataFrame:
    counts = y_train.value_counts().reindex(['D', 'U', 'H']).fillna(0).astype(int)
    return pd.DataFrame([
        {'item': 'epsilon', 'value': float(eps)},
        {'item': 'count_D', 'value': int(counts.get('D', 0))},
        {'item': 'count_U', 'value': int(counts.get('U', 0))},
        {'item': 'count_H', 'value': int(counts.get('H', 0))},
        {'item': 'prior_D', 'value': float(priors.get('D', 0.0))},
        {'item': 'prior_U', 'value': float(priors.get('U', 0.0))},
        {'item': 'prior_H', 'value': float(priors.get('H', 0.0))},
        {'item': 'n_bayes_critical_points', 'value': int(len(crit))},
    ])


def summarize_section61(lambda_top: pd.DataFrame, lambda_distinct: pd.DataFrame) -> pd.DataFrame:
    if lambda_top.empty:
        return pd.DataFrame(columns=['item', 'value'])
    best_row = lambda_top.sort_values(['utility', 'lambda'], ascending=[False, True]).iloc[0]
    most_stable = lambda_distinct.iloc[0]['rule'] if not lambda_distinct.empty else ''
    return pd.DataFrame([
        {'item': 'lambda_with_highest_top_utility', 'value': float(best_row['lambda'])},
        {'item': 'highest_top_utility', 'value': float(best_row['utility'])},
        {'item': 'best_rule_at_that_lambda', 'value': f"{best_row['antecedent']}->{best_row['consequent']}"},
        {'item': 'n_distinct_top_rules_across_lambda_grid', 'value': int(lambda_distinct.shape[0])},
        {'item': 'most_stable_top_rule', 'value': most_stable},
        {'item': 'comment', 'value': 'No single universal lambda is guaranteed; usefulness depends on the trade-off between frequency and accuracy.'},
    ])


def summarize_section7(base_cash: pd.Series, bt_aggr: pd.DataFrame, bt_cons: pd.DataFrame, bt_bob: pd.DataFrame) -> pd.DataFrame:
    final_base = float(base_cash.iloc[-1]) if len(base_cash) else np.nan
    finals = {
        'cash_benchmark': final_base,
        'aggressive': float(bt_aggr['wealth'].iloc[-1]),
        'conservative': float(bt_cons['wealth'].iloc[-1]),
        'bob': float(bt_bob['wealth'].iloc[-1]),
    }
    best_strategy = max(finals, key=finals.get) if finals else ''
    return pd.DataFrame([
        {'item': 'final_cash_benchmark', 'value': final_base},
        {'item': 'final_aggressive', 'value': finals['aggressive']},
        {'item': 'final_conservative', 'value': finals['conservative']},
        {'item': 'final_bob', 'value': finals['bob']},
        {'item': 'best_strategy', 'value': best_strategy},
        {'item': 'bob_minus_aggressive', 'value': finals['bob'] - finals['aggressive']},
        {'item': 'bob_minus_conservative', 'value': finals['bob'] - finals['conservative']},
        {'item': 'aggressive_minus_benchmark', 'value': finals['aggressive'] - final_base},
        {'item': 'conservative_minus_benchmark', 'value': finals['conservative'] - final_base},
        {'item': 'bob_minus_benchmark', 'value': finals['bob'] - final_base},
    ])


def summarize_section81(base_minrisk: pd.Series, bt2ef_aggr: pd.DataFrame, bt2ef_cons: pd.DataFrame) -> pd.DataFrame:
    final_base = float(base_minrisk.iloc[-1]) if len(base_minrisk) else np.nan
    final_aggr = float(bt2ef_aggr['wealth'].iloc[-1])
    final_cons = float(bt2ef_cons['wealth'].iloc[-1])
    below_aggr = int((bt2ef_aggr['p_curr'] + 1e-12 < bt2ef_aggr['pJ_t']).sum())
    below_cons = int((bt2ef_cons['p_curr'] + 1e-12 < bt2ef_cons['pJ_t']).sum())
    return pd.DataFrame([
        {'item': 'final_minrisk_benchmark', 'value': final_base},
        {'item': 'final_aggressive', 'value': final_aggr},
        {'item': 'final_conservative', 'value': final_cons},
        {'item': 'aggressive_minus_benchmark', 'value': final_aggr - final_base},
        {'item': 'conservative_minus_benchmark', 'value': final_cons - final_base},
        {'item': 'n_days_aggressive_p_below_pJ', 'value': below_aggr},
        {'item': 'n_days_conservative_p_below_pJ', 'value': below_cons},
        {'item': 'comment', 'value': 'Efficient-frontier control is considered clean only when p(t) stays at or above p_J(t).'},
    ])


def summarize_section9(bt3_aggr: pd.DataFrame, bt3_cons: pd.DataFrame) -> pd.DataFrame:
    final_aggr = float(bt3_aggr['wealth'].iloc[-1])
    final_cons = float(bt3_cons['wealth'].iloc[-1])
    better = 'aggressive' if final_aggr > final_cons else ('conservative' if final_cons > final_aggr else 'tie')
    return pd.DataFrame([
        {'item': 'final_aggressive', 'value': final_aggr},
        {'item': 'final_conservative', 'value': final_cons},
        {'item': 'better_strategy', 'value': better},
        {'item': 'difference_aggressive_minus_conservative', 'value': final_aggr - final_cons},
    ])


def backtest_two_stocks_original(test_close: pd.DataFrame, risky: str, safe: str,
                                 signal_series: pd.Series, pJ0: float, g: float,
                                 initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    dates = test_close.index
    pr0 = float(test_close[risky].iloc[0])
    ps0 = float(test_close[safe].iloc[0])
    Nr = pJ0 * initial_capital / pr0
    Ns = (1.0 - pJ0) * initial_capital / ps0
    out = []

    for i, dt in enumerate(dates):
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        if i == 0:
            V = Nr * pr + Ns * ps
            p_curr = (Nr * pr) / V if V > 0 else np.nan
            out.append({'date': dt, 'signal_source_date': pd.NaT, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V,
                        'p_curr': p_curr, 'signal': 0, 'trade_risky': 0.0, 'trade_safe': 0.0})
            continue

        prev_dt = dates[i-1]
        sig = int(signal_series.get(prev_dt, 0))
        trade_risky = 0.0
        trade_safe = 0.0
        if sig > 0 and Ns > 0:
            sell_safe = g * Ns
            cash_equiv = sell_safe * ps
            buy_risky = cash_equiv / pr
            Ns -= sell_safe
            Nr += buy_risky
            trade_safe = -sell_safe
            trade_risky = buy_risky
        elif sig < 0 and Nr > 0:
            sell_risky = g * Nr
            cash_equiv = sell_risky * pr
            buy_safe = cash_equiv / ps
            Nr -= sell_risky
            Ns += buy_safe
            trade_risky = -sell_risky
            trade_safe = buy_safe
        V = Nr * pr + Ns * ps
        p_curr = (Nr * pr) / V if V > 0 else np.nan
        out.append({'date': dt, 'signal_source_date': prev_dt, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V,
                    'p_curr': p_curr, 'signal': sig,
                    'trade_risky': trade_risky, 'trade_safe': trade_safe})
    return pd.DataFrame(out).set_index('date')


def backtest_two_stocks_efficient_frontier(test_close: pd.DataFrame, risky: str, safe: str,
                                           signal_series: pd.Series, frontier_stats_available_test: pd.DataFrame,
                                           g: float, pJ0: float,
                                           initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    dates = test_close.index
    pr0 = float(test_close[risky].iloc[0])
    ps0 = float(test_close[safe].iloc[0])
    Nr = pJ0 * initial_capital / pr0
    Ns = (1.0 - pJ0) * initial_capital / ps0
    out = []

    for i, dt in enumerate(dates):
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        V = Nr * pr + Ns * ps
        p_curr = (Nr * pr) / V if V > 0 else 0.0

        stats_row = frontier_stats_available_test.loc[dt] if dt in frontier_stats_available_test.index else None
        pJ_t = float(stats_row['pJ']) if stats_row is not None and pd.notna(stats_row['pJ']) else np.nan
        sigmaJ_t = float(stats_row['sigmaJ']) if stats_row is not None and pd.notna(stats_row['sigmaJ']) else np.nan
        var_r_t = float(stats_row['var_r']) if stats_row is not None and pd.notna(stats_row['var_r']) else np.nan
        var_s_t = float(stats_row['var_s']) if stats_row is not None and pd.notna(stats_row['var_s']) else np.nan
        cov_t = float(stats_row['cov_rs']) if stats_row is not None and pd.notna(stats_row['cov_rs']) else np.nan
        sigma_risky_t = float(stats_row['sigma_risky']) if stats_row is not None and pd.notna(stats_row['sigma_risky']) else np.nan

        if i == 0:
            out.append({'date': dt, 'signal_source_date': pd.NaT, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V, 'p_curr': p_curr,
                        'signal': 0, 'pJ_t': pJ_t, 'sigmaJ_t': sigmaJ_t, 'sigma_curr': np.nan, 'sigma_risky_t': sigma_risky_t,
                        'target_sigma': np.nan, 'p_target': np.nan, 'p_floor_applied': int(False)})
            continue

        prev_dt = dates[i-1]
        sig = int(signal_series.get(prev_dt, 0))
        p_target = np.nan
        target_sigma = np.nan
        sigma_curr = np.nan
        p_floor_applied = False

        if not np.isnan(pJ_t) and not np.isnan(sigmaJ_t) and not np.isnan(var_r_t) and not np.isnan(var_s_t) and not np.isnan(cov_t) and not np.isnan(sigma_risky_t):
            p_floor = float(np.clip(pJ_t, 0.0, 1.0))
            sigma_curr = portfolio_sigma_from_weight(p_curr, var_r_t, var_s_t, cov_t)

            if sig < 0:
                target_sigma = sigma_curr - g * (sigma_curr - sigmaJ_t)
            elif sig > 0:
                target_sigma = sigma_curr + g * (sigma_risky_t - sigma_curr)
            else:
                target_sigma = sigma_curr

            target_sigma = float(np.clip(target_sigma, min(sigmaJ_t, sigma_risky_t), max(sigmaJ_t, sigma_risky_t)))

            # Convert target risk to a risky-stock weight on the frontier.
            if sig == 0:
                # Even on hold days, if updated estimation makes p_J(t) drift above the current weight,
                # rebalance up to p_J(t) so that p(t) does not fall below the minimum-risk fraction.
                p_target = max(p_curr, p_floor)
            else:
                p_target = weight_from_target_sigma(target_sigma, var_r_t, var_s_t, cov_t, p_min=p_floor, p_max=1.0)
                p_target = max(p_target, p_floor)

            p_target = float(np.clip(p_target, p_floor, 1.0))
            p_floor_applied = bool(p_target <= p_floor + 1e-12)

            Nr = (p_target * V) / pr
            Ns = ((1.0 - p_target) * V) / ps
            V = Nr * pr + Ns * ps
            p_curr = (Nr * pr) / V if V > 0 else np.nan

        out.append({'date': dt, 'signal_source_date': prev_dt, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V, 'p_curr': p_curr,
                    'signal': sig, 'pJ_t': pJ_t, 'sigmaJ_t': sigmaJ_t, 'sigma_curr': sigma_curr, 'sigma_risky_t': sigma_risky_t,
                    'target_sigma': target_sigma, 'p_target': p_target, 'p_floor_applied': int(p_floor_applied)})
    return pd.DataFrame(out).set_index('date')

def backtest_two_stocks_money(test_close: pd.DataFrame, risky: str, safe: str,
                              signal_risky: pd.Series, signal_safe: pd.Series,
                              g: float, initial_capital: float = INITIAL_CAPITAL,
                              rf_daily: float = DAILY_RF) -> pd.DataFrame:
    dates = test_close.index
    M = initial_capital
    Nr = 0.0
    Ns = 0.0
    out = []
    for i, dt in enumerate(dates):
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        if i == 0:
            V = M + Nr * pr + Ns * ps
            out.append({'date': dt, 'signal_source_date': pd.NaT, 'money': M, 'N_risky': Nr, 'N_safe': Ns,
                        'wealth': V, 'signal_risky': 0, 'signal_safe': 0,
                        'trade_risky': 0.0, 'trade_safe': 0.0,
                        'trade_cash_risky': 0.0, 'trade_cash_safe': 0.0})
            continue

        M *= (1.0 + rf_daily)
        prev_dt = dates[i-1]
        sr = int(signal_risky.get(prev_dt, 0))
        ss = int(signal_safe.get(prev_dt, 0))

        trade_cash_risky = 0.0
        trade_cash_safe = 0.0
        trade_risky = 0.0
        trade_safe = 0.0

        if sr < 0 and Nr > 0:
            sell = g * Nr
            cash = sell * pr
            M += cash
            Nr -= sell
            trade_cash_risky += cash
            trade_risky -= sell
        if ss < 0 and Ns > 0:
            sell = g * Ns
            cash = sell * ps
            M += cash
            Ns -= sell
            trade_cash_safe += cash
            trade_safe -= sell

        if sr > 0 and M > 0:
            spend = g * M if ss <= 0 else 0.5 * g * M
            buy = spend / pr
            Nr += buy
            M -= spend
            trade_cash_risky -= spend
            trade_risky += buy
        if ss > 0 and M > 0:
            spend = g * M if sr <= 0 else 0.5 * g * M
            buy = spend / ps
            Ns += buy
            M -= spend
            trade_cash_safe -= spend
            trade_safe += buy

        V = M + Nr * pr + Ns * ps
        out.append({'date': dt, 'signal_source_date': prev_dt, 'money': M, 'N_risky': Nr, 'N_safe': Ns,
                    'wealth': V, 'signal_risky': sr, 'signal_safe': ss,
                    'trade_risky': trade_risky, 'trade_safe': trade_safe,
                    'trade_cash_risky': trade_cash_risky, 'trade_cash_safe': trade_cash_safe})
    return pd.DataFrame(out).set_index('date')

# ============================================================
# 11. PLOTTING FUNCTIONS (ALL SAVE TO FILES)
# ============================================================

def plot_prices_returns_split(close: pd.DataFrame, ret: pd.DataFrame, risky: str, safe: str,
                              split_date, out_dir: Path, price_field: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    close[[risky, safe]].plot(ax=axes[0])
    axes[0].axvline(split_date, linestyle='--', color='k', alpha=0.8)
    axes[0].set_title(f'{price_field} Prices with Train/Test Split')

    ret[[risky, safe]].plot(ax=axes[1])
    axes[1].axvline(split_date, linestyle='--', color='k', alpha=0.8)
    axes[1].set_title('Daily Log Returns with Train/Test Split')
    fig.tight_layout()
    save_figure(fig, out_dir, '01_prices_returns_split')


def plot_return_histograms(train_ret: pd.DataFrame, risky: str, safe: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train_ret[risky].dropna(), bins=60)
    axes[0].set_title(f'Training Return Histogram: {risky}')
    axes[1].hist(train_ret[safe].dropna(), bins=60)
    axes[1].set_title(f'Training Return Histogram: {safe}')
    fig.tight_layout()
    save_figure(fig, out_dir, '02_train_return_histograms')


def plot_rolling_volatility(ret: pd.DataFrame, risky: str, safe: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ret[risky].rolling(ROLLING_VOL_WINDOW).std(ddof=1).plot(ax=ax, label=f'{risky} rolling vol')
    ret[safe].rolling(ROLLING_VOL_WINDOW).std(ddof=1).plot(ax=ax, label=f'{safe} rolling vol')
    ax.set_title(f'Rolling Volatility ({ROLLING_VOL_WINDOW}-day)')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '03_rolling_volatility')


def plot_pJ(all_pJ: Dict[str, pd.Series], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, s in all_pJ.items():
        s.plot(ax=ax, label=label)
    ax.set_title('Minimum-risk Weight $p_J(t,h)$')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '04_pJ_curves')


def plot_portfolios(portfolios: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    sharpe_col = f'rolling_sharpe_{ROLLING_SHARPE_WINDOW}d'
    for name, df in portfolios.items():
        if 'S_J_normalized' in df.columns:
            df['S_J_normalized'].dropna().plot(ax=axes[0], label=name)
        if sharpe_col in df.columns:
            df[sharpe_col].dropna().plot(ax=axes[1], label=name)
    axes[0].set_title('Minimum-risk Portfolio $S_J(t,h)$ (normalized)')
    axes[1].set_title(f'Rolling Sharpe Ratio ({ROLLING_SHARPE_WINDOW} days)')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '05_portfolio_SJ_and_sharpe')

def plot_ma_macd(ma_df: pd.DataFrame, macd_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cols = [c for c in ma_df.columns if c != 'price']
    ma_df[['price'] + cols].plot(ax=axes[0])
    axes[0].set_title('Price with SMA and EMA')
    macd_df[['macd', 'signal', 'hist']].plot(ax=axes[1])
    axes[1].set_title('MACD, Signal, Histogram')
    fig.tight_layout()
    save_figure(fig, out_dir, '06_ma_macd')


def plot_pdf_cdf_fit(x: pd.Series, pars: Dict[str, float], out_dir: Path) -> None:
    x = x.dropna().values
    xs, F = empirical_cdf(x)
    grid = np.linspace(np.min(xs), np.max(xs), 800)
    G = norm.cdf(grid, loc=pars['mu'], scale=max(pars['sigma'], 1e-8))
    L = logistic_cdf(grid, pars['x0'], pars['b'])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, F, label='Empirical CDF')
    ax.plot(grid, G, label='Normal CDF fit')
    ax.plot(grid, L, label='Logistic CDF fit')
    ax.set_title('CDF Fits for Risky-stock Returns')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '07_cdf_fits')


def plot_conditional_cdf_pdf(x_t: pd.Series, y_next: pd.Series, cond: Dict[str, ConditionalDensity], out_dir: Path) -> None:
    common_idx = x_t.index.intersection(y_next.index)
    x_t = x_t.loc[common_idx]
    y_next = y_next.loc[common_idx]
    xmin, xmax = float(x_t.min()), float(x_t.max())
    grid = np.linspace(xmin, xmax, 800)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    for lab in ['D', 'U', 'H']:
        xs, F = empirical_cdf(x_t[y_next == lab].values)
        axes[0].plot(xs, F, label=f'Empirical CDF | {lab}')
        axes[0].plot(grid, cond[lab].cdf(grid), '--', label=f'Fit CDF | {lab}')
        axes[1].plot(grid, cond[lab].pdf(grid), label=f'PDF | {lab}')
    axes[0].set_title('Conditional CDFs')
    axes[1].set_title('Conditional PDFs')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '08_conditional_cdf_pdf')


def plot_bayes_boundaries(priors, cond, x_t: pd.Series, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax = float(x_t.min()), float(x_t.max())
    crit, grid, labels = critical_points_bayes(xmin, xmax, priors, cond)
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab in ['D', 'U', 'H']:
        ax.plot(grid, cond[lab].pdf(grid), label=f'PDF | {lab}')
    for c in crit:
        ax.axvline(c, linestyle='--', alpha=0.5, color='k')
    ax.set_title('Conditional PDFs with Bayes Critical Points')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '09_bayes_boundaries')
    return crit, grid, labels

def plot_digitized_counts(y: pd.Series, out_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    y.value_counts().reindex(['D', 'U', 'H']).fillna(0).plot(kind='bar', ax=ax)
    ax.set_title('Digitized Class Counts')
    fig.tight_layout()
    save_figure(fig, out_dir, filename)


def plot_lambda_sweep(lambda_top: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lambda_top['lambda'], lambda_top['utility'], marker='o')
    ax.set_title(r'Section 6.1: Top-rule Utility versus $\lambda$')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Top utility')
    fig.tight_layout()
    save_figure(fig, out_dir, '12_lambda_sweep_top_utility')


def plot_backtests_one_stock(base_curve: pd.Series, bt_aggr: pd.DataFrame,
                             bt_cons: pd.DataFrame, bt_bob: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    base_curve.plot(ax=ax, label='Cash Benchmark')
    bt_aggr['wealth'].plot(ax=ax, label='Aggressive Greed')
    bt_cons['wealth'].plot(ax=ax, label='Conservative Greed')
    bt_bob['wealth'].plot(ax=ax, label='Bob Dynamic Greed')
    ax.set_title('Section 7 Backtest: One Stock + Money')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '10_section7_backtest')


def plot_backtests_two_stock(base_curve: pd.Series, bt_aggr: pd.DataFrame,
                             bt_cons: pd.DataFrame, title: str, out_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    base_curve.plot(ax=ax, label='Minimum-risk Portfolio')
    bt_aggr['wealth'].plot(ax=ax, label='Aggressive Greed')
    bt_cons['wealth'].plot(ax=ax, label='Conservative Greed')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, filename)


def plot_p_vs_pJ(bt2_aggr: pd.DataFrame, bt2_cons: pd.DataFrame, pJ_test: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    bt2_aggr['p_curr'].plot(ax=ax, label='p(t): original scheme, aggressive')
    bt2_cons['p_curr'].plot(ax=ax, label='p(t): original scheme, conservative', alpha=0.8)
    pJ_test.plot(ax=ax, label='p_J(t): minimum-risk fraction', linewidth=2.0)
    ax.set_title('Section 8.1: $p(t)$ vs $p_J(t)$ under both greed levels')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '12_section81_p_vs_pJ')


def plot_backtests_two_stocks_money(bt3_aggr: pd.DataFrame, bt3_cons: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    bt3_aggr['wealth'].plot(ax=ax, label='Aggressive Greed')
    bt3_cons['wealth'].plot(ax=ax, label='Conservative Greed')
    ax.set_title('Section 9: Two Stocks + Money')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '14_section9_backtest')

# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main() -> None:
    dirs = ensure_output_dirs(OUTPUT_ROOT)
    logger = setup_logging(dirs['logs'] / 'run_log.txt')
    nice_plot_style()

    log_and_print(logger, 'Starting MSDM5058 Project II workflow...')

    # ---------- download ----------
    close_downloaded = download_two_stocks(RISKY_CANDIDATE, SAFE_CANDIDATE, START_DATE, END_DATE, PRICE_FIELD)
    cleaning_result = clean_price_history_by_direct_date_drop(close_downloaded, RISKY_CANDIDATE,
                                                              EXTREME_RETURN_ABS_LOG_THRESHOLD, PRICE_FIELD)
    close_raw = cleaning_result['close_clean']      # full cleaned price history
    ret_raw = cleaning_result['ret_clean']          # returns computed from the full cleaned price history
    close = close_raw.copy()                        # do NOT drop the first price row
    if len(ret_raw) != len(close) - 1:
        raise ValueError(
            f'Section 1 inconsistency after cleaning: n_return_rows={len(ret_raw)} '
            f'but n_price_rows-1={len(close)-1}.'
        )
    extreme_return_summary_raw = cleaning_result['raw_summary']
    extreme_return_events_raw = cleaning_result['raw_events']
    removed_price_rows = cleaning_result['removed_rows']
    extreme_return_summary = cleaning_result['clean_summary']
    extreme_return_events = cleaning_result['clean_events']

    # ---------- split ----------
    train_close, test_close, train_ret, test_ret, split_date = split_prices_and_returns_consistently(close, ret_raw)

    risky, safe, train_vols = decide_risky_safe(ret_raw)
    log_and_print(logger, f'Final label from training-set volatility: risky={risky}, safe={safe}')

    # ---------- save basic tables ----------
    meta = make_summary_table([
        {'item': 'candidate_ticker_1', 'value': RISKY_CANDIDATE},
        {'item': 'candidate_ticker_2', 'value': SAFE_CANDIDATE},
        {'item': 'final_risky', 'value': risky},
        {'item': 'final_safe', 'value': safe},
        {'item': 'start_date', 'value': str(close.index.min().date())},
        {'item': 'end_date', 'value': str(close.index.max().date())},
        {'item': 'n_aligned_price_rows', 'value': int(len(close))},
        {'item': 'n_return_rows', 'value': int(len(ret_raw))},
        {'item': 'train_rows', 'value': int(len(train_close))},
        {'item': 'test_rows', 'value': int(len(test_close))},
        {'item': 'split_date_first_test_day', 'value': str(split_date.date())},
        {'item': 'daily_rf', 'value': DAILY_RF},
        {'item': 'aggressive_g', 'value': AGGRESSIVE_G},
        {'item': 'conservative_g', 'value': CONSERVATIVE_G},
        {'item': 'eps_scale', 'value': EPS_SCALE},
        {'item': 'price_field', 'value': PRICE_FIELD},
        {'item': 'download_timestamp_utc', 'value': str(pd.Timestamp.utcnow())},
        {'item': 'conditional_density_kind', 'value': CONDITIONAL_DENSITY_KIND},
        {'item': 'extreme_return_abs_log_threshold', 'value': EXTREME_RETURN_ABS_LOG_THRESHOLD},
        {'item': 'source_1', 'value': f'https://finance.yahoo.com/quote/{RISKY_CANDIDATE}/history'},
        {'item': 'source_2', 'value': f'https://finance.yahoo.com/quote/{SAFE_CANDIDATE}/history'},
    ])
    save_table(meta, dirs['tables'], '00_run_metadata', index=False)
    save_table(descriptive_stats(close), dirs['tables'], '01_price_descriptive_stats')
    save_table(descriptive_stats(ret_raw), dirs['tables'], '02_return_descriptive_stats')
    save_table(train_vols.to_frame('training_volatility'), dirs['tables'], '03_training_volatility')
    save_table(train_ret.corr(), dirs['tables'], '04_train_return_correlation')
    save_table(train_ret.cov(), dirs['tables'], '05_train_return_covariance')
    save_table(close, dirs['tables'], '06_aligned_prices')
    save_table(ret_raw, dirs['tables'], '07_aligned_returns')
    citation_df = pd.DataFrame([
        {'ticker': RISKY_CANDIDATE, 'source': 'Yahoo Finance', 'access_via': 'yfinance', 'url': f'https://finance.yahoo.com/quote/{RISKY_CANDIDATE}/history'},
        {'ticker': SAFE_CANDIDATE, 'source': 'Yahoo Finance', 'access_via': 'yfinance', 'url': f'https://finance.yahoo.com/quote/{SAFE_CANDIDATE}/history'},
    ])
    save_table(citation_df, dirs['tables'], '08_data_source_citation', index=False)
    (dirs['root'] / 'data_source_citation.txt').write_text(
        f'Data source: Yahoo Finance historical adjusted close prices downloaded via yfinance.\n'
        f'{RISKY_CANDIDATE}: https://finance.yahoo.com/quote/{RISKY_CANDIDATE}/history\n'
        f'{SAFE_CANDIDATE}: https://finance.yahoo.com/quote/{SAFE_CANDIDATE}/history\n',
        encoding='utf-8'
    )
    save_table(extreme_return_summary_raw, dirs['tables'], '08b_extreme_return_summary_raw', index=False)
    save_table(extreme_return_events_raw, dirs['tables'], '09_extreme_return_events_raw', index=False)
    save_table(removed_price_rows, dirs['tables'], '09b_removed_price_rows_due_to_AMZN_extreme_event')
    save_table(extreme_return_summary, dirs['tables'], '08_extreme_return_summary', index=False)
    save_table(extreme_return_events, dirs['tables'], '09_extreme_return_events', index=False)
    if removed_price_rows.empty:
        log_and_print(logger, f'No {RISKY_CANDIDATE} extreme log-return dates needed to be removed under |r| >= {EXTREME_RETURN_ABS_LOG_THRESHOLD:.2f}.')
    else:
        log_and_print(logger, f'Removed {len(removed_price_rows)} price row(s) associated with {RISKY_CANDIDATE} extreme log-return event(s).')
    if extreme_return_events.empty:
        log_and_print(logger, f'No extreme log-return events remain after direct date-drop cleaning under |r| >= {EXTREME_RETURN_ABS_LOG_THRESHOLD:.2f}.')
    else:
        log_and_print(logger, f'Extreme log-return events still remain after cleaning: {len(extreme_return_events)} rows saved to 09_extreme_return_events.csv')

    # ---------- section 1 plots ----------
    plot_prices_returns_split(close, ret_raw, risky, safe, split_date, dirs['figures'], PRICE_FIELD)
    plot_return_histograms(train_ret, risky, safe, dirs['figures'])
    plot_rolling_volatility(ret_raw, risky, safe, dirs['figures'])

    # ---------- section 2 ----------
    pJ_dict = {}
    port_dict = {}
    pJ_summary_rows = []

    for h in H_WINDOWS:
        h_label = 'inf' if h is np.inf else str(h)
        label = f'h={h_label}'
        pJ = rolling_min_var_weights(train_ret[[risky, safe]], risky, safe, h)
        pJ_dict[label] = pJ

        port = portfolio_from_dynamic_weights(train_close[[risky, safe]], train_ret[[risky, safe]], pJ, risky, safe)
        port_dict[label] = port

        valid_pJ = pJ.dropna()
        sharpe_col = f'rolling_sharpe_{ROLLING_SHARPE_WINDOW}d'
        pJ_summary_rows.append({
            'window_h': h_label,
            'pJ_count_non_na': int(valid_pJ.count()),
            'pJ_mean': float(valid_pJ.mean()) if len(valid_pJ) else np.nan,
            'pJ_std': float(valid_pJ.std(ddof=1)) if len(valid_pJ) > 1 else np.nan,
            'pJ_last': float(valid_pJ.iloc[-1]) if len(valid_pJ) else np.nan,
            'SJ_normalized_last': float(port['S_J_normalized'].dropna().iloc[-1]) if not port['S_J_normalized'].dropna().empty else np.nan,
            'rolling_sharpe_mean': float(port[sharpe_col].dropna().mean()) if not port[sharpe_col].dropna().empty else np.nan,
            'rolling_sharpe_last': float(port[sharpe_col].dropna().iloc[-1]) if not port[sharpe_col].dropna().empty else np.nan,
        })

        save_series(pJ, dirs['tables'], f'10_pJ_{label}')
        save_table(port, dirs['tables'], f'11_portfolio_{label}')

    pJ_summary = make_summary_table(pJ_summary_rows)
    save_table(pJ_summary, dirs['tables'], '12_pJ_portfolio_summary', index=False)
    plot_pJ(pJ_dict, dirs['figures'])
    plot_portfolios(port_dict, dirs['figures'])

    # choose one pJ series for later use; prefer h=300, fallback to inf
    pJ_train = pJ_dict['h=300'].copy() if 'h=300' in pJ_dict else pJ_dict['h=inf'].copy()
    if pJ_train.dropna().empty:
        pJ_train = pJ_dict['h=inf'].copy()

    all_ret = ret_raw[[risky, safe]].copy()
    frontier_info_all_300 = rolling_min_var_frontier_stats(all_ret, risky, safe, 300)
    save_table(frontier_info_all_300, dirs['tables'], '13_frontier_info_all_300')

    pJ_pretest = float(pJ_train.dropna().iloc[-1]) if not pJ_train.dropna().empty else 0.5
    frontier_pretest = frontier_info_all_300.loc[train_ret.index[-1]].copy() if train_ret.index[-1] in frontier_info_all_300.index else pd.Series({'pJ': pJ_pretest, 'sigmaJ': np.nan, 'var_r': np.nan, 'var_s': np.nan, 'cov_rs': np.nan, 'sigma_risky': np.nan})
    frontier_available_test = frontier_info_all_300.shift(1).reindex(test_ret.index)
    if len(frontier_available_test) > 0:
        frontier_available_test.iloc[0] = frontier_pretest
    save_table(frontier_available_test, dirs['tables'], '14_frontier_available_test')

    pJ_available_test = frontier_available_test['pJ'].rename('pJ_available_test')
    sigmaJ_available_test = frontier_available_test['sigmaJ'].rename('sigmaJ_available_test')
    save_series(pJ_available_test, dirs['tables'], '15_pJ_available_test_series')
    save_series(sigmaJ_available_test, dirs['tables'], '16_sigmaJ_available_test_series')

    # ---------- section 3 ----------
    ma_df = add_moving_averages(train_close[risky], MA_WINDOWS)
    macd_train = add_macd(train_close[risky], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    save_table(ma_df, dirs['tables'], '20_train_moving_averages')
    save_table(macd_train, dirs['tables'], '21_train_macd')

    macd_cross_signal = macd_train.loc[macd_train['cross_signal'].abs() == 1, ['price', 'macd', 'signal', 'cross_signal']].copy()
    macd_cross_zero = macd_train.loc[macd_train['cross_zero'].abs() == 1, ['price', 'macd', 'signal', 'cross_zero']].copy()
    save_table(macd_cross_signal, dirs['tables'], '22_macd_cross_signal_events')
    save_table(macd_cross_zero, dirs['tables'], '23_macd_cross_zero_events')
    plot_ma_macd(ma_df, macd_train, dirs['figures'])

    # ---------- section 4 ----------
    fit_pars = fit_normal_and_logistic(train_ret[risky])
    fit_pars_df = make_summary_table([
        {'parameter': 'mu', 'value': fit_pars['mu']},
        {'parameter': 'sigma', 'value': fit_pars['sigma']},
        {'parameter': 'x0', 'value': fit_pars['x0']},
        {'parameter': 'b', 'value': fit_pars['b']},
    ])
    save_table(fit_pars_df, dirs['tables'], '30_distribution_fit_parameters', index=False)
    log_and_print(logger, f"Section 4 parameters: mu={fit_pars['mu']:.8f}, sigma={fit_pars['sigma']:.8f}, x0={fit_pars['x0']:.8f}, b={fit_pars['b']:.8f}")
    plot_pdf_cdf_fit(train_ret[risky], fit_pars, dirs['figures'])

    x_train_vals = train_ret[risky].dropna().values
    xs_ecdf, F_ecdf = empirical_cdf(x_train_vals)
    grid_fit = np.linspace(np.min(xs_ecdf), np.max(xs_ecdf), 800)
    cdf_compare = pd.DataFrame({
        'x': grid_fit,
        'normal_cdf': norm.cdf(grid_fit, loc=fit_pars['mu'], scale=max(fit_pars['sigma'], 1e-8)),
        'logistic_cdf': logistic_cdf(grid_fit, fit_pars['x0'], fit_pars['b'])
    })
    save_table(cdf_compare, dirs['tables'], '31_cdf_fit_grid', index=False)

    # ---------- section 4.1 and 5 ----------
    eps = EPS_SCALE * train_ret[risky].std(ddof=1)
    log_and_print(logger, f'Digitization threshold epsilon = {eps:.8f}')

    Y_all = digitize_returns(ret_raw[risky], eps)
    Y_train = Y_all.loc[train_ret.index]
    save_series(Y_all, dirs['tables'], '40_digitized_risky_all')
    save_series(Y_train, dirs['tables'], '41_digitized_risky_train')
    save_table(Y_train.value_counts().reindex(['D', 'U', 'H']).fillna(0).to_frame('count'), dirs['tables'], '42_digitized_risky_train_counts')
    plot_digitized_counts(Y_train, dirs['figures'], '11_digitized_counts_risky_train')

    X_t_train = train_ret[risky].iloc[:-1]
    Y_next_train = Y_train.shift(-1).iloc[:-1]

    density_kind = CONDITIONAL_DENSITY_KIND
    priors, cond = bayes_detector_setup(X_t_train, Y_next_train, density_kind)
    priors_df = pd.DataFrame({'class': ['D', 'U', 'H'], 'prior': [priors['D'], priors['U'], priors['H']]})
    save_table(priors_df, dirs['tables'], '43_bayes_priors', index=False)
    log_and_print(logger, f'Section 5 priors: {priors}')

    # conditional samples and fit summaries
    conditional_sample_rows = []
    conditional_fit_rows = []
    for lab in ['D', 'U', 'H']:
        x_sub = X_t_train[Y_next_train == lab]
        conditional_sample_rows.append({
            'label': lab,
            'count': int(x_sub.count()),
            'mean': float(x_sub.mean()) if len(x_sub) else np.nan,
            'std': float(x_sub.std(ddof=1)) if len(x_sub) > 1 else np.nan,
            'min': float(x_sub.min()) if len(x_sub) else np.nan,
            'max': float(x_sub.max()) if len(x_sub) else np.nan,
        })
        fit_kind = cond[lab].kind
        row = {'label': lab, 'density_kind': fit_kind}
        for k, v in cond[lab].params.items():
            row[k] = v
        conditional_fit_rows.append(row)
    save_table(make_summary_table(conditional_sample_rows), dirs['tables'], '44_conditional_sample_stats', index=False)
    save_table(make_summary_table(conditional_fit_rows), dirs['tables'], '45_conditional_fit_params', index=False)

    plot_conditional_cdf_pdf(X_t_train, Y_next_train, cond, dirs['figures'])
    crit, bayes_grid, bayes_labels = plot_bayes_boundaries(priors, cond, X_t_train, dirs['figures'])
    save_table(pd.DataFrame({'critical_point': crit}), dirs['tables'], '46_bayes_critical_points', index=False)
    bayes_score_grid = pd.DataFrame({'x': bayes_grid, 'predicted_label': bayes_labels})
    for lab in ['D', 'U', 'H']:
        bayes_score_grid[f'score_{lab}'] = priors[lab] * cond[lab].pdf(bayes_grid)
    save_table(bayes_score_grid, dirs['tables'], '47_bayes_score_grid', index=False)

    # ---------- section 6 ----------
    rules = make_rules(Y_train, k=5)
    save_table(rules, dirs['tables'], '50_all_association_rules')

    top_support = rules.sort_values(['support', 'confidence'], ascending=False).head(10)
    top_conf = rules.sort_values(['confidence', 'support'], ascending=False).head(10)
    top_geom = rules.sort_values(['geom_mean', 'support'], ascending=False).head(10)
    top_rpf = rules.sort_values(['RPF', 'support'], ascending=False).head(10)
    save_table(top_support, dirs['tables'], '51_top10_support')
    save_table(top_conf, dirs['tables'], '52_top10_confidence')
    save_table(top_geom, dirs['tables'], '53_top10_geom_mean')
    save_table(top_rpf, dirs['tables'], '54_top10_rpf')

    log_and_print(logger, 'Top 10 by support saved.')
    log_and_print(logger, 'Top 10 by confidence saved.')
    log_and_print(logger, 'Top 10 by geometric mean saved.')
    log_and_print(logger, 'Top 10 by RPF saved.')
    lambda_grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    lambda_top, lambda_util_all = lambda_sweep_analysis(rules, lambda_grid)
    lambda_distinct = (lambda_top.assign(rule=lambda_top['antecedent'] + '->' + lambda_top['consequent'])
                       .groupby('rule', as_index=False)
                       .agg(n_lambda=('lambda', 'count'), lambda_min=('lambda', 'min'), lambda_max=('lambda', 'max'))
                       .sort_values(['n_lambda', 'lambda_min'], ascending=[False, True]))
    rules_with_lambda = add_rule_usefulness_columns(rules, [float(x) for x in lambda_grid])
    lambda_summary = summarize_best_lambda_rules(rules, [float(x) for x in lambda_grid], top_n=10)
    save_table(lambda_top, dirs['tables'], '55_lambda_sweep_top_rules', index=False)
    save_table(lambda_util_all, dirs['tables'], '56_lambda_sweep_all_utilities', index=False)
    save_table(lambda_distinct, dirs['tables'], '57_lambda_distinct_top_rules', index=False)
    save_table(rules_with_lambda, dirs['tables'], '58_all_association_rules_with_lambda')
    save_table(lambda_summary, dirs['tables'], '59_top10_rules_by_lambda', index=False)
    save_table(summarize_section61(lambda_top, lambda_distinct), dirs['tables'], '59a_section61_summary', index=False)
    plot_lambda_sweep(lambda_top, dirs['figures'])
    log_and_print(logger, 'Section 6.1 lambda sweep saved.')

    # ---------- build test-period risky signals ----------
    macd_all_risky = add_macd(close[risky], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    save_table(macd_all_risky, dirs['tables'], '60_macd_all_risky')

    signal_table_risky = build_signal_table(test_close.index, ret_raw[risky], macd_all_risky,
                                            priors, cond, Y_all, rules)
    signal_series_risky = signal_table_risky['combined_signal'].astype(int)
    save_table(signal_table_risky, dirs['tables'], '61_signal_table_risky_test_info')
    save_table(signal_table_risky.shift(1).fillna(0).astype(int), dirs['tables'], '62_signal_table_risky_test_trade')

    # ---------- safer stock signals for section 9 ----------
    macd_all_safe = add_macd(close[safe], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    save_table(macd_all_safe, dirs['tables'], '63_macd_all_safe')

    eps_safe = EPS_SCALE * train_ret[safe].std(ddof=1)
    Y_all_safe = digitize_returns(ret_raw[safe], eps_safe)
    Y_train_safe = Y_all_safe.loc[train_ret.index]
    save_series(Y_all_safe, dirs['tables'], '64_digitized_safe_all')
    save_series(Y_train_safe, dirs['tables'], '65_digitized_safe_train')
    save_table(Y_train_safe.value_counts().reindex(['D', 'U', 'H']).fillna(0).to_frame('count'), dirs['tables'], '66_digitized_safe_train_counts')
    plot_digitized_counts(Y_train_safe, dirs['figures'], '13_digitized_counts_safe_train')

    X_t_train_s = train_ret[safe].iloc[:-1]
    Y_next_train_s = Y_train_safe.shift(-1).iloc[:-1]
    priors_s, cond_s = bayes_detector_setup(X_t_train_s, Y_next_train_s, density_kind)
    save_table(pd.DataFrame({'class': ['D', 'U', 'H'], 'prior': [priors_s['D'], priors_s['U'], priors_s['H']]}),
               dirs['tables'], '67_bayes_priors_safe', index=False)
    rules_s = make_rules(Y_train_safe, k=5)
    save_table(rules_s, dirs['tables'], '68_all_association_rules_safe')

    signal_table_safe = build_signal_table(test_close.index, ret_raw[safe], macd_all_safe,
                                           priors_s, cond_s, Y_all_safe, rules_s)
    signal_series_safe = signal_table_safe['combined_signal'].astype(int)
    save_table(signal_table_safe, dirs['tables'], '69_signal_table_safe_test_info')
    save_table(signal_table_safe.shift(1).fillna(0).astype(int), dirs['tables'], '69b_signal_table_safe_test_trade')

    # ---------- section 7 ----------
    bt_aggr = backtest_one_stock_money(test_close[risky], signal_table_risky, AGGRESSIVE_G)
    bt_cons = backtest_one_stock_money(test_close[risky], signal_table_risky, CONSERVATIVE_G)
    bt_bob = backtest_bob_dynamic_greed(test_close[risky], signal_table_risky,
                                        AGGRESSIVE_G, CONSERVATIVE_G)
    save_table(bt_aggr, dirs['tables'], '70_section7_backtest_aggressive')
    save_table(bt_cons, dirs['tables'], '71_section7_backtest_conservative')
    save_table(bt_bob, dirs['tables'], '72_section7_backtest_bob')
    save_table(summarize_bob_policy(bt_bob, AGGRESSIVE_G, CONSERVATIVE_G), dirs['tables'], '72a_section7_bob_policy_summary', index=False)

    base_cash = build_cash_benchmark(test_close.index, INITIAL_CAPITAL, DAILY_RF)
    save_series(base_cash, dirs['tables'], '73_cash_benchmark_test')
    save_table(summarize_section7(base_cash, bt_aggr, bt_cons, bt_bob), dirs['tables'], '74_section7_summary', index=False)
    plot_backtests_one_stock(base_cash, bt_aggr, bt_cons, bt_bob, dirs['figures'])

    # ---------- section 8 ----------
    pJ0 = float(pJ_pretest) if not np.isnan(pJ_pretest) else 0.5
    bt2_aggr = backtest_two_stocks_original(test_close[[risky, safe]], risky, safe,
                                            signal_series_risky, pJ0, AGGRESSIVE_G)
    bt2_cons = backtest_two_stocks_original(test_close[[risky, safe]], risky, safe,
                                            signal_series_risky, pJ0, CONSERVATIVE_G)
    save_table(bt2_aggr, dirs['tables'], '80_section8_backtest_aggressive')
    save_table(bt2_cons, dirs['tables'], '81_section8_backtest_conservative')

    base_minrisk = build_dynamic_minrisk_benchmark(test_close[[risky, safe]], risky, safe,
                                                   pJ_available_test, INITIAL_CAPITAL, initial_pJ=pJ0)
    save_series(base_minrisk, dirs['tables'], '82_base_minrisk_test')

    plot_backtests_two_stock(base_minrisk,
                             bt2_aggr.reindex(base_minrisk.index),
                             bt2_cons.reindex(base_minrisk.index),
                             'Section 8: Two Stocks, Original Trading Scheme',
                             dirs['figures'], '15_section8_backtest_original')

    # ---------- section 8.1 ----------
    bt2ef_aggr = backtest_two_stocks_efficient_frontier(test_close[[risky, safe]], risky, safe,
                                                        signal_series_risky, frontier_available_test,
                                                        AGGRESSIVE_G, pJ0)
    bt2ef_cons = backtest_two_stocks_efficient_frontier(test_close[[risky, safe]], risky, safe,
                                                        signal_series_risky, frontier_available_test,
                                                        CONSERVATIVE_G, pJ0)
    save_table(bt2ef_aggr, dirs['tables'], '90_section81_backtest_aggressive')
    save_table(bt2ef_cons, dirs['tables'], '91_section81_backtest_conservative')

    sec81_diag = pd.DataFrame([
        {'strategy': 'aggressive', 'n_days_p_below_pJ': int((bt2ef_aggr['p_curr'] + 1e-12 < bt2ef_aggr['pJ_t']).sum())},
        {'strategy': 'conservative', 'n_days_p_below_pJ': int((bt2ef_cons['p_curr'] + 1e-12 < bt2ef_cons['pJ_t']).sum())},
    ])
    save_table(sec81_diag, dirs['tables'], '91a_section81_floor_diagnostics', index=False)

    plot_p_vs_pJ(bt2_aggr, bt2_cons, pJ_available_test, dirs['figures'])
    save_table(summarize_section81(base_minrisk, bt2ef_aggr, bt2ef_cons), dirs['tables'], '91b_section81_summary', index=False)

    plot_p_vs_pJ(bt2_aggr, bt2_cons, pJ_available_test, dirs['figures'])
    plot_backtests_two_stock(base_minrisk,
                             bt2ef_aggr.reindex(base_minrisk.index),
                             bt2ef_cons.reindex(base_minrisk.index),
                             'Section 8.1: Two Stocks with Efficient-frontier Control',
                             dirs['figures'], '16_section81_backtest_efficient_frontier')

    # ---------- section 9 ----------
    bt3_aggr = backtest_two_stocks_money(test_close[[risky, safe]], risky, safe,
                                         signal_series_risky, signal_series_safe,
                                         AGGRESSIVE_G)
    bt3_cons = backtest_two_stocks_money(test_close[[risky, safe]], risky, safe,
                                         signal_series_risky, signal_series_safe,
                                         CONSERVATIVE_G)
    save_table(bt3_aggr, dirs['tables'], '100_section9_backtest_aggressive')
    save_table(bt3_cons, dirs['tables'], '101_section9_backtest_conservative')
    save_table(summarize_section9(bt3_aggr, bt3_cons), dirs['tables'], '102_section9_summary', index=False)
    plot_backtests_two_stocks_money(bt3_aggr, bt3_cons, dirs['figures'])

    # ---------- final summary ----------
    final_summary = make_summary_table([
        {'strategy': 'Section 7 aggressive', 'final_wealth': float(bt_aggr['wealth'].iloc[-1])},
        {'strategy': 'Section 7 conservative', 'final_wealth': float(bt_cons['wealth'].iloc[-1])},
        {'strategy': 'Section 7 Bob', 'final_wealth': float(bt_bob['wealth'].iloc[-1])},
        {'strategy': 'Section 8 aggressive', 'final_wealth': float(bt2_aggr['wealth'].iloc[-1])},
        {'strategy': 'Section 8 conservative', 'final_wealth': float(bt2_cons['wealth'].iloc[-1])},
        {'strategy': 'Section 8.1 aggressive', 'final_wealth': float(bt2ef_aggr['wealth'].iloc[-1])},
        {'strategy': 'Section 8.1 conservative', 'final_wealth': float(bt2ef_cons['wealth'].iloc[-1])},
        {'strategy': 'Section 9 aggressive', 'final_wealth': float(bt3_aggr['wealth'].iloc[-1])},
        {'strategy': 'Section 9 conservative', 'final_wealth': float(bt3_cons['wealth'].iloc[-1])},
    ])
    save_table(final_summary, dirs['tables'], '110_final_wealth_summary', index=False)

    log_and_print(logger, 'Final wealth summary:')
    for _, row in final_summary.iterrows():
        log_and_print(logger, f"  {row['strategy']}: {row['final_wealth']:.2f}")

    # one small text guide file
    readme_text = f"""MSDM5058 Project II output folder
=================================

This script created:
- figures/ : report-ready figures
- tables/  : CSV tables for intermediate and final results
- logs/    : run_log.txt

Main files to look at first:
1. figures/01_prices_returns_split.png
2. figures/04_pJ_curves.png
3. figures/05_portfolio_SJ_and_sharpe.png
3b. tables/48_section41_summary.csv
3c. tables/59a_section61_summary.csv
3d. tables/74_section7_summary.csv
3e. tables/91b_section81_summary.csv
3f. tables/102_section9_summary.csv
4. figures/06_ma_macd.png
5. figures/07_cdf_fits.png
6. figures/08_conditional_cdf_pdf.png
7. figures/09_bayes_boundaries.png
8. figures/10_section7_backtest.png
9. figures/15_section8_backtest_original.png
10. figures/16_section81_backtest_efficient_frontier.png
11. figures/14_section9_backtest.png

Key tables:
- tables/00_run_metadata.csv
- tables/08_data_source_citation.csv
- tables/12_pJ_portfolio_summary.csv
- tables/30_distribution_fit_parameters.csv
- tables/43_bayes_priors.csv
- tables/46_bayes_critical_points.csv
- tables/51_top10_support.csv
- tables/55_lambda_sweep_top_rules.csv
- tables/57_lambda_distinct_top_rules.csv
- tables/52_top10_confidence.csv
- tables/53_top10_geom_mean.csv
- tables/54_top10_rpf.csv
- tables/59a_section61_summary.csv
- tables/74_section7_summary.csv
- tables/91b_section81_summary.csv
- tables/102_section9_summary.csv
- tables/110_final_wealth_summary.csv

Section-focused summary files added in this version:
- tables/48_section41_summary.csv
- tables/59a_section61_summary.csv
- tables/74_section7_summary.csv
- tables/91b_section81_summary.csv
- tables/102_section9_summary.csv
"""
    (dirs['root'] / 'README_outputs.txt').write_text(readme_text, encoding='utf-8')

    log_and_print(logger, f'All outputs saved under: {dirs["root"].resolve()}')
    log_and_print(logger, 'Workflow finished successfully.')


if __name__ == '__main__':
    main()
