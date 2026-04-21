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
RISKY_CANDIDATE = "AAPL"   # candidate only; final risky/safe labels are determined by training volatility
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
USE_KDE_FOR_CONDITIONAL_PDF = True
PLOT_DPI = 180
ROLLING_SHARPE_WINDOW = 60
ROLLING_VOL_WINDOW = 60
OUTPUT_ROOT = "outputs"   # folder automatically created
SAVE_FIG_PDF = False       # True -> also save PDF copies of figures

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

def download_two_stocks(ticker1: str, ticker2: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    data = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' not in data.columns.get_level_values(0):
            raise ValueError("Could not find 'Adj Close' in yfinance output.")
        close = data['Adj Close'].copy()
    else:
        raise ValueError("Unexpected yfinance output format.")

    close = close.dropna(how='any').copy()
    close.columns = [ticker1, ticker2]

    if len(close) < 4001:
        raise ValueError(f"Common aligned price history has only {len(close)} rows; need > 4000 days.")

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

# ============================================================
# 3. MARKOWITZ MINIMUM-RISK PORTFOLIO
# ============================================================

def min_var_weight_two_assets(var_risky: float, var_safe: float, cov_rs: float) -> float:
    denom = var_risky + var_safe - 2.0 * cov_rs
    if abs(denom) < 1e-14:
        return 0.5
    p = (var_safe - cov_rs) / denom
    return float(np.clip(p, 0.0, 1.0))


def rolling_min_var_weights(ret2: pd.DataFrame, risky: str, safe: str, h) -> pd.Series:
    out = []
    idxs = []
    for i in range(len(ret2)):
        if h is np.inf:
            window = ret2.iloc[:i+1]
        else:
            if i + 1 < h:
                out.append(np.nan)
                idxs.append(ret2.index[i])
                continue
            window = ret2.iloc[i-h+1:i+1]
        if len(window) < 2:
            out.append(np.nan)
            idxs.append(ret2.index[i])
            continue
        var_r = window[risky].var(ddof=1)
        var_s = window[safe].var(ddof=1)
        cov_rs = window[[risky, safe]].cov().loc[risky, safe]
        p = min_var_weight_two_assets(var_r, var_s, cov_rs)
        out.append(p)
        idxs.append(ret2.index[i])
    return pd.Series(out, index=idxs, name=f'pJ_h_{h}')


def portfolio_from_dynamic_weights(ret2: pd.DataFrame, w_risky: pd.Series, risky: str, safe: str,
                                   rf_daily: float = DAILY_RF) -> pd.DataFrame:
    aligned_w = w_risky.reindex(ret2.index).copy()
    w_prev = aligned_w.shift(1)
    port_ret = w_prev * ret2[risky] + (1.0 - w_prev) * ret2[safe]
    port_ret = port_ret.dropna()

    wealth = pd.Series(index=port_ret.index, dtype=float)
    if len(port_ret) > 0:
        wealth.iloc[0] = 1.0
        for i in range(1, len(port_ret)):
            wealth.iloc[i] = wealth.iloc[i-1] * math.exp(port_ret.iloc[i])

    sharpe_roll = ((port_ret.rolling(ROLLING_SHARPE_WINDOW).mean() - rf_daily) /
                   port_ret.rolling(ROLLING_SHARPE_WINDOW).std(ddof=1)) * np.sqrt(252)

    out = pd.DataFrame({
        'portfolio_return': port_ret,
        'portfolio_wealth': wealth,
        f'rolling_sharpe_{ROLLING_SHARPE_WINDOW}d': sharpe_roll,
        'weight_risky_prevday': w_prev.reindex(port_ret.index)
    })
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


def fit_conditional_density(x: pd.Series, kind: str = 'kde') -> ConditionalDensity:
    x = x.dropna().values
    if len(x) < 5:
        kind = 'normal'
    if kind == 'kde':
        kde = gaussian_kde(x)
        return ConditionalDensity(kind='kde', params={}, kde=kde)
    if kind == 'normal':
        return ConditionalDensity(kind='normal', params={'mu': float(np.mean(x)), 'sigma': float(np.std(x, ddof=1))})
    if kind == 'logistic':
        pars = fit_normal_and_logistic(pd.Series(x))
        return ConditionalDensity(kind='logistic', params={'x0': pars['x0'], 'b': pars['b']})
    raise ValueError('Unsupported conditional density kind')

# ============================================================
# 7. BAYES DETECTOR
# ============================================================

def bayes_detector_setup(x_t: pd.Series, y_next: pd.Series, density_kind: str = 'kde'):
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

# ============================================================
# 10. BACKTESTS
# ============================================================

def backtest_one_stock_money(test_price: pd.Series, test_ret: pd.Series, macd_df: pd.DataFrame,
                             priors, cond, y_all: pd.Series, rules: pd.DataFrame,
                             g: float, initial_capital: float = INITIAL_CAPITAL,
                             rf_daily: float = DAILY_RF) -> pd.DataFrame:
    dates = test_price.index
    M = initial_capital
    N = 0.0
    out = []

    for dt in dates:
        M *= (1.0 + rf_daily)
        action = 0
        macd_s = bayes_s = rule_s = 0

        if dt in test_ret.index and dt in macd_df.index:
            x_today = float(test_ret.loc[dt])
            macd_s = macd_signal_today(macd_df, dt)
            bayes_s = bayes_signal_today(x_today, priors, cond)
            hist_idx = y_all.index.get_loc(dt) if dt in y_all.index else None
            if hist_idx is not None and hist_idx >= 4:
                history5 = tuple(y_all.iloc[hist_idx-4:hist_idx+1].tolist())
                rule_s = rule_signal_today(history5, rules)
            action = combined_signal(macd_s, bayes_s, rule_s)

        price = float(test_price.loc[dt])
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
            'date': dt, 'money': M, 'shares': N, 'price': price, 'wealth': V,
            'macd_signal': macd_s, 'bayes_signal': bayes_s, 'rule_signal': rule_s,
            'action': action, 'g': g, 'trade_cash': trade_cash, 'trade_shares': trade_shares
        })
    return pd.DataFrame(out).set_index('date')


def backtest_bob_dynamic_greed(test_price: pd.Series, test_ret: pd.Series, macd_df: pd.DataFrame,
                               priors, cond, y_all: pd.Series, rules: pd.DataFrame,
                               g_aggr: float, g_cons: float,
                               initial_capital: float = INITIAL_CAPITAL,
                               rf_daily: float = DAILY_RF) -> pd.DataFrame:
    dates = test_price.index
    M = initial_capital
    N = 0.0
    out = []

    for dt in dates:
        M *= (1.0 + rf_daily)
        action = 0
        g_today = 0.0
        macd_s = bayes_s = rule_s = 0

        if dt in test_ret.index and dt in macd_df.index:
            x_today = float(test_ret.loc[dt])
            macd_s = macd_signal_today(macd_df, dt)
            bayes_s = bayes_signal_today(x_today, priors, cond)
            hist_idx = y_all.index.get_loc(dt) if dt in y_all.index else None
            if hist_idx is not None and hist_idx >= 4:
                history5 = tuple(y_all.iloc[hist_idx-4:hist_idx+1].tolist())
                rule_s = rule_signal_today(history5, rules)
            action = combined_signal(macd_s, bayes_s, rule_s)
            conf = confidence_score(macd_s, bayes_s, rule_s)
            g_today = g_aggr if conf >= 2 else g_cons

        price = float(test_price.loc[dt])
        trade_cash = 0.0
        trade_shares = 0.0
        if action > 0 and M > 0:
            spend = g_today * M
            N += spend / price
            M -= spend
            trade_cash = -spend
            trade_shares = spend / price
        elif action < 0 and N > 0:
            sell = g_today * N
            cash = sell * price
            M += cash
            N -= sell
            trade_cash = cash
            trade_shares = -sell

        V = M + N * price
        out.append({'date': dt, 'money': M, 'shares': N, 'price': price, 'wealth': V,
                    'macd_signal': macd_s, 'bayes_signal': bayes_s, 'rule_signal': rule_s,
                    'action': action, 'g_today': g_today, 'trade_cash': trade_cash, 'trade_shares': trade_shares})
    return pd.DataFrame(out).set_index('date')


def backtest_two_stocks_original(test_close: pd.DataFrame, risky: str, safe: str,
                                 signal_series: pd.Series, pJ0: float, g: float,
                                 initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    Nr = pJ0 * initial_capital / float(test_close[risky].iloc[0])
    Ns = (1.0 - pJ0) * initial_capital / float(test_close[safe].iloc[0])
    out = []
    for dt in test_close.index:
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        sig = int(signal_series.get(dt, 0))
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
        out.append({'date': dt, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V,
                    'p_curr': p_curr, 'signal': sig,
                    'trade_risky': trade_risky, 'trade_safe': trade_safe})
    return pd.DataFrame(out).set_index('date')


def backtest_two_stocks_efficient_frontier(test_close: pd.DataFrame, risky: str, safe: str,
                                           signal_series: pd.Series, pJ_series_test: pd.Series,
                                           sigmaJ_series_test: pd.Series, sigma_risky_train: float,
                                           train_cov: float, train_var_safe: float,
                                           g: float, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    pJ0 = float(pJ_series_test.dropna().iloc[0])
    Nr = pJ0 * initial_capital / float(test_close[risky].iloc[0])
    Ns = (1.0 - pJ0) * initial_capital / float(test_close[safe].iloc[0])
    out = []

    for dt in test_close.index:
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        V = Nr * pr + Ns * ps
        p_curr = (Nr * pr) / V if V > 0 else 0.0

        pJ_t = float(pJ_series_test.get(dt, np.nan))
        sigmaJ_t = float(sigmaJ_series_test.get(dt, np.nan))
        sig = int(signal_series.get(dt, 0))
        p_target = np.nan
        target_sigma = np.nan

        if not np.isnan(pJ_t) and not np.isnan(sigmaJ_t):
            sigma_curr = portfolio_sigma_from_weight(p_curr, sigma_risky_train**2, train_var_safe, train_cov)
            if sig < 0:
                target_sigma = sigma_curr - g * (sigma_curr - sigmaJ_t)
            elif sig > 0:
                target_sigma = sigma_curr + g * (sigma_risky_train - sigma_curr)
            else:
                target_sigma = sigma_curr

            target_sigma = float(np.clip(target_sigma, min(sigmaJ_t, sigma_risky_train), max(sigmaJ_t, sigma_risky_train)))
            p_target = weight_from_target_sigma(target_sigma, sigma_risky_train**2, train_var_safe, train_cov,
                                                p_min=pJ_t, p_max=1.0)

            Nr = (p_target * V) / pr
            Ns = ((1.0 - p_target) * V) / ps
            V = Nr * pr + Ns * ps
            p_curr = (Nr * pr) / V if V > 0 else np.nan

        out.append({'date': dt, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V, 'p_curr': p_curr,
                    'signal': sig, 'pJ_t': pJ_t, 'sigmaJ_t': sigmaJ_t,
                    'target_sigma': target_sigma, 'p_target': p_target})
    return pd.DataFrame(out).set_index('date')


def backtest_two_stocks_money(test_close: pd.DataFrame, risky: str, safe: str,
                              signal_risky: pd.Series, signal_safe: pd.Series,
                              g: float, initial_capital: float = INITIAL_CAPITAL,
                              rf_daily: float = DAILY_RF) -> pd.DataFrame:
    M = initial_capital
    Nr = 0.0
    Ns = 0.0
    out = []
    for dt in test_close.index:
        M *= (1.0 + rf_daily)
        pr = float(test_close.loc[dt, risky])
        ps = float(test_close.loc[dt, safe])
        sr = int(signal_risky.get(dt, 0))
        ss = int(signal_safe.get(dt, 0))

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
        out.append({'date': dt, 'money': M, 'N_risky': Nr, 'N_safe': Ns,
                    'wealth': V, 'signal_risky': sr, 'signal_safe': ss,
                    'trade_risky': trade_risky, 'trade_safe': trade_safe,
                    'trade_cash_risky': trade_cash_risky, 'trade_cash_safe': trade_cash_safe})
    return pd.DataFrame(out).set_index('date')

# ============================================================
# 11. PLOTTING FUNCTIONS (ALL SAVE TO FILES)
# ============================================================

def plot_prices_returns_split(close: pd.DataFrame, ret: pd.DataFrame, risky: str, safe: str,
                              split_date, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    close[[risky, safe]].plot(ax=axes[0])
    axes[0].axvline(split_date, linestyle='--', color='k', alpha=0.8)
    axes[0].set_title('Adjusted Close Prices with Train/Test Split')

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
        df['portfolio_wealth'].plot(ax=axes[0], label=name)
        df[sharpe_col].plot(ax=axes[1], label=name)
    axes[0].set_title('Minimum-risk Portfolio Wealth Proxy')
    axes[1].set_title(f'Rolling Sharpe Ratio ({ROLLING_SHARPE_WINDOW} days)')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, out_dir, '05_portfolio_wealth_and_sharpe')


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
        ax.plot(grid, priors[lab] * cond[lab].pdf(grid), label=f'q(y)f_y(x), y={lab}')
    for c in crit:
        ax.axvline(c, linestyle='--', alpha=0.5, color='k')
    ax.set_title('Bayes Detector Score Curves and Critical Points')
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


def plot_p_vs_pJ(bt2_aggr: pd.DataFrame, pJ_test: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    bt2_aggr['p_curr'].plot(ax=ax, label='p(t): original scheme')
    pJ_test.plot(ax=ax, label='p_J(t): minimum-risk fraction')
    ax.set_title('Section 8.1: $p(t)$ vs $p_J(t)$')
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
    close_raw = download_two_stocks(RISKY_CANDIDATE, SAFE_CANDIDATE, START_DATE, END_DATE)
    ret_raw = make_returns(close_raw)
    close = close_raw.loc[ret_raw.index].copy()  # align because returns lose first row

    # ---------- split ----------
    train_close, test_close = split_3_to_1(close)
    train_ret, test_ret = split_3_to_1(ret_raw)
    split_date = test_close.index[0]

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
    ])
    save_table(meta, dirs['tables'], '00_run_metadata', index=False)
    save_table(descriptive_stats(close), dirs['tables'], '01_price_descriptive_stats')
    save_table(descriptive_stats(ret_raw), dirs['tables'], '02_return_descriptive_stats')
    save_table(train_vols.to_frame('training_volatility'), dirs['tables'], '03_training_volatility')
    save_table(train_ret.corr(), dirs['tables'], '04_train_return_correlation')
    save_table(train_ret.cov(), dirs['tables'], '05_train_return_covariance')
    save_table(close, dirs['tables'], '06_aligned_prices')
    save_table(ret_raw, dirs['tables'], '07_aligned_returns')

    # ---------- section 1 plots ----------
    plot_prices_returns_split(close, ret_raw, risky, safe, split_date, dirs['figures'])
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

        port = portfolio_from_dynamic_weights(train_ret[[risky, safe]], pJ, risky, safe)
        port_dict[label] = port

        valid_pJ = pJ.dropna()
        sharpe_col = f'rolling_sharpe_{ROLLING_SHARPE_WINDOW}d'
        pJ_summary_rows.append({
            'window_h': h_label,
            'pJ_count_non_na': int(valid_pJ.count()),
            'pJ_mean': float(valid_pJ.mean()) if len(valid_pJ) else np.nan,
            'pJ_std': float(valid_pJ.std(ddof=1)) if len(valid_pJ) > 1 else np.nan,
            'pJ_last': float(valid_pJ.iloc[-1]) if len(valid_pJ) else np.nan,
            'wealth_last': float(port['portfolio_wealth'].dropna().iloc[-1]) if not port['portfolio_wealth'].dropna().empty else np.nan,
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
    pJ_all_300 = rolling_min_var_weights(all_ret, risky, safe, 300)
    pJ_test = pJ_all_300.reindex(test_ret.index)
    save_series(pJ_test, dirs['tables'], '13_pJ_test_series')

    sigmaJ_test = pd.Series(index=test_ret.index, dtype=float, name='sigmaJ_test')
    for dt in test_ret.index:
        idx = all_ret.index.get_loc(dt)
        window = all_ret.iloc[max(0, idx-300+1):idx+1]
        vr = window[risky].var(ddof=1)
        vs = window[safe].var(ddof=1)
        cov = window[[risky, safe]].cov().loc[risky, safe]
        p = pJ_all_300.loc[dt]
        sigmaJ_test.loc[dt] = portfolio_sigma_from_weight(p, vr, vs, cov)
    save_series(sigmaJ_test, dirs['tables'], '14_sigmaJ_test_series')

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

    density_kind = 'kde' if USE_KDE_FOR_CONDITIONAL_PDF else 'normal'
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

    # ---------- build test-period risky signals ----------
    macd_all_risky = add_macd(close[risky], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    save_table(macd_all_risky, dirs['tables'], '60_macd_all_risky')

    signal_rows_risky = []
    signal_series_risky = pd.Series(index=test_close.index, dtype=int, name='combined_signal_risky')
    for dt in test_close.index:
        macd_s = bayes_s = rule_s = 0
        if dt in ret_raw.index:
            x_today = float(ret_raw.loc[dt, risky])
            macd_s = macd_signal_today(macd_all_risky, dt)
            bayes_s = bayes_signal_today(x_today, priors, cond)
            if dt in Y_all.index:
                loc = Y_all.index.get_loc(dt)
                if loc >= 4:
                    hist5 = tuple(Y_all.iloc[loc-4:loc+1].tolist())
                    rule_s = rule_signal_today(hist5, rules)
        combined = combined_signal(macd_s, bayes_s, rule_s)
        conf = confidence_score(macd_s, bayes_s, rule_s)
        signal_series_risky.loc[dt] = combined
        signal_rows_risky.append({
            'date': dt,
            'macd_signal': macd_s,
            'bayes_signal': bayes_s,
            'rule_signal': rule_s,
            'combined_signal': combined,
            'confidence_score': conf
        })
    signal_table_risky = pd.DataFrame(signal_rows_risky).set_index('date')
    save_table(signal_table_risky, dirs['tables'], '61_signal_table_risky_test')

    # ---------- safer stock signals for section 9 ----------
    macd_all_safe = add_macd(close[safe], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    save_table(macd_all_safe, dirs['tables'], '62_macd_all_safe')

    eps_safe = EPS_SCALE * train_ret[safe].std(ddof=1)
    Y_all_safe = digitize_returns(ret_raw[safe], eps_safe)
    Y_train_safe = Y_all_safe.loc[train_ret.index]
    save_series(Y_all_safe, dirs['tables'], '63_digitized_safe_all')
    save_series(Y_train_safe, dirs['tables'], '64_digitized_safe_train')
    save_table(Y_train_safe.value_counts().reindex(['D', 'U', 'H']).fillna(0).to_frame('count'), dirs['tables'], '65_digitized_safe_train_counts')
    plot_digitized_counts(Y_train_safe, dirs['figures'], '13_digitized_counts_safe_train')

    X_t_train_s = train_ret[safe].iloc[:-1]
    Y_next_train_s = Y_train_safe.shift(-1).iloc[:-1]
    priors_s, cond_s = bayes_detector_setup(X_t_train_s, Y_next_train_s, density_kind)
    save_table(pd.DataFrame({'class': ['D', 'U', 'H'], 'prior': [priors_s['D'], priors_s['U'], priors_s['H']]}),
               dirs['tables'], '66_bayes_priors_safe', index=False)
    rules_s = make_rules(Y_train_safe, k=5)
    save_table(rules_s, dirs['tables'], '67_all_association_rules_safe')

    signal_rows_safe = []
    signal_series_safe = pd.Series(index=test_close.index, dtype=int, name='combined_signal_safe')
    for dt in test_close.index:
        macd_s = bayes_s = rule_s = 0
        if dt in ret_raw.index:
            x_today = float(ret_raw.loc[dt, safe])
            macd_s = macd_signal_today(macd_all_safe, dt)
            bayes_s = bayes_signal_today(x_today, priors_s, cond_s)
            if dt in Y_all_safe.index:
                loc = Y_all_safe.index.get_loc(dt)
                if loc >= 4:
                    hist5 = tuple(Y_all_safe.iloc[loc-4:loc+1].tolist())
                    rule_s = rule_signal_today(hist5, rules_s)
        combined = combined_signal(macd_s, bayes_s, rule_s)
        conf = confidence_score(macd_s, bayes_s, rule_s)
        signal_series_safe.loc[dt] = combined
        signal_rows_safe.append({
            'date': dt,
            'macd_signal': macd_s,
            'bayes_signal': bayes_s,
            'rule_signal': rule_s,
            'combined_signal': combined,
            'confidence_score': conf
        })
    signal_table_safe = pd.DataFrame(signal_rows_safe).set_index('date')
    save_table(signal_table_safe, dirs['tables'], '68_signal_table_safe_test')

    # ---------- section 7 ----------
    bt_aggr = backtest_one_stock_money(test_close[risky], test_ret[risky], macd_all_risky,
                                       priors, cond, Y_all, rules, AGGRESSIVE_G)
    bt_cons = backtest_one_stock_money(test_close[risky], test_ret[risky], macd_all_risky,
                                       priors, cond, Y_all, rules, CONSERVATIVE_G)
    bt_bob = backtest_bob_dynamic_greed(test_close[risky], test_ret[risky], macd_all_risky,
                                        priors, cond, Y_all, rules,
                                        AGGRESSIVE_G, CONSERVATIVE_G)
    save_table(bt_aggr, dirs['tables'], '70_section7_backtest_aggressive')
    save_table(bt_cons, dirs['tables'], '71_section7_backtest_conservative')
    save_table(bt_bob, dirs['tables'], '72_section7_backtest_bob')

    base_cash = pd.Series(INITIAL_CAPITAL * ((1 + DAILY_RF) ** np.arange(len(test_close))),
                          index=test_close.index, name='cash_benchmark')
    save_series(base_cash, dirs['tables'], '73_cash_benchmark_test')
    plot_backtests_one_stock(base_cash, bt_aggr, bt_cons, bt_bob, dirs['figures'])

    # ---------- section 8 ----------
    pJ0 = float(pJ_test.dropna().iloc[0]) if not pJ_test.dropna().empty else 0.5
    bt2_aggr = backtest_two_stocks_original(test_close[[risky, safe]], risky, safe,
                                            signal_series_risky, pJ0, AGGRESSIVE_G)
    bt2_cons = backtest_two_stocks_original(test_close[[risky, safe]], risky, safe,
                                            signal_series_risky, pJ0, CONSERVATIVE_G)
    save_table(bt2_aggr, dirs['tables'], '80_section8_backtest_aggressive')
    save_table(bt2_cons, dirs['tables'], '81_section8_backtest_conservative')

    base_minrisk = pd.Series(index=test_ret.index, dtype=float, name='base_minrisk')
    if len(test_ret) > 0:
        base_minrisk.iloc[0] = INITIAL_CAPITAL
    pJ_prev = pJ_test.shift(1)
    test_port_ret = (pJ_prev * test_ret[risky] + (1-pJ_prev) * test_ret[safe]).dropna()
    for i in range(1, len(test_port_ret)):
        base_minrisk.iloc[i] = base_minrisk.iloc[i-1] * math.exp(test_port_ret.iloc[i])
    base_minrisk = base_minrisk.dropna()
    save_series(base_minrisk, dirs['tables'], '82_base_minrisk_test')

    plot_backtests_two_stock(base_minrisk,
                             bt2_aggr.reindex(base_minrisk.index),
                             bt2_cons.reindex(base_minrisk.index),
                             'Section 8: Two Stocks, Original Trading Scheme',
                             dirs['figures'], '15_section8_backtest_original')

    # ---------- section 8.1 ----------
    train_var_r = train_ret[risky].var(ddof=1)
    train_var_s = train_ret[safe].var(ddof=1)
    train_cov = train_ret[[risky, safe]].cov().loc[risky, safe]
    sigma_risky_train = float(np.sqrt(train_var_r))

    bt2ef_aggr = backtest_two_stocks_efficient_frontier(test_close[[risky, safe]], risky, safe,
                                                        signal_series_risky, pJ_test, sigmaJ_test,
                                                        sigma_risky_train, train_cov, train_var_s,
                                                        AGGRESSIVE_G)
    bt2ef_cons = backtest_two_stocks_efficient_frontier(test_close[[risky, safe]], risky, safe,
                                                        signal_series_risky, pJ_test, sigmaJ_test,
                                                        sigma_risky_train, train_cov, train_var_s,
                                                        CONSERVATIVE_G)
    save_table(bt2ef_aggr, dirs['tables'], '90_section81_backtest_aggressive')
    save_table(bt2ef_cons, dirs['tables'], '91_section81_backtest_conservative')

    plot_p_vs_pJ(bt2_aggr, pJ_test, dirs['figures'])
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
3. figures/05_portfolio_wealth_and_sharpe.png
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
- tables/12_pJ_portfolio_summary.csv
- tables/30_distribution_fit_parameters.csv
- tables/43_bayes_priors.csv
- tables/46_bayes_critical_points.csv
- tables/51_top10_support.csv
- tables/52_top10_confidence.csv
- tables/53_top10_geom_mean.csv
- tables/54_top10_rpf.csv
- tables/110_final_wealth_summary.csv
"""
    (dirs['root'] / 'README_outputs.txt').write_text(readme_text, encoding='utf-8')

    log_and_print(logger, f'All outputs saved under: {dirs["root"].resolve()}')
    log_and_print(logger, 'Workflow finished successfully.')


if __name__ == '__main__':
    main()
