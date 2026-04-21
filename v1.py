# MSDM5058 Project II full workflow
# Requires: yfinance pandas numpy scipy matplotlib statsmodels scikit-learn
# Optional: pip install yfinance pandas numpy scipy matplotlib statsmodels scikit-learn

import warnings
warnings.filterwarnings('ignore')

import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit, brentq
from scipy.stats import norm, gaussian_kde

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance first: pip install yfinance")

# ============================================================
# 0. USER SETTINGS
# ============================================================
RISKY_CANDIDATE = "AAPL"   # candidate only; final labels are decided by volatility in training set
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
PLOT_DPI = 150

# ============================================================
# 1. DATA DOWNLOAD AND PREPROCESSING
# ============================================================

def download_two_stocks(ticker1: str, ticker2: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    data = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
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


def decide_risky_safe(close: pd.DataFrame, ret: pd.DataFrame) -> Tuple[str, str]:
    train_ret, _ = split_3_to_1(ret)
    vols = train_ret.std()
    risky = vols.idxmax()
    safe = vols.idxmin()
    if risky == safe:
        raise ValueError("Could not distinguish risky and safe stocks.")
    return risky, safe

# ============================================================
# 2. MARKOWITZ MINIMUM-RISK PORTFOLIO
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


def portfolio_from_dynamic_weights(close: pd.DataFrame, ret2: pd.DataFrame, w_risky: pd.Series, risky: str, safe: str,
                                   rf_daily: float = DAILY_RF) -> pd.DataFrame:
    # Use w(t-1) to get return on day t; avoids look-ahead bias
    aligned_w = w_risky.reindex(ret2.index).copy()
    w_prev = aligned_w.shift(1)
    port_ret = w_prev * ret2[risky] + (1.0 - w_prev) * ret2[safe]
    port_ret = port_ret.dropna()
    wealth = pd.Series(index=port_ret.index, dtype=float)
    wealth.iloc[0] = 1.0
    for i in range(1, len(port_ret)):
        wealth.iloc[i] = wealth.iloc[i-1] * math.exp(port_ret.iloc[i])
    sharpe_roll = ((port_ret.rolling(60).mean() - rf_daily) / port_ret.rolling(60).std(ddof=1)) * np.sqrt(252)
    out = pd.DataFrame({
        'portfolio_return': port_ret,
        'portfolio_wealth': wealth,
        'rolling_sharpe_60d': sharpe_roll
    })
    return out

# ============================================================
# 3. MA / MACD
# ============================================================

def add_moving_averages(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    df = pd.DataFrame({'price': price.copy()})
    for w in windows:
        df[f'SMA_{w}'] = price.rolling(w).mean()
        df[f'EMA_{w}'] = price.ewm(span=w, adjust=False).mean()
    return df


def add_macd(price: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    out = pd.DataFrame({'price': price, 'macd': macd, 'signal': signal_line, 'hist': hist})
    out['cross_signal'] = np.sign(out['macd'] - out['signal']).diff()
    out['cross_zero'] = np.sign(out['macd']).diff()
    return out

# ============================================================
# 4. PDF / CDF FITTING
# ============================================================

def empirical_cdf(x: np.ndarray):
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
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    xs, F = empirical_cdf(x)

    # better than using L(x*) and L'(0) manually: fit directly to empirical CDF
    p0 = [np.median(x), 1.0 / max(np.std(x, ddof=1), 1e-6)]
    try:
        popt, _ = curve_fit(logistic_cdf, xs, F, p0=p0, maxfev=20000)
        x0, b = float(popt[0]), float(abs(popt[1]))
    except Exception:
        x0 = float(np.median(x))
        b = float(1.0 / max(np.std(x, ddof=1), 1e-6))

    return {'mu': mu, 'sigma': sigma, 'x0': x0, 'b': b}

# ============================================================
# 4.1 DIGITIZATION + CONDITIONAL PDFs
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
            return norm.pdf(xgrid, loc=self.params['mu'], scale=self.params['sigma'])
        if self.kind == 'logistic':
            return logistic_pdf(xgrid, self.params['x0'], self.params['b'])
        if self.kind == 'kde':
            vals = self.kde.evaluate(xgrid)
            return np.maximum(vals, 1e-12)
        raise ValueError('Unknown density kind')

    def cdf(self, xgrid: np.ndarray) -> np.ndarray:
        if self.kind == 'normal':
            return norm.cdf(xgrid, loc=self.params['mu'], scale=self.params['sigma'])
        if self.kind == 'logistic':
            return logistic_cdf(xgrid, self.params['x0'], self.params['b'])
        if self.kind == 'kde':
            vals = np.array([self.kde.integrate_box_1d(-np.inf, xx) for xx in xgrid])
            return np.clip(vals, 0.0, 1.0)
        raise ValueError('Unknown density kind')


def fit_conditional_density(x: pd.Series, kind: str = 'kde') -> ConditionalDensity:
    x = x.dropna().values
    if len(x) < 5:
        # fallback to normal if sample too small
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
# 5. BAYES DETECTOR
# ============================================================

def bayes_detector_setup(x_t: pd.Series, y_next: pd.Series, density_kind='kde'):
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
# 6. ASSOCIATION RULES
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
                'RPF': support * (confidence ** 2),  # consistent with project note when lambda = 2/3
            })
    return pd.DataFrame(rows)


def choose_best_rule_signal(history5: Tuple[str, ...], rules: pd.DataFrame) -> Optional[str]:
    sub = rules[rules['antecedent'] == ''.join(history5)]
    if sub.empty:
        return None
    return sub.sort_values(['confidence', 'support'], ascending=False).iloc[0]['consequent']

# ============================================================
# 7. TRADING SIGNALS
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
# 8. BACKTESTS
# ============================================================

def backtest_one_stock_money(test_price: pd.Series, test_ret: pd.Series, macd_df: pd.DataFrame,
                             priors, cond, y_all: pd.Series, rules: pd.DataFrame,
                             g: float, initial_capital: float = INITIAL_CAPITAL,
                             rf_daily: float = DAILY_RF) -> pd.DataFrame:
    dates = test_price.index
    M = initial_capital
    N = 0.0
    out = []

    for i, dt in enumerate(dates):
        # money compounds at the beginning of each day
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
        if action > 0 and M > 0:
            spend = g * M
            shares = spend / price
            M -= spend
            N += shares
        elif action < 0 and N > 0:
            sell = g * N
            M += sell * price
            N -= sell

        V = M + N * price
        out.append({
            'date': dt, 'money': M, 'shares': N, 'price': price, 'wealth': V,
            'macd_signal': macd_s, 'bayes_signal': bayes_s, 'rule_signal': rule_s,
            'action': action, 'g': g
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

    for i, dt in enumerate(dates):
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
        if action > 0 and M > 0:
            spend = g_today * M
            N += spend / price
            M -= spend
        elif action < 0 and N > 0:
            sell = g_today * N
            M += sell * price
            N -= sell

        V = M + N * price
        out.append({'date': dt, 'money': M, 'shares': N, 'price': price, 'wealth': V,
                    'macd_signal': macd_s, 'bayes_signal': bayes_s, 'rule_signal': rule_s,
                    'action': action, 'g_today': g_today})
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
        if sig > 0 and Ns > 0:
            sell_safe = g * Ns
            cash_equiv = sell_safe * ps
            buy_risky = cash_equiv / pr
            Ns -= sell_safe
            Nr += buy_risky
        elif sig < 0 and Nr > 0:
            sell_risky = g * Nr
            cash_equiv = sell_risky * pr
            buy_safe = cash_equiv / ps
            Nr -= sell_risky
            Ns += buy_safe
        V = Nr * pr + Ns * ps
        p_curr = (Nr * pr) / V if V > 0 else np.nan
        out.append({'date': dt, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V, 'p_curr': p_curr, 'signal': sig})
    return pd.DataFrame(out).set_index('date')


def portfolio_sigma_from_weight(p: float, var_r: float, var_s: float, cov_rs: float) -> float:
    return float(np.sqrt(max(p**2 * var_r + (1-p)**2 * var_s + 2*p*(1-p)*cov_rs, 0.0)))


def weight_from_target_sigma(target_sigma: float, var_r: float, var_s: float, cov_rs: float, p_min=0.0, p_max=1.0):
    grid = np.linspace(p_min, p_max, 5000)
    sigs = np.array([portfolio_sigma_from_weight(p, var_r, var_s, cov_rs) for p in grid])
    idx = np.argmin(np.abs(sigs - target_sigma))
    return float(grid[idx])


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

        if not np.isnan(pJ_t) and not np.isnan(sigmaJ_t):
            sigma_curr = portfolio_sigma_from_weight(p_curr, sigma_risky_train**2, train_var_safe, train_cov)
            if sig < 0:
                target_sigma = sigma_curr - g * (sigma_curr - sigmaJ_t)
            elif sig > 0:
                target_sigma = sigma_curr + g * (sigma_risky_train - sigma_curr)
            else:
                target_sigma = sigma_curr

            target_sigma = float(np.clip(target_sigma, min(sigmaJ_t, sigma_risky_train), max(sigmaJ_t, sigma_risky_train)))
            p_target = weight_from_target_sigma(target_sigma, sigma_risky_train**2, train_var_safe, train_cov, p_min=pJ_t, p_max=1.0)

            Nr = (p_target * V) / pr
            Ns = ((1.0 - p_target) * V) / ps
            V = Nr * pr + Ns * ps
            p_curr = (Nr * pr) / V if V > 0 else np.nan

        out.append({'date': dt, 'N_risky': Nr, 'N_safe': Ns, 'wealth': V, 'p_curr': p_curr,
                    'signal': sig, 'pJ_t': pJ_t, 'sigmaJ_t': sigmaJ_t})
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

        # order matters: sell first, buy later
        if sr < 0 and Nr > 0:
            sell = g * Nr
            M += sell * pr
            Nr -= sell
        if ss < 0 and Ns > 0:
            sell = g * Ns
            M += sell * ps
            Ns -= sell

        if sr > 0 and M > 0:
            spend = g * M if ss <= 0 else 0.5 * g * M
            Nr += spend / pr
            M -= spend
        if ss > 0 and M > 0:
            spend = g * M if sr <= 0 else 0.5 * g * M
            Ns += spend / ps
            M -= spend

        V = M + Nr * pr + Ns * ps
        out.append({'date': dt, 'money': M, 'N_risky': Nr, 'N_safe': Ns,
                    'wealth': V, 'signal_risky': sr, 'signal_safe': ss})
    return pd.DataFrame(out).set_index('date')

# ============================================================
# 9. PLOTTING HELPERS
# ============================================================

def nice_plot_style():
    plt.rcParams['figure.dpi'] = PLOT_DPI
    plt.rcParams['axes.grid'] = True


def plot_prices_and_returns(close: pd.DataFrame, ret: pd.DataFrame, risky: str, safe: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    close[[risky, safe]].plot(ax=axes[0], title='Adjusted Close Prices')
    ret[[risky, safe]].plot(ax=axes[1], title='Daily Log Returns')
    plt.tight_layout()
    plt.show()


def plot_pJ(all_pJ: Dict[str, pd.Series]):
    plt.figure(figsize=(12, 5))
    for label, s in all_pJ.items():
        s.plot(label=label)
    plt.title('Minimum-risk weight p_J(t,h)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_portfolios(portfolios: Dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for name, df in portfolios.items():
        df['portfolio_wealth'].plot(ax=axes[0], label=name)
        df['rolling_sharpe_60d'].plot(ax=axes[1], label=name)
    axes[0].set_title('S_J(t,h) / Wealth proxy')
    axes[1].set_title('Rolling Sharpe Ratio (60d)')
    axes[0].legend(); axes[1].legend()
    plt.tight_layout(); plt.show()


def plot_ma_macd(ma_df: pd.DataFrame, macd_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cols = [c for c in ma_df.columns if c != 'price']
    ma_df[['price'] + cols].plot(ax=axes[0])
    axes[0].set_title('Price with SMA/EMA')
    macd_df[['macd', 'signal', 'hist']].plot(ax=axes[1])
    axes[1].set_title('MACD / Signal / Histogram')
    plt.tight_layout(); plt.show()


def plot_pdf_cdf_fit(x: pd.Series, pars: Dict[str, float]):
    x = x.dropna().values
    xs, F = empirical_cdf(x)
    grid = np.linspace(np.min(xs), np.max(xs), 800)
    G = norm.cdf(grid, loc=pars['mu'], scale=pars['sigma'])
    L = logistic_cdf(grid, pars['x0'], pars['b'])

    plt.figure(figsize=(10, 5))
    plt.plot(xs, F, label='Empirical CDF')
    plt.plot(grid, G, label='Normal CDF fit')
    plt.plot(grid, L, label='Logistic CDF fit')
    plt.title('CDF fits for risky-stock returns')
    plt.legend(); plt.tight_layout(); plt.show()


def plot_conditional_cdf_pdf(x_t: pd.Series, y_next: pd.Series, cond: Dict[str, ConditionalDensity]):
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
    axes[0].set_title('Conditional CDFs'); axes[1].set_title('Conditional PDFs')
    axes[0].legend(); axes[1].legend()
    plt.tight_layout(); plt.show()


def plot_bayes_boundaries(priors, cond, x_t):
    xmin, xmax = float(x_t.min()), float(x_t.max())
    crit, grid, labels = critical_points_bayes(xmin, xmax, priors, cond)
    plt.figure(figsize=(10, 5))
    for lab in ['D', 'U', 'H']:
        plt.plot(grid, priors[lab] * cond[lab].pdf(grid), label=f'q(y)f_y(x), y={lab}')
    for c in crit:
        plt.axvline(c, linestyle='--', alpha=0.5, color='k')
    plt.title('Bayes detector score curves and critical points')
    plt.legend(); plt.tight_layout(); plt.show()
    return crit


def plot_backtests_one_stock(base_curve: pd.Series, bt_aggr: pd.DataFrame, bt_cons: pd.DataFrame, bt_bob: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    base_curve.plot(label='Minimum-risk portfolio / cash benchmark')
    bt_aggr['wealth'].plot(label='Aggressive greed')
    bt_cons['wealth'].plot(label='Conservative greed')
    bt_bob['wealth'].plot(label='Bob dynamic greed')
    plt.title('Section 7 backtest')
    plt.legend(); plt.tight_layout(); plt.show()


def plot_backtests_two_stock(base_curve: pd.Series, bt_aggr: pd.DataFrame, bt_cons: pd.DataFrame, title: str):
    plt.figure(figsize=(12, 5))
    base_curve.plot(label='Minimum-risk portfolio')
    bt_aggr['wealth'].plot(label='Aggressive greed')
    bt_cons['wealth'].plot(label='Conservative greed')
    plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    nice_plot_style()

    # ---------- download ----------
    close_raw = download_two_stocks(RISKY_CANDIDATE, SAFE_CANDIDATE, START_DATE, END_DATE)
    ret_raw = make_returns(close_raw)

    # align because returns lose first row
    close = close_raw.loc[ret_raw.index].copy()

    # decide risky / safe empirically from training volatility
    risky, safe = decide_risky_safe(close, ret_raw)
    print(f"Final label from training-set volatility: risky = {risky}, safe = {safe}")

    # train / test split (3:1)
    train_close, test_close = split_3_to_1(close)
    train_ret, test_ret = split_3_to_1(ret_raw)

    # ---------- section 1 ----------
    plot_prices_and_returns(close, ret_raw, risky, safe)

    # ---------- section 2 ----------
    pJ_dict = {}
    port_dict = {}
    for h in H_WINDOWS:
        label = f"h={h if h is not np.inf else 'inf'}"
        pJ = rolling_min_var_weights(train_ret[[risky, safe]], risky, safe, h)
        pJ_dict[label] = pJ
        port_dict[label] = portfolio_from_dynamic_weights(train_close.loc[pJ.dropna().index],
                                                          train_ret.loc[pJ.dropna().index, [risky, safe]],
                                                          pJ.dropna(), risky, safe)
    plot_pJ(pJ_dict)
    plot_portfolios(port_dict)

    # choose one pJ series for later trading initialization; h=300 is a good default
    pJ_train = pJ_dict['h=300'].copy()
    if pJ_train.dropna().empty:
        pJ_train = pJ_dict['h=inf'].copy()

    # extend pJ into test period with rolling estimation on all observed history up to t
    all_ret = ret_raw[[risky, safe]].copy()
    pJ_all_300 = rolling_min_var_weights(all_ret, risky, safe, 300)
    pJ_test = pJ_all_300.reindex(test_ret.index)

    # sigma_J(t) using current estimated variance formula
    sigmaJ_test = pd.Series(index=test_ret.index, dtype=float)
    for dt in test_ret.index:
        idx = all_ret.index.get_loc(dt)
        window = all_ret.iloc[max(0, idx-300+1):idx+1]
        vr = window[risky].var(ddof=1)
        vs = window[safe].var(ddof=1)
        cov = window[[risky, safe]].cov().loc[risky, safe]
        p = pJ_all_300.loc[dt]
        sigmaJ_test.loc[dt] = portfolio_sigma_from_weight(p, vr, vs, cov)

    # ---------- section 3 ----------
    ma_df = add_moving_averages(train_close[risky], MA_WINDOWS)
    macd_train = add_macd(train_close[risky], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    plot_ma_macd(ma_df, macd_train)

    # ---------- section 4 ----------
    fit_pars = fit_normal_and_logistic(train_ret[risky])
    print("Section 4 parameters:")
    print(f"Normal fit: mu={fit_pars['mu']:.8f}, sigma={fit_pars['sigma']:.8f}")
    print(f"Logistic fit: x*={fit_pars['x0']:.8f}, b={fit_pars['b']:.8f}")
    plot_pdf_cdf_fit(train_ret[risky], fit_pars)

    # ---------- section 4.1 ----------
    eps = EPS_SCALE * train_ret[risky].std(ddof=1)
    print(f"Digitization threshold epsilon = {eps:.8f}")
    Y_all = digitize_returns(ret_raw[risky], eps)
    Y_train = Y_all.loc[train_ret.index]

    # X(t), Y(t+1)
    X_t_train = train_ret[risky].iloc[:-1]
    Y_next_train = Y_train.shift(-1).iloc[:-1]

    density_kind = 'kde' if USE_KDE_FOR_CONDITIONAL_PDF else 'normal'
    priors, cond = bayes_detector_setup(X_t_train, Y_next_train, density_kind)
    print("Section 5 priors:", priors)
    plot_conditional_cdf_pdf(X_t_train, Y_next_train, cond)
    crit = plot_bayes_boundaries(priors, cond, X_t_train)
    print("Bayes critical points:", crit)

    # ---------- section 6 ----------
    rules = make_rules(Y_train, k=5)
    print("Top 10 by support")
    print(rules.sort_values(['support', 'confidence'], ascending=False).head(10)[['antecedent', 'consequent', 'support', 'confidence']])
    print("Top 10 by confidence")
    print(rules.sort_values(['confidence', 'support'], ascending=False).head(10)[['antecedent', 'consequent', 'support', 'confidence']])
    print("Top 10 by geometric mean")
    print(rules.sort_values(['geom_mean', 'support'], ascending=False).head(10)[['antecedent', 'consequent', 'geom_mean', 'support', 'confidence']])
    print("Top 10 by RPF")
    print(rules.sort_values(['RPF', 'support'], ascending=False).head(10)[['antecedent', 'consequent', 'RPF', 'support', 'confidence']])

    # ---------- build test-period signals ----------
    macd_all_risky = add_macd(close[risky], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    signal_series_risky = pd.Series(index=test_close.index, dtype=int)
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
        signal_series_risky.loc[dt] = combined_signal(macd_s, bayes_s, rule_s)

    # safer stock signals for section 9: reuse same framework on safer stock
    macd_all_safe = add_macd(close[safe], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    Y_all_safe = digitize_returns(ret_raw[safe], EPS_SCALE * train_ret[safe].std(ddof=1))
    X_t_train_s = train_ret[safe].iloc[:-1]
    Y_next_train_s = Y_all_safe.loc[train_ret.index].shift(-1).iloc[:-1]
    priors_s, cond_s = bayes_detector_setup(X_t_train_s, Y_next_train_s, density_kind)
    rules_s = make_rules(Y_all_safe.loc[train_ret.index], k=5)

    signal_series_safe = pd.Series(index=test_close.index, dtype=int)
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
        signal_series_safe.loc[dt] = combined_signal(macd_s, bayes_s, rule_s)

    # ---------- section 7 ----------
    bt_aggr = backtest_one_stock_money(test_close[risky], test_ret[risky], macd_all_risky,
                                       priors, cond, Y_all, rules, AGGRESSIVE_G)
    bt_cons = backtest_one_stock_money(test_close[risky], test_ret[risky], macd_all_risky,
                                       priors, cond, Y_all, rules, CONSERVATIVE_G)
    bt_bob = backtest_bob_dynamic_greed(test_close[risky], test_ret[risky], macd_all_risky,
                                        priors, cond, Y_all, rules,
                                        AGGRESSIVE_G, CONSERVATIVE_G)
    base_cash = pd.Series(INITIAL_CAPITAL * ((1 + DAILY_RF) ** np.arange(len(test_close))), index=test_close.index)
    plot_backtests_one_stock(base_cash, bt_aggr, bt_cons, bt_bob)

    # ---------- section 8 ----------
    pJ0 = float(pJ_test.dropna().iloc[0]) if not pJ_test.dropna().empty else 0.5
    bt2_aggr = backtest_two_stocks_original(test_close[[risky, safe]], risky, safe, signal_series_risky, pJ0, AGGRESSIVE_G)
    bt2_cons = backtest_two_stocks_original(test_close[[risky, safe]], risky, safe, signal_series_risky, pJ0, CONSERVATIVE_G)

    # baseline minimum-risk portfolio in test period: dynamic pJ(t)
    base_minrisk = pd.Series(index=test_ret.index, dtype=float)
    base_minrisk.iloc[0] = INITIAL_CAPITAL
    pJ_prev = pJ_test.shift(1)
    test_port_ret = (pJ_prev * test_ret[risky] + (1-pJ_prev) * test_ret[safe]).dropna()
    for i in range(1, len(test_port_ret)):
        base_minrisk.iloc[i] = base_minrisk.iloc[i-1] * math.exp(test_port_ret.iloc[i])
    base_minrisk = base_minrisk.dropna()

    plot_backtests_two_stock(base_minrisk, bt2_aggr.reindex(base_minrisk.index), bt2_cons.reindex(base_minrisk.index),
                             'Section 8: Two stocks, original trading scheme')

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

    plt.figure(figsize=(12, 4))
    bt2_aggr['p_curr'].plot(label='p(t): original scheme')
    pJ_test.plot(label='p_J(t): min-risk fraction')
    plt.title('Section 8.1: p(t) vs p_J(t)')
    plt.legend(); plt.tight_layout(); plt.show()

    plot_backtests_two_stock(base_minrisk, bt2ef_aggr.reindex(base_minrisk.index), bt2ef_cons.reindex(base_minrisk.index),
                             'Section 8.1: Two stocks with efficient-frontier control')

    # ---------- section 9 ----------
    bt3_aggr = backtest_two_stocks_money(test_close[[risky, safe]], risky, safe,
                                         signal_series_risky, signal_series_safe,
                                         AGGRESSIVE_G)
    bt3_cons = backtest_two_stocks_money(test_close[[risky, safe]], risky, safe,
                                         signal_series_risky, signal_series_safe,
                                         CONSERVATIVE_G)
    plt.figure(figsize=(12, 5))
    bt3_aggr['wealth'].plot(label='Aggressive greed')
    bt3_cons['wealth'].plot(label='Conservative greed')
    plt.title('Section 9: Two stocks + money')
    plt.legend(); plt.tight_layout(); plt.show()

    # ---------- summary ----------
    print("\nFinal wealth summary")
    print(f"Section 7 aggressive   : {bt_aggr['wealth'].iloc[-1]:.2f}")
    print(f"Section 7 conservative : {bt_cons['wealth'].iloc[-1]:.2f}")
    print(f"Section 7 Bob          : {bt_bob['wealth'].iloc[-1]:.2f}")
    print(f"Section 8 aggressive   : {bt2_aggr['wealth'].iloc[-1]:.2f}")
    print(f"Section 8 conservative : {bt2_cons['wealth'].iloc[-1]:.2f}")
    print(f"Section 8.1 aggressive : {bt2ef_aggr['wealth'].iloc[-1]:.2f}")
    print(f"Section 8.1 conservative: {bt2ef_cons['wealth'].iloc[-1]:.2f}")
    print(f"Section 9 aggressive   : {bt3_aggr['wealth'].iloc[-1]:.2f}")
    print(f"Section 9 conservative : {bt3_cons['wealth'].iloc[-1]:.2f}")


if __name__ == '__main__':
    main()
