"""
QuantFolio — Quantitative Portfolio Optimization & Factor Analysis Engine
Built by Marin Xhemollari | marinxhemollari.com

v3.0 — Adds Fama-French factor regression and market regime analysis.
v3.0.1 — Dark-mode contrast boost for visible animations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
import io
import urllib.request
import zipfile

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="QuantFolio",
    page_icon="https://marinxhemollari.com/frog-favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS — Charcoal / Emerald Theme (dark-mode-boosted)
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    --charcoal-900: #0a0a0a;
    --charcoal-800: #0d0d0d;
    --charcoal-700: #141414;
    --charcoal-600: #1a1a1a;
    --charcoal-500: #222222;
    --charcoal-400: #2a2a2a;
    --charcoal-300: #333333;
    --emerald-500: #2ecc71;
    --emerald-400: #27ae60;
    --emerald-300: #1abc9c;
    --text-primary: #e8e8e8;
    --text-secondary: rgba(232, 232, 232, 0.6);
    --text-muted: rgba(232, 232, 232, 0.35);
    --glass-bg: rgba(22, 22, 22, 0.7);
    --glass-border: rgba(46, 204, 113, 0.22);
    --glass-border-hover: rgba(46, 204, 113, 0.40);
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 24px rgba(46, 204, 113, 0.18); }
    50%      { box-shadow: 0 0 48px rgba(46, 204, 113, 0.40); }
}
@keyframes shimmerSweep {
    0%   { background-position: -150% center; }
    100% { background-position: 250% center; }
}
@keyframes borderBreathe {
    0%, 100% { border-color: rgba(46, 204, 113, 0.22); }
    50%      { border-color: rgba(46, 204, 113, 0.42); }
}
@keyframes logoFloat {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--charcoal-900) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main .block-container {
    padding-top: 1rem !important;
    max-width: 1400px;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--charcoal-800); }
::-webkit-scrollbar-thumb { background: var(--charcoal-400); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--emerald-400); }

.qf-header {
    background: linear-gradient(135deg,
        var(--charcoal-800) 0%, rgba(46, 204, 113, 0.22) 25%,
        var(--charcoal-700) 50%, rgba(26, 188, 156, 0.22) 75%,
        var(--charcoal-800) 100%);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite,
               fadeInUp 0.8s ease-out,
               borderBreathe 6s ease-in-out infinite;
    border: 1px solid rgba(46, 204, 113, 0.28);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 0 40px rgba(46, 204, 113, 0.15),
                inset 0 1px 0 rgba(46, 204, 113, 0.10);
}
.qf-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(46, 204, 113, 0.14) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
}
.qf-header::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(105deg,
        transparent 40%,
        rgba(46, 204, 113, 0.14) 50%,
        transparent 60%);
    background-size: 200% 100%;
    animation: shimmerSweep 7s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.qf-header-content {
    display: flex; align-items: center; gap: 2.5rem;
    position: relative; z-index: 2; min-width: 0;
}
.qf-logo {
    width: 64px; height: 64px;
    animation: logoFloat 4s ease-in-out infinite;
    filter: drop-shadow(0 0 14px rgba(46, 204, 113, 0.55));
    flex-shrink: 0;
}
.qf-title-wrap { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.qf-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1.1; margin: 0;
    display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.25rem;
}
.qf-title-plain { color: var(--text-primary); }
.qf-title-accent {
    color: var(--emerald-500);
    background: linear-gradient(135deg, #2ecc71 0%, #1abc9c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 30px rgba(46, 204, 113, 0.3);
}
.qf-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem; color: var(--text-secondary);
    margin-top: 0.35rem; font-weight: 300; letter-spacing: 0.02em;
}
.qf-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--emerald-500);
    background: rgba(46, 204, 113, 0.12);
    border: 1px solid rgba(46, 204, 113, 0.25);
    padding: 0.2rem 0.6rem; border-radius: 4px;
    letter-spacing: 0.05em;
    align-self: center; flex-shrink: 0; margin-left: 0.75rem;
    box-shadow: 0 0 12px rgba(46, 204, 113, 0.15);
}

h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
    animation: fadeInUp 0.6s ease-out;
}
h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

div[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.5s ease-out, pulseGlow 4s ease-in-out infinite;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--glass-border-hover) !important;
    background: rgba(26, 26, 26, 0.85) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(46, 204, 113, 0.25);
}
div[data-testid="stMetric"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important; font-weight: 600 !important;
    color: var(--emerald-500) !important;
    white-space: nowrap !important; overflow: visible !important;
    text-shadow: 0 0 20px rgba(46, 204, 113, 0.25);
}

section[data-testid="stSidebar"] {
    background: var(--charcoal-800) !important;
    border-right: 1px solid rgba(46, 204, 113, 0.15) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: var(--text-muted) !important; margin-top: 1.5rem !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stDateInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--emerald-400), var(--emerald-300)) !important;
    color: var(--charcoal-900) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(46, 204, 113, 0.35) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(46, 204, 113, 0.55) !important;
}
.stDownloadButton > button {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--emerald-500) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
    border-color: var(--emerald-500) !important;
    background: rgba(46, 204, 113, 0.1) !important;
    box-shadow: 0 0 20px rgba(46, 204, 113, 0.2) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important; background: var(--charcoal-700) !important;
    border-radius: 10px !important; padding: 4px !important;
    border: 1px solid rgba(46, 204, 113, 0.12);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    border-radius: 8px !important; padding: 8px 16px !important;
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(46, 204, 113, 0.18) !important;
    color: var(--emerald-500) !important;
    box-shadow: 0 0 16px rgba(46, 204, 113, 0.2);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important; overflow: hidden;
    animation: fadeIn 0.5s ease-out;
}

hr {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(46, 204, 113, 0.30), transparent) !important;
    margin: 2rem 0 !important;
}

.qf-strategy-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 0.25rem 0.65rem; border-radius: 4px;
    display: inline-block; margin-bottom: 0.5rem;
}
.qf-strategy-sharpe { color: #22c55e; background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.3); }
.qf-strategy-minvar { color: #f59e0b; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); }
.qf-strategy-rp { color: #ec4899; background: rgba(236, 72, 153, 0.12); border: 1px solid rgba(236, 72, 153, 0.3); }

.qf-regime-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    padding: 0.2rem 0.5rem; border-radius: 4px;
    display: inline-block;
}
.qf-regime-bull { color: #22c55e; background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25); }
.qf-regime-bear { color: #ef4444; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); }
.qf-regime-highvol { color: #f59e0b; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25); }

blockquote {
    border-left: 3px solid var(--emerald-500) !important;
    background: rgba(46, 204, 113, 0.06) !important;
    padding: 0.8rem 1.2rem !important; border-radius: 0 8px 8px 0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    box-shadow: 0 0 16px rgba(46, 204, 113, 0.08);
}

.qf-footer {
    text-align: center; color: var(--text-muted);
    font-family: 'DM Sans', sans-serif; font-size: 0.78rem;
    padding: 2rem 0 1rem; animation: fadeIn 0.8s ease-out;
}
.qf-footer a { color: var(--emerald-500); text-decoration: none; }
.qf-footer a:hover { color: var(--emerald-300); }
.qf-footer-mono {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: var(--text-muted); opacity: 0.6; margin-top: 0.5rem;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid var(--glass-border); border-radius: 12px;
    overflow: hidden; animation: fadeIn 0.5s ease-out;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: var(--glass-border-hover);
    box-shadow: 0 4px 32px rgba(46, 204, 113, 0.15);
}

[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(46, 204, 113, 0.15) !important;
    border: 1px solid rgba(46, 204, 113, 0.3) !important;
    color: var(--emerald-500) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 6px !important;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    animation: fadeIn 0.4s ease-out;
}

@media (max-width: 768px) {
    .qf-header { padding: 1.5rem 1.5rem; }
    .qf-header-content { gap: 1.25rem; flex-wrap: wrap; }
    .qf-logo { width: 48px; height: 48px; }
    .qf-title { font-size: 2rem; }
    .qf-subtitle { font-size: 0.85rem; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
TRADING_DAYS = 252

POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "BAC",
    "XOM", "KO", "PEP", "COST", "ABBV", "MRK", "CRM", "AMD",
    "NFLX", "DIS", "INTC", "CSCO", "QCOM", "TXN", "SPY", "QQQ",
    "VTI", "IWM", "GLD", "SLV", "TLT", "BND",
]

POPULAR_CRYPTO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
    "DOT-USD", "LINK-USD", "MATIC-USD", "ATOM-USD",
]

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

# ──────────────────────────────────────────────────────────────
# PORTFOLIO MATH
# ──────────────────────────────────────────────────────────────

def calc_portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    ret, vol = calc_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return calc_portfolio_performance(weights, mean_returns, cov_matrix)[1]


def find_optimal_portfolio(mean_returns, cov_matrix, risk_free_rate, objective="sharpe"):
    n_assets = len(mean_returns)
    initial_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    if objective == "sharpe":
        result = minimize(neg_sharpe_ratio, initial_weights,
                          args=(mean_returns, cov_matrix, risk_free_rate),
                          method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        result = minimize(portfolio_volatility, initial_weights,
                          args=(mean_returns, cov_matrix),
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def find_risk_parity_portfolio(cov_matrix):
    n_assets = cov_matrix.shape[0]
    initial_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0.01, 1.0) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    def risk_parity_objective(weights):
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights / port_vol
        risk_contrib = weights * marginal_contrib
        target = port_vol / n_assets
        return np.sum((risk_contrib - target) ** 2)
    result = minimize(risk_parity_objective, initial_weights,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def compute_risk_contributions(weights, cov_matrix):
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights / port_vol
    risk_contrib = weights * marginal_contrib
    return risk_contrib / risk_contrib.sum() * 100


def compute_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, n_points=100):
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    initial_weights = np.ones(n_assets) / n_assets
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier_volatilities = []
    frontier_returns = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        result = minimize(portfolio_volatility, initial_weights,
                          args=(mean_returns, cov_matrix),
                          method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            frontier_volatilities.append(result.fun)
            frontier_returns.append(target)
    return np.array(frontier_volatilities), np.array(frontier_returns)


def run_monte_carlo(mean_returns, cov_matrix, risk_free_rate, n_simulations):
    n_assets = len(mean_returns)
    results_return = np.zeros(n_simulations)
    results_vol = np.zeros(n_simulations)
    results_sharpe = np.zeros(n_simulations)
    results_weights = np.zeros((n_simulations, n_assets))
    for i in range(n_simulations):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol = calc_portfolio_performance(weights, mean_returns, cov_matrix)
        results_return[i] = ret
        results_vol[i] = vol
        results_sharpe[i] = (ret - risk_free_rate) / vol
        results_weights[i] = weights
    return results_return, results_vol, results_sharpe, results_weights


def compute_var_cvar(daily_returns, weights, confidence=0.95):
    portfolio_returns = daily_returns.values @ weights
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return var, cvar


def backtest_portfolio(prices, weights, tickers):
    daily_returns = prices[tickers].pct_change().dropna()
    portfolio_returns = daily_returns.values @ weights
    cumulative = (1 + portfolio_returns).cumprod() * 100
    return pd.Series(cumulative, index=daily_returns.index, name="Portfolio")


def calc_backtest_stats(curve, risk_free):
    returns = curve.pct_change().dropna()
    total_return = (curve.iloc[-1] / 100 - 1) * 100
    ann_return = ((curve.iloc[-1] / 100) ** (TRADING_DAYS / len(returns)) - 1) * 100
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS) * 100
    sharpe = (ann_return / 100 - risk_free) / (ann_vol / 100) if ann_vol > 0 else 0
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS) * 100 if len(downside) > 0 else ann_vol
    sortino = (ann_return / 100 - risk_free) / (downside_vol / 100) if downside_vol > 0 else 0
    running_max = curve.cummax()
    max_dd = ((curve - running_max) / running_max).min() * 100
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    return {
        "Total Return": f"{total_return:.1f}%",
        "Ann. Return": f"{ann_return:.1f}%",
        "Ann. Volatility": f"{ann_vol:.1f}%",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Sortino Ratio": f"{sortino:.3f}",
        "Max Drawdown": f"{max_dd:.1f}%",
        "Calmar Ratio": f"{calmar:.3f}",
    }


# ──────────────────────────────────────────────────────────────
# FAMA-FRENCH FACTOR DATA & REGRESSION (v3.0)
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_ff_factors(model="FF5"):
    try:
        url = FF5_URL if model == "FF5" else FF3_URL
        with urllib.request.urlopen(url, timeout=30) as response:
            zip_data = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                raw = f.read().decode("utf-8", errors="ignore")

        lines = raw.split("\n")
        start_idx = None
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[0] and parts[0][0].isdigit() and len(parts[0]) >= 8:
                start_idx = i
                break
        if start_idx is None:
            return None
        header_idx = start_idx - 1
        while header_idx >= 0 and not lines[header_idx].strip():
            header_idx -= 1

        end_idx = start_idx
        while end_idx < len(lines):
            line = lines[end_idx].strip()
            if not line:
                break
            parts = [p.strip() for p in line.split(",")]
            if not (parts[0] and parts[0][0].isdigit() and len(parts[0]) >= 8):
                break
            end_idx += 1

        header_line = lines[header_idx]
        data_lines = lines[start_idx:end_idx]
        csv_text = header_line + "\n" + "\n".join(data_lines)

        df = pd.read_csv(io.StringIO(csv_text))
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df / 100.0
        return df

    except Exception:
        return None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_momentum_factor():
    try:
        with urllib.request.urlopen(MOM_URL, timeout=30) as response:
            zip_data = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                raw = f.read().decode("utf-8", errors="ignore")
        lines = raw.split("\n")
        start_idx = None
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] and parts[0][0].isdigit() and len(parts[0]) >= 8:
                start_idx = i
                break
        if start_idx is None:
            return None
        header_idx = start_idx - 1
        while header_idx >= 0 and not lines[header_idx].strip():
            header_idx -= 1
        end_idx = start_idx
        while end_idx < len(lines):
            line = lines[end_idx].strip()
            if not line:
                break
            parts = [p.strip() for p in line.split(",")]
            if not (parts[0] and parts[0][0].isdigit() and len(parts[0]) >= 8):
                break
            end_idx += 1
        header_line = lines[header_idx]
        data_lines = lines[start_idx:end_idx]
        csv_text = header_line + "\n" + "\n".join(data_lines)
        df = pd.read_csv(io.StringIO(csv_text))
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df / 100.0
        mom_col = [c for c in df.columns if "Mom" in c or "MOM" in c.upper()]
        if mom_col:
            df = df[mom_col].rename(columns={mom_col[0]: "MOM"})
            return df
        return None
    except Exception:
        return None


def run_factor_regression(portfolio_returns, factor_df, factor_cols):
    merged = pd.concat([portfolio_returns.rename("R_p"), factor_df], axis=1).dropna()
    if len(merged) < 30:
        return None

    excess = merged["R_p"] - merged["RF"]
    X = merged[factor_cols].values
    y = excess.values

    n, k = X.shape
    X_ = np.column_stack([np.ones(n), X])

    try:
        XtX_inv = np.linalg.inv(X_.T @ X_)
    except np.linalg.LinAlgError:
        return None
    beta = XtX_inv @ X_.T @ y
    y_hat = X_ @ beta
    residuals = y - y_hat
    rss = np.sum(residuals ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - rss / tss if tss > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0

    sigma2 = rss / (n - k - 1) if n > k + 1 else rss
    var_beta = sigma2 * np.diag(XtX_inv)
    se_beta = np.sqrt(np.maximum(var_beta, 0))
    t_stats = beta / np.where(se_beta > 0, se_beta, np.nan)

    alpha_daily = beta[0]
    alpha_annual = alpha_daily * TRADING_DAYS

    return {
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "alpha_tstat": t_stats[0],
        "betas": dict(zip(factor_cols, beta[1:])),
        "t_stats": dict(zip(factor_cols, t_stats[1:])),
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "n_obs": n,
        "residuals": pd.Series(residuals, index=merged.index),
        "fitted": pd.Series(y_hat, index=merged.index),
        "actual": pd.Series(y, index=merged.index),
    }


# ──────────────────────────────────────────────────────────────
# REGIME DETECTION (v3.0)
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_proxy(start, end):
    try:
        data = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"]["SPY"].to_frame("SPY")
        return data[["Close"]].rename(columns={"Close": "SPY"})
    except Exception:
        return pd.DataFrame()


def detect_regimes(market_prices, dd_threshold=0.10, vol_lookback=20, vol_threshold_quantile=0.85):
    prices = market_prices["SPY"] if "SPY" in market_prices.columns else market_prices.iloc[:, 0]
    returns = prices.pct_change().dropna()

    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max

    rolling_vol = returns.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS)
    vol_threshold = rolling_vol.quantile(vol_threshold_quantile)

    regime = pd.Series("BULL", index=prices.index)
    regime[drawdown < -dd_threshold] = "BEAR"
    regime[rolling_vol > vol_threshold] = "HIGH_VOL"

    return regime, drawdown, rolling_vol


def regime_conditional_stats(portfolio_returns, regimes, risk_free_daily):
    merged = pd.concat([portfolio_returns.rename("R"), regimes.rename("regime")], axis=1).dropna()
    stats = {}
    for reg in ["BULL", "BEAR", "HIGH_VOL"]:
        sub = merged[merged["regime"] == reg]["R"]
        if len(sub) < 5:
            stats[reg] = {"days": len(sub), "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
            continue
        ann_return = sub.mean() * TRADING_DAYS
        ann_vol = sub.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - risk_free_daily * TRADING_DAYS) / ann_vol if ann_vol > 0 else np.nan
        stats[reg] = {
            "days": len(sub),
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        }
    return stats


# ──────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    return prices.dropna()


# ──────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────

COLORS = {
    "frontier":     "#2ecc71",
    "cml":          "#1abc9c",
    "sharpe":       "#22c55e",
    "min_var":      "#f59e0b",
    "risk_parity":  "#ec4899",
    "assets":       "#64748b",
    "equal_weight": "#94a3b8",
    "bg":           "rgba(0,0,0,0)",
    "grid":         "rgba(46, 204, 113, 0.08)",
    "text":         "rgba(232, 232, 232, 0.6)",
    "text_bright":  "rgba(232, 232, 232, 0.85)",
    "bull":         "rgba(34, 197, 94, 0.15)",
    "bear":         "rgba(239, 68, 68, 0.18)",
    "highvol":      "rgba(245, 158, 11, 0.18)",
}

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], size=12, family="DM Sans, sans-serif"),
    margin=dict(l=48, r=24, t=56, b=48),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False,
               linecolor="rgba(46, 204, 113, 0.15)",
               tickfont=dict(family="JetBrains Mono, monospace", size=10)),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False,
               linecolor="rgba(46, 204, 113, 0.15)",
               tickfont=dict(family="JetBrains Mono, monospace", size=10)),
    title_font=dict(family="DM Sans, sans-serif", size=16, color=COLORS["text_bright"]),
    hoverlabel=dict(bgcolor="rgba(14, 14, 14, 0.95)",
                    bordercolor="rgba(46, 204, 113, 0.3)",
                    font=dict(family="JetBrains Mono, monospace", size=12, color="#e8e8e8")),
)

EMERALD_PALETTE = [
    "#2ecc71", "#1abc9c", "#27ae60", "#16a085",
    "#22c55e", "#10b981", "#059669", "#047857",
    "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5",
]


def plot_efficient_frontier(frontier_vol, frontier_ret, mc_vol, mc_ret, mc_sharpe,
                             sharpe_vol, sharpe_ret, minvar_vol, minvar_ret,
                             rp_vol, rp_ret, asset_vols, asset_rets, tickers,
                             risk_free, show_cml, show_rp):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc_vol * 100, y=mc_ret * 100, mode="markers",
        marker=dict(size=2.5, opacity=0.25, color=mc_sharpe,
                    colorscale=[[0, "#0d0d0d"], [0.3, "#1a1a1a"], [0.6, "#1abc9c"], [1, "#2ecc71"]],
                    colorbar=dict(title=dict(text="Sharpe", font=dict(size=11)),
                                  thickness=10, len=0.4,
                                  tickfont=dict(family="JetBrains Mono", size=9))),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=frontier_vol * 100, y=frontier_ret * 100,
                             mode="lines", line=dict(color=COLORS["frontier"], width=3),
                             name="Efficient Frontier"))
    if show_cml:
        cml_x_max = max(asset_vols.max(), sharpe_vol) * 1.3
        cml_x = np.linspace(0, cml_x_max, 100)
        cml_slope = (sharpe_ret - risk_free) / sharpe_vol
        cml_y = risk_free + cml_slope * cml_x
        fig.add_trace(go.Scatter(x=cml_x * 100, y=cml_y * 100, mode="lines",
                                 line=dict(color=COLORS["cml"], width=2, dash="dash"),
                                 name="Capital Market Line"))
        fig.add_trace(go.Scatter(x=[0], y=[risk_free * 100], mode="markers",
                                 marker=dict(size=10, color="white", symbol="x",
                                             line=dict(width=2, color="white")),
                                 name=f"Risk-Free ({risk_free*100:.1f}%)"))
    fig.add_trace(go.Scatter(x=[sharpe_vol * 100], y=[sharpe_ret * 100], mode="markers",
                             marker=dict(size=16, color=COLORS["sharpe"], symbol="star",
                                         line=dict(width=1.5, color="white")),
                             name="Max Sharpe"))
    fig.add_trace(go.Scatter(x=[minvar_vol * 100], y=[minvar_ret * 100], mode="markers",
                             marker=dict(size=16, color=COLORS["min_var"], symbol="diamond",
                                         line=dict(width=1.5, color="white")),
                             name="Min Variance"))
    if show_rp:
        fig.add_trace(go.Scatter(x=[rp_vol * 100], y=[rp_ret * 100], mode="markers",
                                 marker=dict(size=16, color=COLORS["risk_parity"], symbol="hexagon",
                                             line=dict(width=1.5, color="white")),
                                 name="Risk Parity"))
    fig.add_trace(go.Scatter(x=asset_vols * 100, y=asset_rets * 100,
                             mode="markers+text",
                             marker=dict(size=10, color=COLORS["assets"],
                                         line=dict(width=1, color="rgba(255,255,255,0.3)")),
                             text=tickers, textposition="top center",
                             textfont=dict(size=10, color="rgba(232,232,232,0.7)", family="JetBrains Mono"),
                             name="Individual Assets"))
    title = "Efficient Frontier with Capital Market Line" if show_cml else "Efficient Frontier"
    fig.update_layout(**PLOT_LAYOUT, title=title,
                      xaxis_title="Annualized Volatility (%)",
                      yaxis_title="Annualized Return (%)",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="center", x=0.5,
                                  font=dict(size=11, family="DM Sans")),
                      height=580)
    return fig


def plot_correlation_heatmap(corr_matrix, tickers):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=tickers, y=tickers,
        colorscale=[[0, "#0d0d0d"], [0.25, "#1a1a1a"], [0.5, "#333333"],
                     [0.75, "#1abc9c"], [1, "#2ecc71"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont=dict(size=11, family="JetBrains Mono"),
        colorbar=dict(thickness=10, len=0.8,
                      tickfont=dict(family="JetBrains Mono", size=9)),
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Return Correlation Matrix", height=450)
    fig.update_xaxes(side="bottom", tickangle=-45)
    return fig


def plot_weights_pie(weights, tickers, title):
    mask = weights > 0.005
    filtered_weights = weights[mask]
    filtered_tickers = [t for t, m in zip(tickers, mask) if m]
    fig = go.Figure(data=go.Pie(
        labels=filtered_tickers,
        values=np.round(filtered_weights * 100, 2),
        hole=0.5, textinfo="label+percent",
        textfont=dict(size=11, family="JetBrains Mono"),
        marker=dict(colors=EMERALD_PALETTE[:len(filtered_tickers)],
                    line=dict(color="rgba(10,10,10,0.8)", width=2)),
    ))
    fig.update_layout(**PLOT_LAYOUT, title=title, height=380, showlegend=False)
    return fig


def plot_cumulative_returns(prices, tickers):
    normalized = (prices / prices.iloc[0]) * 100
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], mode="lines",
                                 name=ticker,
                                 line=dict(width=2, color=EMERALD_PALETTE[i % len(EMERALD_PALETTE)])))
    fig.update_layout(**PLOT_LAYOUT, title="Cumulative Returns (Normalized to 100)",
                      xaxis_title="Date", yaxis_title="Value", height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_drawdown(prices, tickers):
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        series = prices[ticker]
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max * 100
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name=ticker,
                                 line=dict(width=1.5, color=EMERALD_PALETTE[i % len(EMERALD_PALETTE)]),
                                 fill="tozeroy", opacity=0.6))
    fig.update_layout(**PLOT_LAYOUT, title="Drawdown Analysis",
                      xaxis_title="Date", yaxis_title="Drawdown (%)", height=350,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_backtest(prices, sharpe_w, minvar_w, rp_w, tickers, show_rp):
    eq_w = np.ones(len(tickers)) / len(tickers)
    sharpe_curve = backtest_portfolio(prices, sharpe_w, tickers)
    minvar_curve = backtest_portfolio(prices, minvar_w, tickers)
    rp_curve = backtest_portfolio(prices, rp_w, tickers) if show_rp else None
    equal_curve = backtest_portfolio(prices, eq_w, tickers)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sharpe_curve.index, y=sharpe_curve.values, mode="lines",
                             name="Max Sharpe", line=dict(width=2.5, color=COLORS["sharpe"])))
    fig.add_trace(go.Scatter(x=minvar_curve.index, y=minvar_curve.values, mode="lines",
                             name="Min Variance", line=dict(width=2.5, color=COLORS["min_var"])))
    if show_rp and rp_curve is not None:
        fig.add_trace(go.Scatter(x=rp_curve.index, y=rp_curve.values, mode="lines",
                                 name="Risk Parity", line=dict(width=2.5, color=COLORS["risk_parity"])))
    fig.add_trace(go.Scatter(x=equal_curve.index, y=equal_curve.values, mode="lines",
                             name="Equal Weight", line=dict(width=2, color=COLORS["equal_weight"], dash="dot")))
    fig.update_layout(**PLOT_LAYOUT, title="Portfolio Backtest (Normalized to 100)",
                      xaxis_title="Date", yaxis_title="Portfolio Value", height=450,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="center", x=0.5, font=dict(size=11, family="DM Sans")))
    return fig, sharpe_curve, minvar_curve, rp_curve, equal_curve


def plot_backtest_with_regimes(curve, regimes, title="Equity Curve with Market Regimes"):
    fig = go.Figure()

    aligned = regimes.reindex(curve.index, method="ffill").fillna("BULL")
    changes = (aligned != aligned.shift(1)).cumsum()
    for _, grp in aligned.groupby(changes):
        reg = grp.iloc[0]
        if reg not in ("BEAR", "HIGH_VOL"):
            continue
        x0 = grp.index[0]
        x1 = grp.index[-1]
        color = COLORS["bear"] if reg == "BEAR" else COLORS["highvol"]
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.6,
                      layer="below", line_width=0)

    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines",
                             name="Max Sharpe Portfolio",
                             line=dict(width=2.5, color=COLORS["sharpe"])))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=14, color="rgba(239, 68, 68, 0.5)", symbol="square"),
                             name="Bear (> 10% drawdown)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=14, color="rgba(245, 158, 11, 0.5)", symbol="square"),
                             name="High Volatility (top 15%)"))

    fig.update_layout(**PLOT_LAYOUT, title=title,
                      xaxis_title="Date", yaxis_title="Portfolio Value",
                      height=440,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="center", x=0.5,
                                  font=dict(size=11, family="DM Sans")))
    return fig


def plot_return_distribution(daily_returns, weights, title, color):
    port_returns = daily_returns.values @ weights
    mu = port_returns.mean()
    sigma = port_returns.std()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_returns * 100, nbinsx=80,
                               marker_color=color, opacity=0.7,
                               name="Observed", histnorm="probability density"))
    x_range = np.linspace(port_returns.min(), port_returns.max(), 200)
    normal_pdf = norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(x=x_range * 100, y=normal_pdf / 100,
                             mode="lines",
                             line=dict(color="rgba(232,232,232,0.6)", width=2, dash="dash"),
                             name="Normal Fit"))
    fig.update_layout(**PLOT_LAYOUT, title=title,
                      xaxis_title="Daily Return (%)", yaxis_title="Density", height=380,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_risk_contributions(weights, cov_matrix, tickers, title):
    rc = compute_risk_contributions(weights, cov_matrix)
    mask = rc > 0.5
    filtered_rc = rc[mask]
    filtered_tickers = [t for t, m in zip(tickers, mask) if m]
    fig = go.Figure(data=go.Bar(x=filtered_tickers, y=filtered_rc,
                                 marker_color=EMERALD_PALETTE[:len(filtered_tickers)],
                                 text=[f"{v:.1f}%" for v in filtered_rc],
                                 textposition="outside",
                                 textfont=dict(size=11, family="JetBrains Mono")))
    fig.update_layout(**PLOT_LAYOUT, title=title,
                      xaxis_title="Asset", yaxis_title="Risk Contribution (%)", height=380)
    return fig


def plot_rolling_volatility(daily_returns, tickers, window=30):
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        rolling_vol = daily_returns[ticker].rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines",
                                 name=ticker,
                                 line=dict(width=1.5, color=EMERALD_PALETTE[i % len(EMERALD_PALETTE)])))
    fig.update_layout(**PLOT_LAYOUT, title=f"Rolling {window}-Day Annualized Volatility (%)",
                      xaxis_title="Date", yaxis_title="Volatility (%)", height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_rolling_correlation(daily_returns, tickers, window=60):
    fig = go.Figure()
    color_idx = 0
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            rolling_corr = daily_returns[tickers[i]].rolling(window).corr(daily_returns[tickers[j]])
            fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode="lines",
                                     name=f"{tickers[i]} / {tickers[j]}",
                                     line=dict(width=1.5,
                                               color=EMERALD_PALETTE[color_idx % len(EMERALD_PALETTE)])))
            color_idx += 1
    fig.update_layout(**PLOT_LAYOUT, title=f"Rolling {window}-Day Pairwise Correlation",
                      xaxis_title="Date", yaxis_title="Correlation", height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_factor_betas(reg_results, strategy_names):
    factor_names = list(reg_results[strategy_names[0]]["betas"].keys())
    fig = go.Figure()
    strategy_colors = {
        "Max Sharpe": COLORS["sharpe"],
        "Min Variance": COLORS["min_var"],
        "Risk Parity": COLORS["risk_parity"],
    }
    for strat in strategy_names:
        betas = list(reg_results[strat]["betas"].values())
        tstats = list(reg_results[strat]["t_stats"].values())
        hover = [f"t-stat: {t:.2f}" for t in tstats]
        fig.add_trace(go.Bar(x=factor_names, y=betas, name=strat,
                             marker_color=strategy_colors.get(strat, "#888"),
                             text=[f"{b:.3f}" for b in betas],
                             textposition="outside",
                             textfont=dict(size=10, family="JetBrains Mono"),
                             customdata=hover,
                             hovertemplate="%{x}<br>β: %{y:.3f}<br>%{customdata}<extra>" + strat + "</extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(232,232,232,0.3)")
    fig.update_layout(**PLOT_LAYOUT, title="Factor Loadings by Strategy (Fama-French Regression)",
                      xaxis_title="Factor", yaxis_title="Beta (Factor Exposure)",
                      height=430, barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


def plot_regime_stats_bar(regime_stats_dict, strategy_names):
    regimes = ["BULL", "BEAR", "HIGH_VOL"]
    regime_display = {"BULL": "Bull", "BEAR": "Bear", "HIGH_VOL": "High Volatility"}
    fig = go.Figure()
    strategy_colors = {
        "Max Sharpe": COLORS["sharpe"],
        "Min Variance": COLORS["min_var"],
        "Risk Parity": COLORS["risk_parity"],
    }
    for strat in strategy_names:
        rets = [regime_stats_dict[strat][r]["ann_return"] * 100 if not np.isnan(regime_stats_dict[strat][r]["ann_return"]) else 0
                for r in regimes]
        sharpes = [regime_stats_dict[strat][r]["sharpe"] for r in regimes]
        hover = [f"Sharpe: {s:.2f}" if not np.isnan(s) else "insufficient data"
                 for s in sharpes]
        fig.add_trace(go.Bar(x=[regime_display[r] for r in regimes],
                             y=rets, name=strat,
                             marker_color=strategy_colors.get(strat, "#888"),
                             text=[f"{r:.1f}%" for r in rets],
                             textposition="outside",
                             textfont=dict(size=10, family="JetBrains Mono"),
                             customdata=hover,
                             hovertemplate="%{x}<br>Ann. Return: %{y:.1f}%<br>%{customdata}<extra>" + strat + "</extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(232,232,232,0.3)")
    fig.update_layout(**PLOT_LAYOUT, title="Annualized Return by Market Regime",
                      xaxis_title="Regime", yaxis_title="Annualized Return (%)",
                      height=430, barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="center", x=0.5, font=dict(family="DM Sans")))
    return fig


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

st.sidebar.markdown("## Configuration")

st.sidebar.markdown("### Asset Selection")
asset_type = st.sidebar.radio("Asset class", ["Stocks / ETFs", "Crypto", "Custom"], horizontal=True)

if asset_type == "Stocks / ETFs":
    selected_tickers = st.sidebar.multiselect(
        "Select tickers", options=POPULAR_STOCKS,
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"],
    )
elif asset_type == "Crypto":
    selected_tickers = st.sidebar.multiselect(
        "Select tickers", options=POPULAR_CRYPTO,
        default=["BTC-USD", "ETH-USD", "SOL-USD"],
    )
else:
    custom_input = st.sidebar.text_input(
        "Enter tickers (comma-separated)", value="AAPL, MSFT, GOOGL",
    )
    selected_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

st.sidebar.markdown("### Date Range")
col_start, col_end = st.sidebar.columns(2)
default_end = datetime.today()
default_start = default_end - timedelta(days=3 * 365)
start_date = col_start.date_input("Start", value=default_start)
end_date = col_end.date_input("End", value=default_end)

st.sidebar.markdown("### Parameters")
risk_free = st.sidebar.slider(
    "Risk-free rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25,
) / 100.0

n_simulations = st.sidebar.select_slider(
    "Monte Carlo simulations",
    options=[1000, 5000, 10000, 25000, 50000], value=10000,
)

st.sidebar.markdown("### Display Options")
show_risk_parity = st.sidebar.checkbox("Risk Parity portfolio", value=True)
show_cml = st.sidebar.checkbox("Capital Market Line", value=True)
show_var = st.sidebar.checkbox("VaR / CVaR metrics", value=True)
show_backtest = st.sidebar.checkbox("Portfolio backtest", value=True)
show_distributions = st.sidebar.checkbox("Return distributions", value=False)
show_risk_contrib = st.sidebar.checkbox("Risk contribution breakdown", value=False)
show_rolling = st.sidebar.checkbox("Rolling volatility / correlation", value=False)

st.sidebar.markdown("### Advanced Analysis")
show_factor = st.sidebar.checkbox("Factor regression (Fama-French)", value=True)
show_regimes = st.sidebar.checkbox("Market regime analysis", value=True)
if show_factor:
    ff_model = st.sidebar.radio("Factor model", ["FF5 + MOM", "FF5", "FF3 + MOM", "FF3"], index=0)
else:
    ff_model = "FF5"

if show_var:
    var_confidence = st.sidebar.slider(
        "VaR confidence level (%)", min_value=90.0, max_value=99.9, value=95.0, step=0.5,
    ) / 100.0
else:
    var_confidence = 0.95

if show_rolling:
    rolling_window = st.sidebar.select_slider(
        "Rolling window (days)",
        options=[15, 21, 30, 60, 90, 120], value=30,
    )
else:
    rolling_window = 30

run_button = st.sidebar.button("Optimize", use_container_width=True, type="primary")

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="qf-header">
    <div class="qf-header-content">
        <img src="https://marinxhemollari.com/frog-logo.svg"
             alt="QuantFolio" class="qf-logo"
             onerror="this.style.display='none'">
        <div class="qf-title-wrap">
            <div class="qf-title">
                <span class="qf-title-plain">Quant</span><span class="qf-title-accent">Folio</span>
                <span class="qf-version">v3.0</span>
            </div>
            <div class="qf-subtitle">
                Mean-variance optimization · Fama-French factor analysis · Market regime detection
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if not selected_tickers or len(selected_tickers) < 2:
    st.warning("Select at least **2 tickers** to build a portfolio.")
    st.stop()

if run_button or "results" in st.session_state:
    with st.spinner("Fetching price data..."):
        prices = fetch_price_data(selected_tickers, str(start_date), str(end_date))

    if prices.empty or len(prices) < 30:
        st.error("Not enough data. Try different tickers or a wider date range.")
        st.stop()

    valid_tickers = [t for t in selected_tickers if t in prices.columns]
    if len(valid_tickers) < 2:
        st.error("At least 2 tickers must have valid data for the selected date range.")
        st.stop()

    prices = prices[valid_tickers]

    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS
    corr_matrix = daily_returns.corr()

    with st.spinner("Running optimization..."):
        sharpe_weights = find_optimal_portfolio(
            mean_returns.values, cov_matrix.values, risk_free, objective="sharpe")
        sharpe_ret, sharpe_vol = calc_portfolio_performance(
            sharpe_weights, mean_returns.values, cov_matrix.values)
        sharpe_ratio = (sharpe_ret - risk_free) / sharpe_vol

        minvar_weights = find_optimal_portfolio(
            mean_returns.values, cov_matrix.values, risk_free, objective="min_variance")
        minvar_ret, minvar_vol = calc_portfolio_performance(
            minvar_weights, mean_returns.values, cov_matrix.values)
        minvar_sharpe = (minvar_ret - risk_free) / minvar_vol

        if show_risk_parity:
            rp_weights = find_risk_parity_portfolio(cov_matrix.values)
            rp_ret, rp_vol = calc_portfolio_performance(
                rp_weights, mean_returns.values, cov_matrix.values)
            rp_sharpe = (rp_ret - risk_free) / rp_vol
        else:
            rp_weights = np.ones(len(valid_tickers)) / len(valid_tickers)
            rp_ret, rp_vol, rp_sharpe = 0, 0, 0

    with st.spinner("Computing efficient frontier..."):
        frontier_vol, frontier_ret = compute_efficient_frontier(
            mean_returns.values, cov_matrix.values, risk_free, n_points=150)

    with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
        mc_ret, mc_vol, mc_sharpe, mc_weights = run_monte_carlo(
            mean_returns.values, cov_matrix.values, risk_free, n_simulations)

    if show_var:
        sharpe_var, sharpe_cvar = compute_var_cvar(daily_returns, sharpe_weights, var_confidence)
        minvar_var, minvar_cvar = compute_var_cvar(daily_returns, minvar_weights, var_confidence)
        if show_risk_parity:
            rp_var, rp_cvar = compute_var_cvar(daily_returns, rp_weights, var_confidence)

    asset_vols = np.sqrt(np.diag(cov_matrix.values))
    asset_rets = mean_returns.values
    st.session_state["results"] = True

    st.markdown("## Optimal Portfolios")

    if show_risk_parity:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
        col3 = None

    with col1:
        st.markdown('<div class="qf-strategy-label qf-strategy-sharpe">MAX SHARPE RATIO</div>',
                    unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Return", f"{sharpe_ret * 100:.1f}%")
        m2.metric("Volatility", f"{sharpe_vol * 100:.1f}%")
        m3, m4 = st.columns(2)
        m3.metric("Sharpe", f"{sharpe_ratio:.3f}")
        if show_var:
            m4.metric(f"CVaR {var_confidence*100:.0f}%", f"{sharpe_cvar*100:.2f}%")

    with col2:
        st.markdown('<div class="qf-strategy-label qf-strategy-minvar">MINIMUM VARIANCE</div>',
                    unsafe_allow_html=True)
        m5, m6 = st.columns(2)
        m5.metric("Return", f"{minvar_ret * 100:.1f}%")
        m6.metric("Volatility", f"{minvar_vol * 100:.1f}%")
        m7, m8 = st.columns(2)
        m7.metric("Sharpe", f"{minvar_sharpe:.3f}")
        if show_var:
            m8.metric(f"CVaR {var_confidence*100:.0f}%", f"{minvar_cvar*100:.2f}%")

    if show_risk_parity and col3 is not None:
        with col3:
            st.markdown('<div class="qf-strategy-label qf-strategy-rp">RISK PARITY</div>',
                        unsafe_allow_html=True)
            m9, m10 = st.columns(2)
            m9.metric("Return", f"{rp_ret * 100:.1f}%")
            m10.metric("Volatility", f"{rp_vol * 100:.1f}%")
            m11, m12 = st.columns(2)
            m11.metric("Sharpe", f"{rp_sharpe:.3f}")
            if show_var:
                m12.metric(f"CVaR {var_confidence*100:.0f}%", f"{rp_cvar*100:.2f}%")

    st.markdown("---")

    st.plotly_chart(
        plot_efficient_frontier(
            frontier_vol, frontier_ret, mc_vol, mc_ret, mc_sharpe,
            sharpe_vol, sharpe_ret, minvar_vol, minvar_ret,
            rp_vol, rp_ret,
            asset_vols, asset_rets, valid_tickers, risk_free,
            show_cml, show_risk_parity,
        ),
        use_container_width=True,
    )

    st.markdown("## Portfolio Allocations")
    if show_risk_parity:
        cp1, cp2, cp3 = st.columns(3)
    else:
        cp1, cp2 = st.columns(2)
        cp3 = None

    with cp1:
        st.plotly_chart(plot_weights_pie(sharpe_weights, valid_tickers, "Max Sharpe"),
                        use_container_width=True)
    with cp2:
        st.plotly_chart(plot_weights_pie(minvar_weights, valid_tickers, "Min Variance"),
                        use_container_width=True)
    if show_risk_parity and cp3 is not None:
        with cp3:
            st.plotly_chart(plot_weights_pie(rp_weights, valid_tickers, "Risk Parity"),
                            use_container_width=True)

    if show_risk_contrib:
        st.markdown("### Risk Contributions")
        if show_risk_parity:
            rc1, rc2, rc3 = st.columns(3)
        else:
            rc1, rc2 = st.columns(2)
            rc3 = None
        with rc1:
            st.plotly_chart(plot_risk_contributions(sharpe_weights, cov_matrix.values, valid_tickers, "Max Sharpe"),
                            use_container_width=True)
        with rc2:
            st.plotly_chart(plot_risk_contributions(minvar_weights, cov_matrix.values, valid_tickers, "Min Variance"),
                            use_container_width=True)
        if show_risk_parity and rc3 is not None:
            with rc3:
                st.plotly_chart(plot_risk_contributions(rp_weights, cov_matrix.values, valid_tickers, "Risk Parity"),
                                use_container_width=True)

    st.markdown("### Detailed Weights")
    weight_data = {
        "Ticker": valid_tickers,
        "Max Sharpe (%)": np.round(sharpe_weights * 100, 2),
        "Min Variance (%)": np.round(minvar_weights * 100, 2),
    }
    if show_risk_parity:
        weight_data["Risk Parity (%)"] = np.round(rp_weights * 100, 2)

    weight_df = pd.DataFrame(weight_data).set_index("Ticker")
    filter_mask = (weight_df["Max Sharpe (%)"] > 0.5) | (weight_df["Min Variance (%)"] > 0.5)
    if show_risk_parity:
        filter_mask = filter_mask | (weight_df["Risk Parity (%)"] > 0.5)
    st.dataframe(weight_df[filter_mask], use_container_width=True)

    csv_buffer = io.StringIO()
    weight_df.to_csv(csv_buffer)
    st.download_button("Download weights as CSV", data=csv_buffer.getvalue(),
                       file_name="quantfolio_weights.csv", mime="text/csv")

    st.markdown("---")

    if show_backtest:
        st.markdown("## Backtest")
        bt_fig, sharpe_curve, minvar_curve, rp_curve, equal_curve = plot_backtest(
            prices, sharpe_weights, minvar_weights, rp_weights, valid_tickers, show_risk_parity)
        st.plotly_chart(bt_fig, use_container_width=True)

        bt_data = {
            "Max Sharpe": calc_backtest_stats(sharpe_curve, risk_free),
            "Min Variance": calc_backtest_stats(minvar_curve, risk_free),
        }
        if show_risk_parity and rp_curve is not None:
            bt_data["Risk Parity"] = calc_backtest_stats(rp_curve, risk_free)
        bt_data["Equal Weight"] = calc_backtest_stats(equal_curve, risk_free)
        st.dataframe(pd.DataFrame(bt_data), use_container_width=True)
        st.markdown("---")
    else:
        sharpe_curve = backtest_portfolio(prices, sharpe_weights, valid_tickers)
        minvar_curve = backtest_portfolio(prices, minvar_weights, valid_tickers)
        rp_curve = backtest_portfolio(prices, rp_weights, valid_tickers) if show_risk_parity else None

    if show_factor:
        st.markdown("## Factor Regression — Fama-French Decomposition")
        st.markdown(
            "> Regresses portfolio excess returns on academic risk factors to isolate "
            "**alpha** (skill) from **beta** (exposure to known risk premia). A high positive alpha "
            "with statistical significance (|t| > 2) suggests the strategy delivers return beyond "
            "what's explained by conventional factor exposures. Low alpha with high R² means the "
            "strategy's returns are mostly explained by its factor bets — which may or may not be "
            "intentional."
        )

        with st.spinner("Loading Fama-French factors from Ken French data library..."):
            want_mom = "MOM" in ff_model
            base = "FF5" if "FF5" in ff_model else "FF3"
            ff_data = fetch_ff_factors(base)
            mom_data = fetch_momentum_factor() if want_mom else None

        if ff_data is None:
            st.warning("Could not fetch Fama-French factor data. The Ken French data server may be "
                       "unreachable. Skipping factor regression.")
        else:
            if want_mom and mom_data is not None:
                factor_df = ff_data.join(mom_data, how="inner")
                factor_cols = [c for c in factor_df.columns if c != "RF"]
            else:
                factor_df = ff_data
                factor_cols = [c for c in factor_df.columns if c != "RF"]

            port_returns = {
                "Max Sharpe": sharpe_curve.pct_change().dropna(),
                "Min Variance": minvar_curve.pct_change().dropna(),
            }
            if show_risk_parity and rp_curve is not None:
                port_returns["Risk Parity"] = rp_curve.pct_change().dropna()

            reg_results = {}
            for strat, ret in port_returns.items():
                res = run_factor_regression(ret, factor_df, factor_cols)
                if res is not None:
                    reg_results[strat] = res

            if not reg_results:
                st.warning("Not enough overlapping dates between portfolio returns and factor data to run regression.")
            else:
                strat_names = list(reg_results.keys())
                n_strat = len(strat_names)
                cols = st.columns(n_strat)
                for i, strat in enumerate(strat_names):
                    r = reg_results[strat]
                    with cols[i]:
                        label_class = {
                            "Max Sharpe": "qf-strategy-sharpe",
                            "Min Variance": "qf-strategy-minvar",
                            "Risk Parity": "qf-strategy-rp",
                        }.get(strat, "qf-strategy-sharpe")
                        st.markdown(f'<div class="qf-strategy-label {label_class}">{strat.upper()}</div>',
                                    unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric("Alpha (ann.)", f"{r['alpha_annual']*100:.2f}%",
                                  help=f"Daily α t-stat = {r['alpha_tstat']:.2f}")
                        m2.metric("R²", f"{r['r_squared']*100:.1f}%")

                st.plotly_chart(plot_factor_betas(reg_results, strat_names),
                                use_container_width=True)

                st.markdown("### Regression Coefficients")

                def _fmt_coef(v, t):
                    stars = ""
                    if abs(t) >= 2.58:
                        stars = "***"
                    elif abs(t) >= 1.96:
                        stars = "**"
                    elif abs(t) >= 1.64:
                        stars = "*"
                    return f"{v:.4f}{stars}"

                detail_rows = []
                for strat, r in reg_results.items():
                    row = {"Strategy": strat,
                           "α (daily)": _fmt_coef(r["alpha_daily"], r["alpha_tstat"]),
                           "α t-stat": f"{r['alpha_tstat']:.2f}"}
                    for f in factor_cols:
                        row[f"β_{f}"] = _fmt_coef(r["betas"][f], r["t_stats"][f])
                    row["R²"] = f"{r['r_squared']*100:.1f}%"
                    row["Adj. R²"] = f"{r['adj_r_squared']*100:.1f}%"
                    row["N"] = r["n_obs"]
                    detail_rows.append(row)

                detail_df = pd.DataFrame(detail_rows).set_index("Strategy")
                st.dataframe(detail_df, use_container_width=True)

                st.markdown(
                    "> **Significance codes:** *** = 99% (|t| ≥ 2.58), ** = 95% (|t| ≥ 1.96), "
                    "* = 90% (|t| ≥ 1.64). Positive α t-stat with absolute value ≥ 1.96 indicates "
                    "statistically significant outperformance relative to the factor model.\n"
                    ">\n"
                    "> **Factor interpretations:** *Mkt-RF* = market excess return (market beta); "
                    "*SMB* = small minus big (size tilt); *HML* = high minus low book-to-market (value tilt); "
                    "*RMW* = robust minus weak profitability (quality tilt); *CMA* = conservative minus "
                    "aggressive investment (quality tilt); *MOM* = momentum (trend-following tilt)."
                )
                st.markdown("---")

    if show_regimes:
        st.markdown("## Market Regime Analysis")
        st.markdown(
            "> Classifies each trading day into one of three regimes using the S&P 500 (SPY) as the market proxy. "
            "**Bear** regime triggers when SPY is in a > 10% drawdown from its prior peak. "
            "**High Volatility** regime triggers when SPY's 20-day realized volatility exceeds the 85th percentile. "
            "Everything else is **Bull**. Regime-conditional statistics reveal which strategies are truly "
            "diversified versus which only perform in calm uptrends."
        )

        with st.spinner("Detecting market regimes from SPY..."):
            spy_prices = fetch_market_proxy(str(start_date), str(end_date))

        if spy_prices.empty:
            st.warning("Could not fetch SPY data for regime detection. Skipping regime analysis.")
        else:
            regimes, mkt_dd, mkt_vol = detect_regimes(spy_prices)

            reg_counts = regimes.value_counts()
            total_days = len(regimes)

            reg_col1, reg_col2, reg_col3 = st.columns(3)
            reg_col1.metric("Bull Days",
                            f"{reg_counts.get('BULL', 0)}",
                            f"{reg_counts.get('BULL', 0)/total_days*100:.0f}% of period")
            reg_col2.metric("Bear Days",
                            f"{reg_counts.get('BEAR', 0)}",
                            f"{reg_counts.get('BEAR', 0)/total_days*100:.0f}% of period")
            reg_col3.metric("High-Vol Days",
                            f"{reg_counts.get('HIGH_VOL', 0)}",
                            f"{reg_counts.get('HIGH_VOL', 0)/total_days*100:.0f}% of period")

            st.plotly_chart(plot_backtest_with_regimes(sharpe_curve, regimes,
                                                       title="Max Sharpe Portfolio — Equity Curve with Regime Overlay"),
                            use_container_width=True)

            port_returns_for_regime = {
                "Max Sharpe": sharpe_curve.pct_change().dropna(),
                "Min Variance": minvar_curve.pct_change().dropna(),
            }
            if show_risk_parity and rp_curve is not None:
                port_returns_for_regime["Risk Parity"] = rp_curve.pct_change().dropna()

            regime_stats = {
                strat: regime_conditional_stats(ret, regimes, risk_free / TRADING_DAYS)
                for strat, ret in port_returns_for_regime.items()
            }

            st.plotly_chart(plot_regime_stats_bar(regime_stats, list(regime_stats.keys())),
                            use_container_width=True)

            st.markdown("### Regime-Conditional Performance")
            rows = []
            for strat, stats in regime_stats.items():
                for reg_name, reg_label in [("BULL", "Bull"), ("BEAR", "Bear"), ("HIGH_VOL", "High Vol")]:
                    s = stats[reg_name]
                    rows.append({
                        "Strategy": strat,
                        "Regime": reg_label,
                        "Days": s["days"],
                        "Ann. Return": f"{s['ann_return']*100:.1f}%" if not np.isnan(s['ann_return']) else "—",
                        "Ann. Vol": f"{s['ann_vol']*100:.1f}%" if not np.isnan(s['ann_vol']) else "—",
                        "Sharpe": f"{s['sharpe']:.2f}" if not np.isnan(s['sharpe']) else "—",
                    })
            regime_df = pd.DataFrame(rows)
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

            st.markdown(
                "> **Reading the table:** The strategy with the highest Bull-regime return is often not "
                "the most resilient in Bear or High-Vol regimes. A risk parity or min-variance portfolio "
                "typically sacrifices some Bull upside in exchange for much better downside performance. "
                "The true 'diversified' strategy is the one with the smallest gap between Bull and Bear Sharpe ratios."
            )
            st.markdown("---")

    if show_distributions:
        st.markdown("## Return Distribution")
        if show_risk_parity:
            dc1, dc2, dc3 = st.columns(3)
        else:
            dc1, dc2 = st.columns(2)
            dc3 = None
        with dc1:
            st.plotly_chart(plot_return_distribution(daily_returns, sharpe_weights,
                                                    "Max Sharpe Returns", COLORS["sharpe"]),
                            use_container_width=True)
        with dc2:
            st.plotly_chart(plot_return_distribution(daily_returns, minvar_weights,
                                                    "Min Variance Returns", COLORS["min_var"]),
                            use_container_width=True)
        if show_risk_parity and dc3 is not None:
            with dc3:
                st.plotly_chart(plot_return_distribution(daily_returns, rp_weights,
                                                        "Risk Parity Returns", COLORS["risk_parity"]),
                                use_container_width=True)
        st.markdown("---")

    st.markdown("## Analysis")
    tab_names = ["Cumulative Returns", "Drawdown", "Correlation"]
    if show_rolling:
        tab_names.extend(["Rolling Volatility", "Rolling Correlation"])
    tab_names.append("Asset Stats")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    with tabs[tab_idx]:
        st.plotly_chart(plot_cumulative_returns(prices, valid_tickers), use_container_width=True)
    tab_idx += 1

    with tabs[tab_idx]:
        st.plotly_chart(plot_drawdown(prices, valid_tickers), use_container_width=True)
    tab_idx += 1

    with tabs[tab_idx]:
        st.plotly_chart(plot_correlation_heatmap(corr_matrix, valid_tickers), use_container_width=True)
        st.markdown("> Assets with low or negative correlation reduce portfolio variance more effectively.")
    tab_idx += 1

    if show_rolling:
        with tabs[tab_idx]:
            st.plotly_chart(plot_rolling_volatility(daily_returns, valid_tickers, rolling_window),
                            use_container_width=True)
        tab_idx += 1
        with tabs[tab_idx]:
            st.plotly_chart(plot_rolling_correlation(daily_returns, valid_tickers, rolling_window),
                            use_container_width=True)
        tab_idx += 1

    with tabs[tab_idx]:
        stats_df = pd.DataFrame({
            "Ticker": valid_tickers,
            "Ann. Return (%)": np.round(asset_rets * 100, 2),
            "Ann. Volatility (%)": np.round(asset_vols * 100, 2),
            "Sharpe Ratio": np.round((asset_rets - risk_free) / asset_vols, 3),
            "Max Drawdown (%)": [round((prices[t] / prices[t].cummax() - 1).min() * 100, 2)
                                 for t in valid_tickers],
            "Daily Skewness": np.round(daily_returns[valid_tickers].skew().values, 3),
            "Daily Kurtosis": np.round(daily_returns[valid_tickers].kurtosis().values, 3),
        }).set_index("Ticker")
        st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")
    st.markdown("## Data Summary")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Trading Days", f"{len(prices):,}")
    info_col2.metric("Start", prices.index[0].strftime('%Y-%m-%d'))
    info_col3.metric("End", prices.index[-1].strftime('%Y-%m-%d'))

st.markdown("---")
st.markdown("""
<div class="qf-footer">
    Built by <a href="https://marinxhemollari.com" target="_blank">Marin Xhemollari</a> ·
    Markowitz MVO · Fama-French Factor Model · Regime Detection
    <div class="qf-footer-mono">quantfolio v3.0.1 · scipy.optimize.SLSQP · Ken French Data Library · plotly.js</div>
</div>
""", unsafe_allow_html=True)
