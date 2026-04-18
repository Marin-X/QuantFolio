"""
QuantFolio — Quantitative Portfolio Optimization Engine
Built by Marin Xhemollari | marinxhemollari.com

Implements:
- Mean-Variance Optimization (Markowitz, 1952)
- Efficient Frontier + Capital Market Line
- Maximum Sharpe Ratio / Global Minimum Variance / Risk Parity portfolios
- Monte Carlo simulation of random allocations
- Portfolio backtesting with equity curves
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Return distribution analysis with normal overlay
- Rolling volatility and correlation analysis
- Risk contribution decomposition
- CSV export of optimal weights
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

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="QuantFolio",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem !important;
        color: rgba(255, 255, 255, 0.5) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 8px 16px; }
    hr { border-color: rgba(255, 255, 255, 0.06) !important; }
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
        result = minimize(
            neg_sharpe_ratio, initial_weights,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP", bounds=bounds, constraints=constraints,
        )
    else:
        result = minimize(
            portfolio_volatility, initial_weights,
            args=(mean_returns, cov_matrix),
            method="SLSQP", bounds=bounds, constraints=constraints,
        )
    return result.x


def find_risk_parity_portfolio(cov_matrix):
    """
    Risk Parity: each asset contributes equally to total portfolio risk.
    Minimizes squared differences between each asset's risk contribution
    and the target (1/n of total risk).
    """
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

    result = minimize(
        risk_parity_objective, initial_weights,
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
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
        result = minimize(
            portfolio_volatility, initial_weights,
            args=(mean_returns, cov_matrix),
            method="SLSQP", bounds=bounds, constraints=constraints,
        )
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
# PLOTTING
# ──────────────────────────────────────────────────────────────

COLORS = {
    "frontier": "#6366f1",
    "cml": "#a78bfa",
    "sharpe": "#22c55e",
    "min_var": "#f59e0b",
    "risk_parity": "#ec4899",
    "assets": "#ef4444",
    "equal_weight": "#94a3b8",
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(255,255,255,0.06)",
    "text": "rgba(255,255,255,0.7)",
}

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
)


def plot_efficient_frontier(
        frontier_vol, frontier_ret,
        mc_vol, mc_ret, mc_sharpe,
        sharpe_vol, sharpe_ret,
        minvar_vol, minvar_ret,
        rp_vol, rp_ret,
        asset_vols, asset_rets, tickers,
        risk_free, show_cml, show_rp,
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=mc_vol * 100, y=mc_ret * 100, mode="markers",
        marker=dict(size=2.5, opacity=0.3, color=mc_sharpe,
                    colorscale="Viridis",
                    colorbar=dict(title="Sharpe", thickness=12, len=0.5)),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=frontier_vol * 100, y=frontier_ret * 100,
        mode="lines", line=dict(color=COLORS["frontier"], width=3),
        name="Efficient Frontier",
    ))

    if show_cml:
        cml_x_max = max(asset_vols.max(), sharpe_vol) * 1.3
        cml_x = np.linspace(0, cml_x_max, 100)
        cml_slope = (sharpe_ret - risk_free) / sharpe_vol
        cml_y = risk_free + cml_slope * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x * 100, y=cml_y * 100,
            mode="lines", line=dict(color=COLORS["cml"], width=2, dash="dash"),
            name="Capital Market Line",
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[risk_free * 100], mode="markers",
            marker=dict(size=10, color="white", symbol="x",
                        line=dict(width=2, color="white")),
            name=f"Risk-Free ({risk_free*100:.1f}%)",
        ))

    fig.add_trace(go.Scatter(
        x=[sharpe_vol * 100], y=[sharpe_ret * 100], mode="markers",
        marker=dict(size=16, color=COLORS["sharpe"], symbol="star",
                    line=dict(width=1, color="white")),
        name="Max Sharpe",
        hovertemplate=f"Max Sharpe<br>Vol: {sharpe_vol*100:.2f}%<br>Ret: {sharpe_ret*100:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[minvar_vol * 100], y=[minvar_ret * 100], mode="markers",
        marker=dict(size=16, color=COLORS["min_var"], symbol="diamond",
                    line=dict(width=1, color="white")),
        name="Min Variance",
        hovertemplate=f"Min Variance<br>Vol: {minvar_vol*100:.2f}%<br>Ret: {minvar_ret*100:.2f}%<extra></extra>",
    ))

    if show_rp:
        fig.add_trace(go.Scatter(
            x=[rp_vol * 100], y=[rp_ret * 100], mode="markers",
            marker=dict(size=16, color=COLORS["risk_parity"], symbol="hexagon",
                        line=dict(width=1, color="white")),
            name="Risk Parity",
            hovertemplate=f"Risk Parity<br>Vol: {rp_vol*100:.2f}%<br>Ret: {rp_ret*100:.2f}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=asset_vols * 100, y=asset_rets * 100,
        mode="markers+text",
        marker=dict(size=10, color=COLORS["assets"],
                    line=dict(width=1, color="white")),
        text=tickers, textposition="top center",
        textfont=dict(size=10, color="white"),
        name="Individual Assets",
        hovertemplate="%{text}<br>Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
    ))

    title = "Efficient Frontier with Capital Market Line" if show_cml else "Efficient Frontier"
    fig.update_layout(
        **PLOT_LAYOUT, title=title,
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5, font=dict(size=11)),
        height=550,
    )
    return fig


def plot_correlation_heatmap(corr_matrix, tickers):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=tickers, y=tickers,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(thickness=12, len=0.8),
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
        hole=0.45, textinfo="label+percent", textfont=dict(size=12),
        marker=dict(colors=px.colors.qualitative.Set2[:len(filtered_tickers)],
                    line=dict(color="rgba(0,0,0,0.3)", width=1)),
    ))
    fig.update_layout(**PLOT_LAYOUT, title=title, height=380, showlegend=False)
    return fig


def plot_cumulative_returns(prices, tickers):
    normalized = (prices / prices.iloc[0]) * 100
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[ticker], mode="lines",
            name=ticker, line=dict(width=2, color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        **PLOT_LAYOUT, title="Cumulative Returns (Normalized to 100)",
        xaxis_title="Date", yaxis_title="Value", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def plot_drawdown(prices, tickers):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, ticker in enumerate(tickers):
        series = prices[ticker]
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max * 100
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown, mode="lines", name=ticker,
            line=dict(width=1.5, color=colors[i % len(colors)]),
            fill="tozeroy", opacity=0.6,
        ))
    fig.update_layout(
        **PLOT_LAYOUT, title="Drawdown Analysis",
        xaxis_title="Date", yaxis_title="Drawdown (%)", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def plot_backtest(prices, sharpe_w, minvar_w, rp_w, tickers, show_rp):
    eq_w = np.ones(len(tickers)) / len(tickers)
    sharpe_curve = backtest_portfolio(prices, sharpe_w, tickers)
    minvar_curve = backtest_portfolio(prices, minvar_w, tickers)
    rp_curve = backtest_portfolio(prices, rp_w, tickers) if show_rp else None
    equal_curve = backtest_portfolio(prices, eq_w, tickers)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sharpe_curve.index, y=sharpe_curve.values, mode="lines",
        name="Max Sharpe", line=dict(width=2.5, color=COLORS["sharpe"]),
    ))
    fig.add_trace(go.Scatter(
        x=minvar_curve.index, y=minvar_curve.values, mode="lines",
        name="Min Variance", line=dict(width=2.5, color=COLORS["min_var"]),
    ))
    if show_rp and rp_curve is not None:
        fig.add_trace(go.Scatter(
            x=rp_curve.index, y=rp_curve.values, mode="lines",
            name="Risk Parity", line=dict(width=2.5, color=COLORS["risk_parity"]),
        ))
    fig.add_trace(go.Scatter(
        x=equal_curve.index, y=equal_curve.values, mode="lines",
        name="Equal Weight", line=dict(width=2, color=COLORS["equal_weight"], dash="dot"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT, title="Portfolio Backtest (Normalized to 100)",
        xaxis_title="Date", yaxis_title="Portfolio Value", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5, font=dict(size=11)),
    )
    return fig, sharpe_curve, minvar_curve, rp_curve, equal_curve


def plot_return_distribution(daily_returns, weights, title, color):
    port_returns = daily_returns.values @ weights
    mu = port_returns.mean()
    sigma = port_returns.std()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_returns * 100, nbinsx=80,
        marker_color=color, opacity=0.7,
        name="Observed", histnorm="probability density",
    ))
    x_range = np.linspace(port_returns.min(), port_returns.max(), 200)
    normal_pdf = norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range * 100, y=normal_pdf / 100,
        mode="lines", line=dict(color="white", width=2, dash="dash"),
        name="Normal Fit",
    ))
    fig.update_layout(
        **PLOT_LAYOUT, title=title,
        xaxis_title="Daily Return (%)", yaxis_title="Density", height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def plot_risk_contributions(weights, cov_matrix, tickers, title):
    rc = compute_risk_contributions(weights, cov_matrix)
    mask = rc > 0.5
    filtered_rc = rc[mask]
    filtered_tickers = [t for t, m in zip(tickers, mask) if m]
    fig = go.Figure(data=go.Bar(
        x=filtered_tickers, y=filtered_rc,
        marker_color=px.colors.qualitative.Set2[:len(filtered_tickers)],
        text=[f"{v:.1f}%" for v in filtered_rc],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        **PLOT_LAYOUT, title=title,
        xaxis_title="Asset", yaxis_title="Risk Contribution (%)", height=380,
    )
    return fig


def plot_rolling_volatility(daily_returns, tickers, window=30):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, ticker in enumerate(tickers):
        rolling_vol = daily_returns[ticker].rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values, mode="lines",
            name=ticker, line=dict(width=1.5, color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        **PLOT_LAYOUT, title=f"Rolling {window}-Day Annualized Volatility (%)",
        xaxis_title="Date", yaxis_title="Volatility (%)", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def plot_rolling_correlation(daily_returns, tickers, window=60):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            rolling_corr = daily_returns[tickers[i]].rolling(window).corr(daily_returns[tickers[j]])
            fig.add_trace(go.Scatter(
                x=rolling_corr.index, y=rolling_corr.values, mode="lines",
                name=f"{tickers[i]} / {tickers[j]}",
                line=dict(width=1.5, color=colors[color_idx % len(colors)]),
            ))
            color_idx += 1
    fig.update_layout(
        **PLOT_LAYOUT, title=f"Rolling {window}-Day Pairwise Correlation",
        xaxis_title="Date", yaxis_title="Correlation", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

st.sidebar.markdown("## Configuration")

# Asset selection
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

# Date range
st.sidebar.markdown("### Date Range")
col_start, col_end = st.sidebar.columns(2)
default_end = datetime.today()
default_start = default_end - timedelta(days=3 * 365)
start_date = col_start.date_input("Start", value=default_start)
end_date = col_end.date_input("End", value=default_end)

# Parameters
st.sidebar.markdown("### Parameters")
risk_free = st.sidebar.slider(
    "Risk-free rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25,
) / 100.0

n_simulations = st.sidebar.select_slider(
    "Monte Carlo simulations",
    options=[1000, 5000, 10000, 25000, 50000], value=10000,
)

# Display options
st.sidebar.markdown("### Display Options")

show_risk_parity = st.sidebar.checkbox("Risk Parity portfolio", value=True)
show_cml = st.sidebar.checkbox("Capital Market Line", value=True)
show_var = st.sidebar.checkbox("VaR / CVaR metrics", value=True)
show_backtest = st.sidebar.checkbox("Portfolio backtest", value=True)
show_distributions = st.sidebar.checkbox("Return distributions", value=False)
show_risk_contrib = st.sidebar.checkbox("Risk contribution breakdown", value=False)
show_rolling = st.sidebar.checkbox("Rolling volatility / correlation", value=False)

# Conditional parameters
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

st.markdown("# QuantFolio")
st.markdown(
    "Mean-variance optimization, efficient frontier construction, "
    "and Monte Carlo simulation for multi-asset portfolios."
)
st.markdown("---")


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

    daily_returns = np.log(prices / prices.shift(1)).dropna()
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

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════

    st.markdown("## Optimal Portfolios")

    if show_risk_parity:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
        col3 = None

    with col1:
        st.markdown("#### Max Sharpe Ratio")
        m1, m2 = st.columns(2)
        m1.metric("Return", f"{sharpe_ret * 100:.1f}%")
        m2.metric("Volatility", f"{sharpe_vol * 100:.1f}%")
        m3, m4 = st.columns(2)
        m3.metric("Sharpe", f"{sharpe_ratio:.3f}")
        if show_var:
            m4.metric(f"CVaR {var_confidence*100:.0f}%", f"{sharpe_cvar*100:.2f}%")

    with col2:
        st.markdown("#### Minimum Variance")
        m5, m6 = st.columns(2)
        m5.metric("Return", f"{minvar_ret * 100:.1f}%")
        m6.metric("Volatility", f"{minvar_vol * 100:.1f}%")
        m7, m8 = st.columns(2)
        m7.metric("Sharpe", f"{minvar_sharpe:.3f}")
        if show_var:
            m8.metric(f"CVaR {var_confidence*100:.0f}%", f"{minvar_cvar*100:.2f}%")

    if show_risk_parity and col3 is not None:
        with col3:
            st.markdown("#### Risk Parity")
            m9, m10 = st.columns(2)
            m9.metric("Return", f"{rp_ret * 100:.1f}%")
            m10.metric("Volatility", f"{rp_vol * 100:.1f}%")
            m11, m12 = st.columns(2)
            m11.metric("Sharpe", f"{rp_sharpe:.3f}")
            if show_var:
                m12.metric(f"CVaR {var_confidence*100:.0f}%", f"{rp_cvar*100:.2f}%")

    st.markdown("---")

    # Efficient Frontier
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

    # Allocations
    st.markdown("## Portfolio Allocations")
    if show_risk_parity:
        cp1, cp2, cp3 = st.columns(3)
    else:
        cp1, cp2 = st.columns(2)
        cp3 = None

    with cp1:
        st.plotly_chart(
            plot_weights_pie(sharpe_weights, valid_tickers, "Max Sharpe"),
            use_container_width=True)
    with cp2:
        st.plotly_chart(
            plot_weights_pie(minvar_weights, valid_tickers, "Min Variance"),
            use_container_width=True)
    if show_risk_parity and cp3 is not None:
        with cp3:
            st.plotly_chart(
                plot_weights_pie(rp_weights, valid_tickers, "Risk Parity"),
                use_container_width=True)

    # Risk contributions
    if show_risk_contrib:
        st.markdown("### Risk Contributions")
        if show_risk_parity:
            rc1, rc2, rc3 = st.columns(3)
        else:
            rc1, rc2 = st.columns(2)
            rc3 = None

        with rc1:
            st.plotly_chart(
                plot_risk_contributions(sharpe_weights, cov_matrix.values, valid_tickers, "Max Sharpe"),
                use_container_width=True)
        with rc2:
            st.plotly_chart(
                plot_risk_contributions(minvar_weights, cov_matrix.values, valid_tickers, "Min Variance"),
                use_container_width=True)
        if show_risk_parity and rc3 is not None:
            with rc3:
                st.plotly_chart(
                    plot_risk_contributions(rp_weights, cov_matrix.values, valid_tickers, "Risk Parity"),
                    use_container_width=True)

    # Weight table + export
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
    st.download_button(
        "Download weights as CSV",
        data=csv_buffer.getvalue(),
        file_name="quantfolio_weights.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Backtest
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

    # Return distributions
    if show_distributions:
        st.markdown("## Return Distribution")
        if show_risk_parity:
            dc1, dc2, dc3 = st.columns(3)
        else:
            dc1, dc2 = st.columns(2)
            dc3 = None

        with dc1:
            st.plotly_chart(
                plot_return_distribution(daily_returns, sharpe_weights,
                                         "Max Sharpe Returns", COLORS["sharpe"]),
                use_container_width=True)
        with dc2:
            st.plotly_chart(
                plot_return_distribution(daily_returns, minvar_weights,
                                         "Min Variance Returns", COLORS["min_var"]),
                use_container_width=True)
        if show_risk_parity and dc3 is not None:
            with dc3:
                st.plotly_chart(
                    plot_return_distribution(daily_returns, rp_weights,
                                             "Risk Parity Returns", COLORS["risk_parity"]),
                    use_container_width=True)
        st.markdown("---")

    # Analysis tabs
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
        st.markdown(
            "> Assets with low or negative correlation "
            "reduce portfolio variance more effectively. "
            "The optimizer exploits this to construct efficient portfolios."
        )
    tab_idx += 1

    if show_rolling:
        with tabs[tab_idx]:
            st.plotly_chart(
                plot_rolling_volatility(daily_returns, valid_tickers, rolling_window),
                use_container_width=True)
        tab_idx += 1

        with tabs[tab_idx]:
            st.plotly_chart(
                plot_rolling_correlation(daily_returns, valid_tickers, rolling_window),
                use_container_width=True)
        tab_idx += 1

    with tabs[tab_idx]:
        stats_df = pd.DataFrame({
            "Ticker": valid_tickers,
            "Ann. Return (%)": np.round(asset_rets * 100, 2),
            "Ann. Volatility (%)": np.round(asset_vols * 100, 2),
            "Sharpe Ratio": np.round((asset_rets - risk_free) / asset_vols, 3),
            "Max Drawdown (%)": [
                round((prices[t] / prices[t].cummax() - 1).min() * 100, 2)
                for t in valid_tickers
            ],
            "Daily Skewness": np.round(daily_returns[valid_tickers].skew().values, 3),
            "Daily Kurtosis": np.round(daily_returns[valid_tickers].kurtosis().values, 3),
        }).set_index("Ticker")
        st.dataframe(stats_df, use_container_width=True)
        st.markdown(
            "> Skewness < 0 indicates left-tail risk. "
            "Excess kurtosis > 0 indicates fat tails — extreme moves occur more "
            "frequently than a normal distribution predicts."
        )

    # Data summary
    st.markdown("---")
    st.markdown("## Data Summary")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Trading Days", f"{len(prices):,}")
    info_col2.metric("Start", prices.index[0].strftime('%Y-%m-%d'))
    info_col3.metric("End", prices.index[-1].strftime('%Y-%m-%d'))


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.85rem;'>"
    "Built by <a href='https://marinxhemollari.com' target='_blank' "
    "style='color: rgba(255,255,255,0.5);'>Marin Xhemollari</a> · "
    "Markowitz Mean-Variance Optimization · "
    "Market data via Yahoo Finance"
    "</div>",
    unsafe_allow_html=True,
)