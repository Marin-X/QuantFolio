# Portfolio Optimizer

Interactive portfolio optimization dashboard built with Python and Streamlit. Implements Modern Portfolio Theory (Markowitz, 1952) to find optimal asset allocations across stocks, ETFs, and cryptocurrencies.

**[Live Demo →](https://portfolio-optimizer-mx.streamlit.app)**

---

## Features

- **Efficient Frontier** — traces the set of portfolios offering the highest return for each level of risk via constrained optimization (SLSQP)
- **Optimal Portfolios** — finds the maximum Sharpe ratio (tangency) portfolio and global minimum variance portfolio under long-only constraints
- **Monte Carlo Simulation** — generates up to 50,000 random portfolio allocations to visualize the feasible region
- **Correlation Analysis** — heatmap of return correlations to assess diversification potential
- **Risk Metrics** — annualized return, volatility, Sharpe ratio, max drawdown, skewness, and kurtosis per asset
- **Drawdown Analysis** — peak-to-trough decline visualization for each asset
- **Real Data** — pulls historical adjusted close prices from Yahoo Finance via `yfinance`
- **Flexible Inputs** — pre-built lists for popular stocks/ETFs and crypto, plus custom ticker entry

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Visualization | Plotly |
| Optimization | SciPy (SLSQP) |
| Data | yfinance, Pandas, NumPy |
| Deployment | Streamlit Cloud |

## How It Works

### Mean-Variance Optimization

Given *n* assets with expected return vector **μ** and covariance matrix **Σ**, the portfolio return and variance are:

```
R_p = w^T · μ
σ²_p = w^T · Σ · w
```

The optimizer solves for the weight vector **w** that either maximizes the Sharpe ratio `(R_p - R_f) / σ_p` or minimizes `σ_p`, subject to:
- Weights sum to 1 (fully invested)
- Each weight ∈ [0, 1] (long-only, no short selling)

### Efficient Frontier

For each target return level, the optimizer finds the minimum-variance portfolio. The resulting curve of (volatility, return) pairs is the efficient frontier — the upper boundary of the feasible set.

### Monte Carlo Simulation

Random weight vectors are drawn from a Dirichlet distribution (uniform over the simplex). Each is evaluated for return, volatility, and Sharpe ratio, producing the scatter cloud on the frontier plot.

## Run Locally

```bash
git clone https://github.com/Marin-X/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
portfolio-optimizer/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme and server config
└── README.md
```

## Author

**Marin Xhemollari**
- Portfolio: [marinxhemollari.com](https://marinxhemollari.com)
- GitHub: [github.com/Marin-X](https://github.com/Marin-X)
- LinkedIn: [linkedin.com/in/marinxhemollari](https://linkedin.com/in/marinxhemollari)
