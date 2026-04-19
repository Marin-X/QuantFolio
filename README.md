# QuantFolio

Quantitative portfolio optimization and factor analysis engine built with Python and Streamlit. Implements mean-variance optimization (Markowitz, 1952), Fama-French factor regression, and market regime detection to decompose portfolio returns into risk premia and analyze performance across bull, bear, and high-volatility regimes.

**[Live Demo →](https://quantfolio-marinx.streamlit.app)**

---

## Screenshots

![Dashboard](dashboard.png)
![Efficient Frontier](frontier.png)
![Factor Analysis](factor_analysis.png)
![Regime Analysis](regime_analysis.png)
![Backtest](backtest.png)

---

## Features

### Core Portfolio Optimization
- **Efficient Frontier + Capital Market Line** — traces optimal risk-return tradeoff via constrained optimization (SLSQP)
- **Three Optimization Strategies** — Maximum Sharpe Ratio, Global Minimum Variance, Risk Parity
- **Monte Carlo Simulation** — up to 50,000 random portfolio allocations to visualize the feasible region
- **Portfolio Backtesting** — equity curves comparing all strategies vs equal-weight benchmark
- **VaR / CVaR** — Value at Risk and Conditional VaR at configurable confidence levels
- **Rolling Analysis** — rolling volatility and pairwise correlation over configurable windows

### Fama-French Factor Regression (v3.0)
- Regresses portfolio excess returns on academic risk factors from Ken French's Data Library
- Four model choices: FF3, FF3+MOM, FF5, FF5+MOM
- Reports **alpha** (annualized), **factor betas**, **t-statistics**, **R²**, and **adjusted R²** per strategy
- Significance-coded coefficients (\*\*\* 99%, \*\* 95%, \* 90% confidence)
- Side-by-side factor-loading comparison across all three strategies

### Market Regime Detection (v3.0)
- Classifies each trading day into Bull / Bear / High-Volatility regimes using SPY as the market proxy
- Bear trigger: > 10% drawdown from rolling peak
- High-vol trigger: 20-day realized volatility in the top 15% of the sample
- Regime-shaded equity curves and regime-conditional Sharpe ratios
- Identifies which strategies are genuinely diversified vs. which only perform in calm uptrends

### Data & Export
- Real historical prices from Yahoo Finance via `yfinance`
- Daily factor data from Ken French's Data Library (Tuck School of Business, Dartmouth)
- CSV export of optimized weights

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Visualization | Plotly |
| Optimization | SciPy (SLSQP) |
| Numerical | NumPy, Pandas |
| Market Data | yfinance |
| Factor Data | Ken French Data Library (CSV/ZIP) |
| Deployment | Streamlit Cloud |

---

## How It Works

### Return Definitions

Daily simple returns are used throughout for mathematical consistency with portfolio aggregation:

```
R_i,t = (P_i,t / P_i,t−1) − 1
```

This ensures the portfolio return is exactly the weighted sum of asset returns, `R_p = w · R` — a property that does not hold for log returns. Annualized statistics assume 252 trading days per year.

### Mean-Variance Optimization

Given *n* assets with expected return vector **μ** and covariance matrix **Σ**, the portfolio return and variance are:

```
R_p = wᵀ · μ
σ²_p = wᵀ · Σ · w
```

Subject to:
- Sum of weights = 1 (fully invested)
- 0 ≤ w_i ≤ 1 (long-only, no leverage)

Solved numerically with Sequential Least Squares Programming (`scipy.optimize.minimize`, `method="SLSQP"`).

### Three Portfolio Strategies

**Maximum Sharpe Ratio** — maximizes the risk-adjusted excess return:

```
maximize  (wᵀμ − r_f) / √(wᵀΣw)
```

**Global Minimum Variance** — minimizes total portfolio variance:

```
minimize  wᵀΣw
```

**Risk Parity** — equalizes each asset's contribution to total portfolio risk:

```
minimize  Σᵢ (RC_i − σ_p/n)²

where  RC_i = w_i · (Σw)_i / √(wᵀΣw)
```

### Efficient Frontier

Traced by minimizing variance subject to a target return constraint, producing the set of Pareto-optimal portfolios.

### Capital Market Line

The line from the risk-free rate through the tangency (Max Sharpe) portfolio:

```
E[R_p] = r_f + [(E[R_T] − r_f) / σ_T] · σ_p
```

### Value at Risk (VaR) & Conditional VaR (CVaR)

At confidence level α:

```
VaR_α = percentile of portfolio returns at (1 − α) × 100%
CVaR_α = E[R_p | R_p ≤ VaR_α]
```

CVaR (Expected Shortfall) is more sensitive to tail risk than VaR alone.

### Fama-French Factor Regression

Portfolio excess returns are regressed on factor returns via OLS:

```
R_p,t − R_f,t = α + β₁·MKT_t + β₂·SMB_t + β₃·HML_t + … + ε_t
```

Where factors are drawn from Ken French's Data Library:
- **MKT-RF** — market excess return (market beta)
- **SMB** — small minus big (size tilt)
- **HML** — high minus low book-to-market (value tilt)
- **RMW** — robust minus weak profitability (quality tilt, FF5 only)
- **CMA** — conservative minus aggressive investment (quality tilt, FF5 only)
- **MOM** — winners minus losers (momentum tilt, optional add-on)

The intercept **α** is the portion of excess return unexplained by known factor exposures — the "skill" or "mispricing" component. A positive, statistically significant α (|t| ≥ 1.96) indicates outperformance beyond the factor model's predictions. T-statistics are computed from OLS standard errors:

```
β = (XᵀX)⁻¹ Xᵀ y
SE(β) = √(σ² · diag((XᵀX)⁻¹))
t = β / SE(β)
```

### Market Regime Detection

Each trading day is classified using SPY as the market proxy:

```
BEAR      if drawdown from rolling peak > 10%
HIGH_VOL  else if 20-day realized vol > 85th percentile of sample vols
BULL      otherwise
```

Realized volatility is annualized: `σ_realized = std(r_20) · √252`.

For each strategy, regime-conditional statistics are computed:

```
Ann. Return_regime = mean(r_daily | regime) · 252
Ann. Vol_regime    = std(r_daily | regime) · √252
Sharpe_regime      = (Ann. Return_regime − r_f) / Ann. Vol_regime
```

### Backtest Statistics

| Metric | Formula |
|--------|---------|
| Annualized Return | (final / initial)^(252/N) − 1 |
| Annualized Volatility | σ_daily · √252 |
| Sharpe Ratio | (R_ann − r_f) / σ_ann |
| Sortino Ratio | (R_ann − r_f) / σ_downside,ann |
| Max Drawdown | min((P_t − max(P_{≤t})) / max(P_{≤t})) |
| Calmar Ratio | R_ann / \|MDD\| |

---

## Assumptions & Limitations

- Long-only portfolios — no short selling or leverage
- Historical returns and covariances used as forward-looking estimates (standard assumption; subject to estimation error)
- No transaction costs or taxes modeled in backtest
- Mean-variance framework assumes either normally distributed returns or quadratic utility; real returns exhibit skewness and fat tails (the Asset Stats tab surfaces this)
- Weights are optimized once in-sample — no walk-forward or rolling re-optimization
- Factor regression uses OLS with homoskedastic standard errors (no Newey-West adjustment for autocorrelation)
- Regime classification uses SPY only; not suitable for non-US or non-equity benchmarks without substitution
- Mispricing of alpha: a statistically significant alpha over a backtest window does not imply persistence out-of-sample

---

## Running Locally

```bash
git clone https://github.com/Marin-X/QuantFolio.git
cd QuantFolio
pip install -r requirements.txt
streamlit run app.py
```

The Fama-French factor data is fetched on-demand from Ken French's Data Library and cached for 24 hours.

---

## References

- Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1966). "Mutual Fund Performance." *The Journal of Business*, 39(1), 119–138.
- Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3–56.
- Fama, E. F. and French, K. R. (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics*, 116(1), 1–22.
- Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance." *The Journal of Finance*, 52(1), 57–82.
- Maillard, S., Roncalli, T., and Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management*, 36(4), 60–70.
- Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2, 21–42.

Factor data: [Ken French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), Tuck School of Business, Dartmouth College.

---

Built by [Marin Xhemollari](https://marinxhemollari.com) · [Portfolio](https://marinxhemollari.com) · [LinkedIn](https://linkedin.com/in/marinxhemollari)
