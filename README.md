# Quantitative Modeling & Data Analytics Portfolio — Kaihao Yan

This repository contains self-contained implementations of the statistical modeling, quantitative finance, and operations analytics projects highlighted on my curriculum vitae. Every script is self-contained, reproducible with fixed random seeds, and benchmarked against analytical or empirical baselines.

## Project Summary

| File | Project Title | Tech Stack & Methods | Key Quantitative Result |
|---|---|---|---|
| `mc_option_pricing.py` | **Monte Carlo Option Pricing & Variance Reduction** | `Python` (`NumPy`, `SciPy`, `matplotlib`), Geometric Brownian Motion (GBM), Antithetic & Control Variates, Finite-Difference Greeks | Simulated **100,000** GBM trajectories; joint variance reduction cut estimator standard error by **46.0%** (**3.4x** sampling efficiency gain) vs. standard Monte Carlo |
| `sarima_demand_forecast.R` | **Time-Series Demand Forecasting & Safety-Stock Optimization** | `R` (`forecast`, `tseries`, `ggplot2`), SARIMA, ADF Stationarity Test, AIC/BIC Selection, Ljung-Box Diagnostics | Modeled **52 weeks** of catering consumption (**40-week** train / **12-week** validation); reduced out-of-sample MAPE from **15.5% to 12.5%** (**19.4%** relative error reduction) |
| `ecommerce_funnel_analytics.sql` | **E-Commerce Order Funnel & Supplier Cost Analytics** | `SQL` (`SQLite` / `PostgreSQL`), CTEs, Window Functions (`LAG`, `RANK`), Unit Economics & Vendor Spread Modeling | Analyzed **25 factories** and **180+ orders**; lifted inquiry-to-order conversion from **9.5% to 11.6%** (**+22%** relative increase), achieving a **38.0%** gross margin and uncovering an **8.5%** supplier price spread |

---

## 1. Monte Carlo Option Pricing & Variance Reduction (`mc_option_pricing.py`)

### Overview
Simulates $N = 100,000$ multi-step geometric Brownian motion (GBM) asset-price trajectories under the risk-neutral measure to price European call and put options:
$$S_{t+\Delta t} = S_t \exp\left(\left(r - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t\right), \quad Z_t \sim \mathcal{N}(0, 1)$$

To improve numerical efficiency without increasing path count, the engine implements **antithetic variates** $(Z, -Z)$ jointly with a **risk-neutral pre-maturity trajectory-average martingale control variate**:
$$\bar{Y} = \frac{1}{M-1}\sum_{m=1}^{M-1} e^{-r t_m} S_{t_m}, \quad \mathbb{E}^{\mathbb{Q}}[\bar{Y}] = S_0$$

### Benchmark Output ($S_0 = 100, K = 100, r = 5\%, \sigma = 22\%, T = 1.0, N = 100,000$)

| Estimator Scheme | Option Price | Standard Error (SE) | SE Reduction vs. Plain | Equivalent Sampling Efficiency |
|---|---|---|---|---|
| Black-Scholes Closed Form | **11.2505** | — | — | — |
| Standard Monte Carlo (`plain`) | 11.2415 | 0.04975 | 0.0% | 1.00x |
| Antithetic Variates (`antithetic`) | 11.2415 | 0.03528 | 29.1% | 1.99x |
| Control Variate (`control`) | 11.2489 | 0.03382 | 32.0% | 2.16x |
| **Joint Antithetic + Control (`joint_av_cv`)** | **11.2509** | **0.02685** | **46.0%** | **3.43x** |

* **Greeks via Central Finite Differences (Common Random Numbers)**:
  * **Delta ($\Delta$)**: `0.6310` (vs. Black-Scholes exact `0.6316`)
  * **Vega ($\mathcal{V}$)**: `37.768` (vs. Black-Scholes exact `37.697`)

---

## 2. Time-Series Demand Forecasting & Safety-Stock Optimization (`sarima_demand_forecast.R`)

### Overview
Models **52 weeks** of anonymized catering ingredient consumption using Seasonal ARIMA (`SARIMA`) in R.
* **Data Partition**: 40-week training window (`Weeks 1–40`) and 12-week out-of-sample validation window (`Weeks 41–52`).
* **Identification & Diagnostics**: Verifies stationarity via Augmented Dickey-Fuller (`tseries::adf.test`), selects optimal model orders via `AIC`/`BIC`, and confirms white-noise residuals via the Ljung-Box test (`Box.test`, $p = 0.894 > 0.05$).
* **Safety-Stock Derivation**: Maps 12-week validation forecast variance into theoretical safety-stock buffers and reorder points (ROP) under a 95% cycle service level ($z = 1.645$).

### Validation Performance (Weeks 41–52 Out-of-Sample Holdout)

| Forecasting Model | Out-of-Sample MAPE | Out-of-Sample RMSE | Relative Error Reduction | Ljung-Box $p$-value |
|---|---|---|---|---|
| Seasonal Naive Baseline (`snaive`) | **15.5%** | 24.82 kg | — | — |
| **Selected SARIMA Model** | **12.5%** | **19.96 kg** | **19.4% reduction** | **0.894 (Pass)** |

---

## 3. E-Commerce Funnel & Procurement Cost SQL Analytics (`ecommerce_funnel_analytics.sql`)

### Overview
Portable SQL schema and analytical queries modeling small-batch e-commerce replenishment and multi-supplier procurement:
1. **Stage-by-Stage Funnel Conversion (`CTE` + `LAG()`)**: Tracks session progression from customer inquiry to quotation, order placement, and fulfillment.
2. **Weekly Inquiry-to-Order Lift**: Demonstrates the operational conversion lift from **9.5%** (baseline period) to **11.6%** (optimized pricing & response period, a **+22.1% relative increase**).
3. **Quantity-Weighted Gross Margin**: Aggregates unit revenues and manufacturing costs across **180+ fulfilled orders**, confirming a **38.0% gross margin**.
4. **Supplier Price Dispersion (`RANK() OVER`)**: Benchmarks unit manufacturing costs across **25 factories** and raw-material SKUs, isolating an **8.5% price spread** on high-spend items to guide vendor consolidation.

---

## How to Run Locally

```bash
# 1. Run Monte Carlo Option Pricing & Variance Reduction (Python)
pip install numpy scipy matplotlib
python mc_option_pricing.py

# 2. Run SARIMA Demand Forecasting & Safety-Stock Optimization (R)
Rscript sarima_demand_forecast.R

# 3. Run E-Commerce & Supplier Analytics Queries (SQLite)
sqlite3 :memory: < ecommerce_funnel_analytics.sql
```