# Quantitative Modeling & Data Analytics Portfolio — Kaihao Yan

This repository demonstrates quantitative finance, time-series modeling, and SQL analytics. The scripts are illustrative implementations, not independent verification of claims about real-world business outcomes.

## Data provenance

The **52-week catering consumption series** in `sarima_demand_forecast.R` and the **SQL example tables** in `ecommerce_funnel_analytics.sql` contain **synthetic / illustrative data inspired by personal catering and e-commerce experience**. They are not actual anonymized company records or evidence of real operational outcomes. The figures calculated from them describe these examples only. The assumed four-week cycle is an illustrative business/modeling assumption, not a verified property of real demand.

| File | Project | Methods |
|---|---|---|
| `mc_option_pricing.py` | Monte Carlo Option Pricing & Variance Reduction | GBM, independent plain MC, antithetic and control variates, finite-difference Greeks |
| `sarima_demand_forecast.R` | Time-Series Demand Forecasting & Safety-Stock Estimation | SARIMA candidate comparison, ADF and Ljung–Box tests, holdout evaluation |
| `ecommerce_funnel_analytics.sql` | E-Commerce Funnel & Supplier Cost Analytics | SQLite, CTEs, window functions, aggregate margin and price spread |

## 1. Monte Carlo option pricing

Simulates geometric Brownian motion under the risk-neutral measure, using a pre-maturity trajectory-average discounted asset price as a control variate:

$$\bar Y = \frac{1}{M-1}\sum_{m=1}^{M-1}e^{-rt_m}S_{t_m},\quad E^Q[\bar Y]=S_0.$$

Plain MC uses independently generated paths, distinct from the antithetic stream. Control-only uses the independent plain paths; the joint estimator uses pair averages. Each method uses the same number of paths, and pair-based standard errors use the number of independent pairs. Control coefficients are estimated on the same sample, so the usual standard errors are approximate and finite-sample bias is possible. Efficiency is the squared ratio of standard errors, not a measured runtime speedup.

### Call benchmark

Actual output of `python mc_option_pricing.py`, seed 2022, $S_0=100$, $K=100$, $r=5\%$, $\sigma=22\%$, $T=1$, four time steps and 100,000 paths. Black–Scholes call price: **11.2028**.

| Estimator | Price | Standard error | SE reduction vs. plain | Relative sampling efficiency |
|---|---:|---:|---:|---:|
| Plain | 11.1503 | 0.05108 | 0.0% | 1.00x |
| Antithetic | 11.1254 | 0.03700 | 27.6% | 1.91x |
| Control | 11.1657 | 0.03558 | 30.3% | 2.06x |
| Joint antithetic + control | 11.1602 | 0.02754 | 46.1% | 3.44x |

### Convergence comparison

Same parameters and seed; absolute errors are against Black–Scholes. A single random run need not improve monotonically at every sample size.

| Paths per estimator | Plain price | Plain SE | Plain absolute error | Joint price | Joint SE | Joint absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 10.4727 | 0.49398 | 0.7301 | 10.7976 | 0.26497 | 0.4052 |
| 10,000 | 11.0405 | 0.16109 | 0.1623 | 11.1479 | 0.08662 | 0.0549 |
| 50,000 | 11.1166 | 0.07212 | 0.0862 | 11.1152 | 0.03882 | 0.0876 |
| 100,000 | 11.1503 | 0.05108 | 0.0525 | 11.1602 | 0.02754 | 0.0426 |

Central finite differences reuse common random numbers for bumped valuations: call Delta **0.6320** (Black–Scholes 0.6320), Vega **37.459** (Black–Scholes 37.689).

## 2. Demand forecasting and safety-stock estimation

The synthetic series has 52 weekly observations. The first 40 weeks are used for training and the last 12 for validation; frequency 4 represents an **assumed** four-week cycle, giving ten training cycles. Four prespecified SARIMA candidates with common differencing orders are fitted on training data; the script prints AIC and BIC and selects the lowest finite **AIC**. It reports ADF and Ljung–Box p-values with conditional interpretations. Neither test proves stationarity or white noise.

### Example run: synthetic data

The following results are from a user-supplied R run of `sarima_demand_forecast.R`. Model selection and diagnostics use the 40-week training set only. Results may vary with R and package versions.

| Candidate SARIMA | AIC | BIC |
|---|---:|---:|
| (0,1,0)(0,1,0)[4] | 178.7053 | 180.2606 |
| (1,1,0)(0,1,0)[4] | 179.2693 | 182.3799 |
| **(0,1,1)(0,1,1)[4]** | **163.3689** | **168.0349** |
| (1,1,0)(1,1,0)[4] | 170.7279 | 175.3940 |

The selected model is **SARIMA(0,1,1)(0,1,1)[4]**, based on minimum AIC among these candidates. The ADF test on the first-differenced training series gave **p = 0.1157**, so the unit-root null was not rejected at 5%. The selected model's Ljung–Box test gave **Q(8) = 4.369, p = 0.627**: no significant residual autocorrelation was detected at 5%, which does not prove white noise.

Unadjusted forecasts were evaluated on the 12-week holdout:

| Model | MAPE | RMSE |
|---|---:|---:|
| Seasonal naive | 15.6% | 41.35 kg |
| Selected SARIMA | 15.1% | 37.89 kg |

The relative MAPE reduction is **3.7%**, calculated from unrounded values. This is one holdout on synthetic data, not a demonstrated real-world improvement.

### Illustrative safety-stock calculation

The same run produced a theoretical safety stock of **77.0 kg** and reorder point (ROP) of **537.9 kg**. These are **theoretical estimates for a synthetic example under simplified independent, homoscedastic forecast-error assumptions**, with a two-week lead time and a 95% normal-quantile target. The formula scales weekly error SD by the square root of lead time; it ignores possible serial correlation, forecast bias and parameter uncertainty. The error SD is calculated from validation actuals, so this is a retrospective illustration, not a reorder decision available before the holdout period. A 95% target is not evidence that a 95% service level was achieved.

## 3. E-commerce and supplier SQL examples

The illustrative SQL tables contain 25 factory entries and 184 aggregate orders. Their phase-level calculation yields conversion rates of 9.5% and 11.6% (22.1% relative increase after rounding) and 38.0% gross margin; the core fabric example has an 8.5% min-to-max price spread. These are properties of constructed data, **not proof of actual performance or causal impact**. The table contains weekly aggregates, not individual orders or a full stage-by-stage funnel.

## Run locally

```bash
# Python (NumPy and SciPy)
pip install numpy scipy
python mc_option_pricing.py
python -m unittest discover -s tests

# R (install dependencies first if necessary)
Rscript -e 'install.packages(c("forecast", "tseries", "ggplot2"), repos="https://cloud.r-project.org")'
Rscript sarima_demand_forecast.R

# SQLite
sqlite3 :memory: < ecommerce_funnel_analytics.sql
```
