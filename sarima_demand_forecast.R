# ==============================================================================
# Time-Series Demand Forecasting & Safety-Stock Optimization
# Author: Kaihao Yan
#
# Models 52 weeks of synthetic catering consumption inspired by personal
# experience; the four-week cycle is an illustrative business assumption.
# Forecast evaluation uses 40 training weeks and 12 untouched holdout weeks.
# Safety stock is a theoretical estimate under simplified independent errors.
# ==============================================================================
suppressPackageStartupMessages({
library(forecast)
library(tseries)
library(ggplot2)
})
# 1. Synthetic illustrative 52-week catering consumption series (kg/week)
# Assumed four-week replenishment cycle, not a demonstrated empirical seasonality
consumption_kg <- c(
# Weeks 1-40: Training Window (10 four-week cycles)
210.4, 195.2, 182.6, 228.5,  214.8, 198.0, 185.1, 232.0,
218.2, 201.5, 188.4, 236.8,  215.0, 199.2, 186.0, 234.1,
222.5, 205.8, 191.2, 241.0,  226.4, 209.1, 194.5, 245.2,
224.0, 207.3, 192.8, 242.6,  229.8, 212.4, 197.0, 248.5,
233.1, 215.6, 200.2, 252.0,  236.5, 218.0, 202.5, 255.4,
# Weeks 41-52: Out-of-Sample Validation Window (12 weeks)
278.5, 252.0, 174.5, 298.0,  284.2, 256.8, 176.2, 304.5,
289.0, 261.2, 178.0, 309.2
)
season_period <- 4
train_weeks   <- 40
valid_weeks   <- 12
y_full  <- ts(consumption_kg, frequency = season_period)
y_train <- window(y_full, end = c(10, 4))
y_valid <- window(y_full, start = c(11, 1))
cat("\n")
cat("TIME-SERIES DEMAND FORECASTING & SAFETY-STOCK OPTIMIZATION (52 WEEKS)\n")
cat("\n")
cat(sprintf("Total observations : %d weeks (Train: %d weeks | Validation: %d weeks)\n",
length(y_full), length(y_train), length(y_valid)))
# 2. ADF test on the first-differenced training series
adf_res <- suppressWarnings(adf.test(diff(y_train)))
adf_message <- if (is.na(adf_res$p.value)) {
  "Test inconclusive"
} else if (adf_res$p.value < 0.05) {
  "Reject unit-root null at 5%"
} else {
  "Cannot reject unit-root null at 5%"
}
cat(sprintf("ADF test on differenced training series: p-value = %.4f (%s)\n\n",
            adf_res$p.value, adf_message))

# 3. Compare a prespecified candidate set on training data only. All candidates
# share the same differencing orders; select the lowest finite AIC, not BIC.
candidates <- list(
  list(order = c(0, 1, 0), seasonal = c(0, 1, 0)),
  list(order = c(1, 1, 0), seasonal = c(0, 1, 0)),
  list(order = c(0, 1, 1), seasonal = c(0, 1, 1)),
  list(order = c(1, 1, 0), seasonal = c(1, 1, 0))
)
fits <- lapply(candidates, function(spec) {
  tryCatch(Arima(y_train, order = spec$order,
                 seasonal = list(order = spec$seasonal, period = season_period),
                 method = "ML"),
           error = function(e) {
             message("Candidate fit failed: ", conditionMessage(e))
             NULL
           })
})
labels <- vapply(candidates, function(spec) {
  sprintf("(%s)(%s)[%d]", paste(spec$order, collapse = ","),
          paste(spec$seasonal, collapse = ","), season_period)
}, character(1))
criteria <- data.frame(
  Model = labels,
  AIC = vapply(fits, function(fit) if (is.null(fit)) NA_real_ else fit$aic, numeric(1)),
  BIC = vapply(fits, function(fit) if (is.null(fit)) NA_real_ else fit$bic, numeric(1))
)
cat("Candidate models (training data only):\n")
print(criteria, row.names = FALSE)
valid_aic <- which(is.finite(criteria$AIC))
if (!length(valid_aic)) stop("No candidate produced a finite AIC")
selected <- valid_aic[which.min(criteria$AIC[valid_aic])]
sarima_fit <- fits[[selected]]
cat(sprintf("Selected by minimum AIC: SARIMA%s\n\n", labels[selected]))

# 4. Ljung-Box test: account for fitted AR and MA parameters.
spec <- candidates[[selected]]
fitdf <- sum(spec$order[c(1, 3)], spec$seasonal[c(1, 3)])
lb_test <- Box.test(residuals(sarima_fit), lag = 8, type = "Ljung-Box", fitdf = fitdf)
lb_message <- if (is.na(lb_test$p.value)) {
  "Test inconclusive"
} else if (lb_test$p.value > 0.05) {
  "No significant residual autocorrelation detected"
} else {
  "Significant residual autocorrelation detected"
}
cat(sprintf("Ljung-Box residual diagnostic: Q(8) = %.3f, p-value = %.3f (%s)\n\n",
            lb_test$statistic, lb_test$p.value, lb_message))
# 5. Out-of-Sample Validation (12-Week Holdout) vs. Seasonal Naive Baseline
fc_sarima <- forecast(sarima_fit, h = valid_weeks)
fc_snaive <- snaive(y_train, h = valid_weeks)
sarima_pred <- as.numeric(fc_sarima$mean)
snaive_pred <- as.numeric(fc_snaive$mean)
actual_val  <- as.numeric(y_valid)
mape_snaive <- mean(abs((actual_val - snaive_pred) / actual_val)) * 100
mape_sarima <- mean(abs((actual_val - sarima_pred) / actual_val)) * 100
rel_reduction <- 100 * (1 - mape_sarima / mape_snaive)
rmse_snaive <- sqrt(mean((actual_val - snaive_pred)^2))
rmse_sarima <- sqrt(mean((actual_val - sarima_pred)^2))
cat("----------------------------------------------------------------------\n")
cat("12-WEEK VALIDATION SPLIT ACCURACY BENCHMARK\n")
cat("----------------------------------------------------------------------\n")
cat(sprintf("Seasonal Naive Baseline MAPE : %.1f%%  (RMSE = %.2f kg)\n", mape_snaive, rmse_snaive))
cat(sprintf("SARIMA Model Forecast MAPE   : %.1f%%  (RMSE = %.2f kg)\n", mape_sarima, rmse_sarima))
cat(sprintf("Relative MAPE Reduction      : %.1f%%\n\n", rel_reduction))
# 6. Theoretical estimate under simplified independent, homoscedastic errors.
# The sqrt(lead time) scaling ignores serial correlation, bias and uncertainty
# in estimating error variance. The 95% target is not an observed service level.
service_level <- 0.95
z_score       <- qnorm(service_level)
lead_time_wks <- 2
sigma_error   <- sd(actual_val - sarima_pred)
safety_stock  <- z_score * sigma_error * sqrt(lead_time_wks)
mean_lt_demand <- mean(sarima_pred[1:lead_time_wks]) * lead_time_wks
reorder_point <- mean_lt_demand + safety_stock
cat("----------------------------------------------------------------------\n")
cat("THEORETICAL INVENTORY ESTIMATE (95% TARGET, 2-WEEK LEAD TIME)\n")
cat("----------------------------------------------------------------------\n")
cat(sprintf("Theoretical Safety Stock     : %.1f kg\n", safety_stock))
cat(sprintf("Derived Reorder Point (ROP)  : %.1f kg\n", reorder_point))
cat("======================================================================\n")
