# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Volatility of GB Electricity Cash-Out Prices
#
# This notebook models the volatility of Great Britain electricity cash-out
# (imbalance) prices using 2024 settlement data from Elexon's BMRS API. The
# single imbalance price is the price at which participants settle their energy
# imbalances, and it is the price a flexible asset such as a battery is exposed
# to whenever it deviates from its contracted position. Understanding how that
# price behaves, and in particular how volatile it is, is the starting point for
# valuing flexibility.
#
# The analysis builds an OLS-to-GARCH pipeline and reaches two findings:
#
# 1. Net imbalance volume (NIV) is the proximate driver of the cash-out price.
#    Demand and wind forecast error act through it rather than independently.
# 2. Price volatility is strongly asymmetric. Surprises when the system is short
#    are large and unbounded on the upside, while surprises when the system is
#    long are compressed near the price floor.
#
# Data is read from cached parquet files produced by a separate pull script
# against the Elexon BMRS API.

# %%
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.stats.diagnostic import (
    het_white,
    het_breuschpagan,
    acorr_breusch_godfrey,
    het_arch,
)
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Data
#
# All series are half-hourly settlement-period data for calendar year 2024.
#
# One caveat on vintage. The price and NIV series reflect near-settled values
# rather than the point-in-time figures available in real time, because BMRS
# revises settlement data through reconciliation runs. The results below should
# therefore be read as a description of settled outcomes, not as a real-time
# tradeable signal. Demand uses the initial transmission system demand outturn,
# which is closer to a point-in-time figure.

# %%
price = pd.read_parquet("system_prices_2024.parquet")
demand = pd.read_parquet("demand_2024.parquet")
wind = pd.read_parquet("wind_2024.parquet")
forecast = pd.read_parquet("forecast_2024.parquet")

# %%
# Keep the columns used downstream and join on the half-hourly index.
price = price[["netImbalanceVolume", "systemSellPrice", "sellPriceAdjustment"]]
demand_series = demand[["initialTransmissionSystemDemandOutturn"]]

df = price.join(demand_series, how="inner")
df = df.join(wind, how="inner")

# The wind forecast is published less frequently than the half-hourly grid,
# so it is reindexed onto the price index and forward-filled by at most one step.
forecast_hh = forecast.reindex(df.index).ffill(limit=1)
df["wind_forecast"] = forecast_hh["generation"]

# %% [markdown]
# ## 2. Feature construction
#
# NIV is split by sign because the price responds asymmetrically to a short
# versus a long system. Under the Elexon convention a positive NIV means the
# system is short (more energy was needed than was contracted), so `NIV_short`
# carries the positive values and `NIV_long` the negative ones.
#
# Net demand is demand less wind generation. The raw wind forecast error is
# actual wind less forecast wind, which is revisited in the next section before
# it is used.

# %%
df["NIV_short"] = df["netImbalanceVolume"] * (df["netImbalanceVolume"] >= 0)
df["NIV_long"] = df["netImbalanceVolume"] * (df["netImbalanceVolume"] < 0)
df["net_demand"] = df["initialTransmissionSystemDemandOutturn"] - df["wind"]

# Naive forecast error. Tested for bias in section 3 before any use.
df["wind_error"] = df["wind"] - df["wind_forecast"]

# %%
df.dropna(inplace=True)

# Calendar controls.
df.index = pd.to_datetime(df.index)
df["peak"] = ((df.index.hour >= 16) & (df.index.hour <= 19)).astype(int)
df["winter"] = df.index.month.isin([11, 12, 1, 2, 3]).astype(int)
df["weekend"] = (df.index.dayofweek >= 5).astype(int)

# %% [markdown]
# ## 3. Is the wind forecast unbiased?
#
# Before using wind forecast error as a regressor it is worth checking whether
# the forecast is rational. Regressing actual wind on forecast wind is a
# Mincer-Zarnowitz test: an efficient, correctly-scoped forecast should give an
# intercept of zero and a slope of one.
#
# The regression below fails both conditions. The intercept is well above zero
# and the slope is materially below one. That pattern means the forecast and the
# outturn do not cover an identical definition of wind, so the raw error carries
# a deterministic, predictable component rather than pure surprise. Feeding it
# straight into the price model would inject that mechanical bias.
#
# The fix is to use the regression residual as the wind "surprise", which isolates the
# wind dynamics unable to be forecasted.

# %%
wind_reg = sm.ols("wind ~ wind_forecast", data=df).fit(
    cov_type="HAC", cov_kwds={"maxlags": 48}
)
wind_reg.summary()

# %%
# Corrected wind surprise: uncorrelated with the forecast.
df["true_wind_error"] = wind_reg.resid

# %% [markdown]
# ## 4. Linear model and diagnostics
#
# The first specification regresses the cash-out price on the NIV terms, the
# true wind error, net demand, and calendar dummies, with HAC standard
# errors for heteroskedasticity and serial correlation.
#
# The calendar dummies and the wind error are insignificant. The wind result is
# the first sign of mediation: once realised NIV is in the model, wind surprise
# has no separate effect on the price level, because its effect runs through the
# imbalance it creates. The model is then reduced to the NIV terms and net demand
# for the diagnostic stage.

# %%
results = sm.ols(
    "systemSellPrice ~ NIV_short + NIV_long + true_wind_error + net_demand"
    " + C(peak) + C(winter) + C(weekend)",
    data=df,
).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
results.summary()

# %%
# Parsimonious spec used for the diagnostic ladder.
results_no_dummy = sm.ols(
    "systemSellPrice ~ NIV_short + NIV_long + net_demand",
    data=df,
).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
results_no_dummy.summary()

# %% [markdown]
# ### Residual diagnostics
#
# The residuals are tested for heteroskedasticity (Breusch-Pagan, White), serial
# correlation (Breusch-Godfrey), and ARCH effects (Engle).

# %%
bp_stat, bp_p, _, _ = het_breuschpagan(
    results_no_dummy.resid, results_no_dummy.model.exog
)
white_stat, white_p, _, _ = het_white(
    results_no_dummy.resid, results_no_dummy.model.exog
)
print(f"Breusch-Pagan p-value: {bp_p:.4g}")
print(f"White's      p-value: {white_p:.4g}")

# %%
bg_stat, bg_p, _, _ = acorr_breusch_godfrey(results_no_dummy, nlags=48)
print(f"Breusch-Godfrey p-value (48 lags): {bg_p:.4g}")

# %%
arch_stat, arch_p, _, _ = het_arch(results_no_dummy.resid, nlags=12)
print(f"Engle ARCH-LM p-value (12 lags): {arch_p:.4g}")

# %%
plot_acf(results_no_dummy.resid, lags=48)
plt.title("ACF of residuals (no lag term)")
plt.show()

# %% [markdown]
# The residuals show strong serial correlation, so a t-1 price lag is
# added.

# %%
df["sell_lag1"] = df["systemSellPrice"].shift(1)

results_w_lag = sm.ols(
    "systemSellPrice ~ NIV_short + NIV_long + net_demand + sell_lag1",
    data=df,
).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
results_w_lag.summary()

# %%
bg_stat, bg_p_lag, _, _ = acorr_breusch_godfrey(results_w_lag, nlags=48)
print(f"Breusch-Godfrey p-value (48 lags): {bg_p_lag:.4g}")

# %%
plot_acf(results_w_lag.resid, lags=48)
plt.title("ACF of residuals (with lag term)")
plt.show()

plot_acf(results_w_lag.resid**2, lags=48)
plt.title("ACF of squared residuals (with lag term)")
plt.show()

# %% [markdown]
# The lag removes most of the autocorrelation in the residuals, but the squared
# residuals remain autocorrelated. This implies conditional
# heteroskedasticity, and it is the motivation for a GARCH variance model.

# %% [markdown]
# ## 5. GJR-GARCH
#
# The model is a GJR-GARCH with Student-t innovations and an AR(1) mean carrying
# the NIV and demand regressors. GJR is chosen over plain GARCH so the variance
# can respond asymmetrically to positive and negative shocks. The Student-t
# captures the heavy tails of cash-out prices. Regressors are scaled to
# gigawatts to help the optimiser converge.

# %%
df["net_demand_gw"] = df["net_demand"] / 1000
df["NIV_short_gw"] = df["NIV_short"] / 1000
df["NIV_long_gw"] = df["NIV_long"] / 1000

# %%
model = arch_model(
    df["systemSellPrice"],
    x=df[["NIV_short_gw", "NIV_long_gw", "net_demand_gw"]],
    mean="ARX",
    lags=1,
    vol="GARCH",
    o=1,  # asymmetry term -> GJR-GARCH
    p=1,
    q=1,
    dist="t",
)
res = model.fit(disp="off")
print(res.summary())

# %% [markdown]
# Three features of the fit are worth noting.
#
# - NIV dominates the mean equation and is highly significant on both sides.
#   Wind surprise was dropped here because it was insignificant once NIV was
#   present (section 4).
# - The volatility persistence (alpha + beta + gamma/2) sits close to one, so
#   variance shocks decay slowly and there is little mean reversion in
#   volatility. This matters for any multi-step forecast built on the model.
# - The degrees-of-freedom parameter is low, confirming very heavy tails.
#
# The asymmetry term gamma is significant. The next section investigates what it
# represents.

# %% [markdown]
# ## 6. What does the asymmetry represent?
#
# A significant asymmetry term is often labelled a scarcity effect in power
# markets. To find what out what the asymmetry represents, the 100 largest positive
# and 100 largest negative residuals are pulled and their characteristics
# compared against the full sample.

# %%
top_100 = res.resid.nlargest(100)
bottom_100 = res.resid.nsmallest(100)

check_top = df.loc[top_100.index]
check_bot = df.loc[bottom_100.index]

cols = ["systemSellPrice", "netImbalanceVolume", "net_demand", "true_wind_error"]

# %%
print("Largest POSITIVE residuals (price above model prediction):")
print(check_top[cols].describe())

# %%
print("Largest NEGATIVE residuals (price below model prediction):")
print(check_bot[cols].describe())

# %%
print("Full sample, for reference:")
print(df[cols].describe())

# %% [markdown]
# The split is clear and it is driven by NIV, not by demand or wind.
#
# The large positive residuals (price came in above the model) occur when the
# system is short, with prices running high. The large negative residuals (price
# below the model) occur when the system is long, with prices compressed near the
# floor. Demand and wind surprise sit near their full-sample means in both tails,
# so neither separates the extremes.
#
# The economic content is that a short system has unbounded upside through
# scarcity pricing, whereas a long system is floored ~0 or slightly negative. That asymmetry in
# the size of the residuals is what the gamma term captures. It is the bounded-versus-unbounded shape of
# imbalance pricing.

# %% [markdown]
# ## 7. Standardised residual diagnostics
#
# As a final check, the standardised residuals are inspected for remaining
# structure. If the variance model is adequate, the clustering visible earlier in
# the squared residuals should be largely removed.

# %%
std_resid = res.std_resid.dropna()

plot_acf(std_resid**2, lags=48)
plt.title("ACF of squared standardised residuals")
plt.show()

plot_acf(std_resid, lags=48)
plt.title("ACF of standardised residuals")
plt.show()

# %% [markdown]
# ## 8. Conclusion
#
# The cash-out price in 2024 is driven primarily by net imbalance volume, with
# demand and wind forecast error mediated through it. Volatility is asymmetric,
# and the asymmetry reflects the unbounded upside of a short system against the
# floored downside of a long one.
#
# ### Limitations
#
# - In-sample analysis on a single year, so it describes one volatility regime
#   (post-2022 gas normalisation) rather than a stable structural relationship.
# - NIV enters contemporaneously, so the model explains the price but does not
#   forecast it. i.e. realised NIV is not known before the settlement period.
# - The data are near-settled vintages rather than point-in-time.
#
# These are deliberate scope choices for a descriptive volatility study before moving to a
# forecast model.
#
# ### Future work
#
# - An out-of-sample density forecast with walk-forward evaluation, to test
#   predictive calibration. The central problem is that the main driver, NIV, is
#   not known ex ante, so the forecast must condition on the pre-settlement
#   information set or on NIV scenarios.
# - Testing whether imbalance magnitude drives the conditional variance directly,
#   via an exogenous term in the variance equation (GARCH-X). This needs a custom
#   volatility process and raises a contemporaneity question, since NIV is
#   realised within the period.
