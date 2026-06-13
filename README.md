# GB Electricity Cash-Out Price Volatility

Modelling the volatility of GB's electricity imbalance (cash-out) price using 2024 settlement data from the Elexon BMRS API. The cash-out price is what a participant pays or receives when it is out of balance against its contracted position, and it is the price a flexible asset such as a battery is exposed to whenever it deviates from plan. This project characterises how that price behaves and, in particular, what drives its volatility.

## Findings

- Net imbalance volume (NIV) is the proximate driver of the cash-out price. Demand and wind forecast error have no separate effect on the price level once NIV is included; they act through the imbalance they create.
- Price volatility is asymmetric. The largest surprises split cleanly by the sign of the system imbalance. When the system is short, prices run high with unbounded upside through scarcity pricing. When it is long, prices are compressed near the floor. The asymmetry term in the volatility model (GJR-GARCH) captures this bounded-versus-unbounded shape rather than a conventional scarcity effect.

## Data

All series are half-hourly settlement-period data for 2024, pulled from the Elexon BMRS API.

| Series | Endpoint | Notes |
|---|---|---|
| System price, NIV | `/balancing/settlement/system-prices` | single imbalance price |
| Demand | `/demand/outturn` | initial transmission demand outturn |
| Wind generation | `/generation/outturn/summary` | actual, by fuel type |
| Wind forecast | `/forecast/generation/wind/earliest` | earliest published WINDFOR |

Vintage: price and NIV are near-settled values rather than point-in-time, because BMRS revises settlement data through reconciliation runs. The wind forecast uses the earliest published value for each period, so it is ex-ante with no look-ahead. Results are therefore a description of settled outcomes, not a real-time tradeable signal.

Market data is published by Elexon via the BMRS API and is subject to Elexon's data licence terms.

## Repository

- `pull_elexon_data.py` ingestion: fetches the four series and caches them as parquet.
- `gb_cashout_volatility.ipynb` analysis: the full pipeline with narration.
- `gb_cashout_volatility.py` the notebook as a jupytext percent script (paired source).

## Reproduce

Requires Python 3.10+ with pandas, numpy, requests, tqdm, statsmodels, arch, matplotlib, and pyarrow.

```bash
pip install pandas numpy requests tqdm statsmodels arch matplotlib pyarrow jupytext

python pull_elexon_data.py                    # fetch and cache the parquet files
jupyter notebook gb_cashout_volatility.ipynb  # then run all cells
```

The pull takes a few minutes and writes four parquet files to the working directory. The notebook reads those and runs end to end.

## Method

The pipeline escalates from a linear model to a volatility model, with each step motivated by a diagnostic on the previous one.

1. OLS of the cash-out price on the NIV terms, net demand, and a wind forecast error, with HAC standard errors. A Mincer-Zarnowitz test first showed the raw wind forecast was mechanically biased against the outturn series, so the wind surprise is taken as the residual of actual on forecast wind to capture the "surprise".
2. Residual diagnostics (Breusch-Pagan, White, Breusch-Godfrey, Engle ARCH-LM) reject homoskedasticity and independence. The squared residuals remain autocorrelated after adding a price lag, which is the signature of conditional heteroskedasticity and the motivation for a GARCH variance.
3. A GJR-GARCH with Student-t innovations and an AR(1) mean. The significant asymmetry term is then investigated by inspecting the largest residuals, rather than assumed to represent scarcity.

## Scope and limitations

This is an in-sample analysis of a single year, so it describes one volatility regime (post-2022 gas normalisation) rather than a stable structural relationship. NIV enters contemporaneously, so the model explains the price but does not forecast it: realised NIV is not known before the settlement period.

## Future work

- An out-of-sample density forecast with walk-forward evaluation. The central difficulty is that NIV, the main driver, is not known ex ante, so the forecast must condition on the pre-settlement information set or on NIV scenarios.
- Testing whether imbalance magnitude drives the conditional variance directly through an exogenous term in the variance equation (GARCH-X), which requires handling the contemporaneity of NIV.
