# Soccer Value Predictor (free-data research tool)

This project improves the uploaded `api_football_daily.py` with several complementary methods:
- Team form and home/away splits
- Elo ratings
- Logistic regression, random forest and histogram gradient boosting for 1X2
- Poisson goal model with a Dixon–Coles-style low-score adjustment
- A transparent probability blend with normalized bookmaker 1X2 probabilities when odds exist
- 1X2, double chance, Over/Under 2.5, BTTS, and most-likely correct score estimates
- Value screening using estimated probability, odds, probability edge, and expected value
- Chronological holdout backtest with accuracy, Brier score, log loss, and illustrative ROI

## Important limitations
1. **Not truly worldwide.** `football-data.co.uk` provides selected leagues and not every country or competition. Predictions are limited to leagues and fixtures in its files. The script does not pretend to cover every global match.
2. **Odds coverage is incomplete.** If odds for a market are absent, it cannot assess value for that market. This version screens 1X2 and goal markets only when the relevant odds exist; BTTS and double-chance odds are usually unavailable in the free source.
3. **No proven profitability.** Backtesting is a historical diagnostic, not a guarantee. The current backtest tests 1X2 model probabilities against historical 1X2 odds and uses a chronological holdout, but it is not a full walk-forward retraining simulation for every match. Treat ROI as preliminary.
4. Team names and league codes must match the source. Newly promoted teams and low-data teams can be unreliable.
5. The scoreline output is the single most likely score, not a high-probability outcome. Exact-score probabilities are usually low.
6. The blend weights and low-score correction are transparent heuristics and should be selected using nested chronological validation, not optimized against the final test set.

## Run on Windows
Install Python 3.10+ from python.org, then in the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python api_football_daily.py --date today
```

Open `output/report.html` in your browser. Other outputs are `output/predictions.csv` and `output/predictions.json`.

## Android + GitHub Actions (recommended for phone-only use)
1. Create a GitHub account and a new repository.
2. Upload `api_football_daily.py`, `requirements.txt`, and `.github/workflows/daily.yml`.
3. Open the repository's **Actions** tab and enable workflows.
4. Run **Daily soccer analysis** manually using **Run workflow**, or wait for the scheduled run.
5. The workflow uploads the `output` folder as a downloadable artifact. On Android, open the workflow run in GitHub and download the artifact, then open `report.html`.
6. To use a particular date, edit the workflow's `--date today` argument to `--date YYYY-MM-DD`, or run locally in Termux.

GitHub-hosted scheduled workflows may start late; they are not guaranteed to run at an exact minute.

## Commands
```bash
# Today's fixtures
python api_football_daily.py --date today

# Tomorrow
python api_football_daily.py --date tomorrow

# Particular date
python api_football_daily.py --date 2026-10-31

# Run only the historical holdout test
python api_football_daily.py --backtest-only

# Restrict to supported leagues (examples: E0 England Premier League, SP1 Spain La Liga, I1 Italy Serie A)
python api_football_daily.py --leagues E0,SP1,I1

# Looser/tighter value screen
python api_football_daily.py --min-edge 0.05 --min-ev 0.04

# Download more seasons (2 to 8)
python api_football_daily.py --seasons 7
```

## How value is calculated
For decimal odds `O` and estimated probability `p`:
- Expected value per unit stake: `EV = p * O - 1`
- Raw implied probability: `1 / O`
- Edge shown by this script: `p - (1 / O)`

The edge compares against raw odds and therefore does not remove the bookmaker margin for an individual outcome. For 1X2 model features, the 3-way market probabilities are normalized to sum to 100%, but the displayed candidate edge uses raw implied probability. Use these as screening signals, not proof of an advantage.

## Recommended next improvements
- Add more genuinely global free data sources (availability and licensing vary).
- Store downloaded data in a cache so each run is faster and less dependent on source availability.
- Add market-specific historical odds and backtests for O/U and BTTS.
- Add calibration plots, confidence intervals, closing-line value, and rolling out-of-sample retraining.
- Use separate models per league where sample size supports it.
