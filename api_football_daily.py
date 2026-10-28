#!/usr/bin/env python3
"""
Free-data soccer prediction and value-bet research tool.

Data source: football-data.co.uk (historical results/odds + upcoming fixtures.csv).
Models: chronological team-form features, Elo ratings, multinomial logistic regression,
random forest, Poisson score probabilities with a low-score Dixon-Coles-style correction,
and a market/model ensemble. Outputs are estimates, not guaranteed tips.

Important coverage limitation:
football-data.co.uk covers selected leagues, not every league worldwide. The script
only predicts fixtures present in its fixtures.csv file and supported league list.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_URL = "https://www.football-data.co.uk"
USER_AGENT = "Mozilla/5.0 (compatible; SoccerValueResearch/1.0)"
LEAGUES = [
    ("E0", "Premier League", "England"), ("E1", "Championship", "England"),
    ("E2", "League One", "England"), ("E3", "League Two", "England"),
    ("EC", "National League", "England"), ("SC0", "Premiership", "Scotland"),
    ("SC1", "Championship", "Scotland"), ("SC2", "League One", "Scotland"),
    ("SC3", "League Two", "Scotland"), ("D1", "Bundesliga", "Germany"),
    ("D2", "2. Bundesliga", "Germany"), ("I1", "Serie A", "Italy"),
    ("I2", "Serie B", "Italy"), ("SP1", "La Liga", "Spain"),
    ("SP2", "La Liga 2", "Spain"), ("F1", "Ligue 1", "France"),
    ("F2", "Ligue 2", "France"), ("N1", "Eredivisie", "Netherlands"),
    ("B1", "Pro League", "Belgium"), ("P1", "Primeira Liga", "Portugal"),
    ("T1", "Super Lig", "Turkey"), ("G1", "Super League", "Greece"),
]
DIV_INFO = {code: (name, country) for code, name, country in LEAGUES}
DIV_ORDER = {code: i for i, (code, _, _) in enumerate(LEAGUES)}
FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga",
    "home_home_gf", "home_home_ga", "away_away_gf", "away_away_ga",
    "home_form_points", "away_form_points", "home_rest_days", "away_rest_days",
    "market_home", "market_draw", "market_away",
]
MIN_TEAM_MATCHES = 3
DEFAULT_SEASONS = 5
RHO = -0.08  # conservative low-score dependence correction; tune only by validation


def fetch_text(url: str, required: bool = False):
    last_error = "unknown error"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=35) as response:
                return response.read().decode("latin-1", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = f"HTTP {exc.code}"
        except Exception as exc:
            last_error = str(exc)
        if attempt < 2:
            time.sleep(attempt + 1)
    if required:
        raise RuntimeError(f"Could not download {url}: {last_error}")
    print(f"Warning: skipping {url}: {last_error}", file=sys.stderr)
    return None


def read_csv_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def parse_match_date(value):
    value = (value or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def number(row, *keys):
    for key in keys:
        try:
            val = float(str(row.get(key, "")).strip())
            if math.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass
    return None


def season_codes(today):
    start = today.year if today.month >= 7 else today.year - 1
    codes = []
    for offset in range(DEFAULT_SEASONS):
        y = start - offset
        codes.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return codes


def local_fixture_datetime(day, clock, timezone_name="Africa/Nairobi"):
    try:
        hh, mm = [int(x) for x in (clock or "15:00").split(":")[:2]]
    except Exception:
        hh, mm = 15, 0
    naive = datetime(day.year, day.month, day.day, hh, mm)
    try:
        from zoneinfo import ZoneInfo
        return naive.replace(tzinfo=ZoneInfo("Europe/London")).astimezone(
            ZoneInfo(timezone_name)).replace(tzinfo=None)
    except Exception:
        return naive


def odds_from_row(row):
    def first(*keys):
        for key in keys:
            v = number(row, key)
            if v is not None and v > 1.0:
                return v
        return None
    return {
        "home": first("AvgH", "B365H", "PSH", "WHH"),
        "draw": first("AvgD", "B365D", "PSD", "WHD"),
        "away": first("AvgA", "B365A", "PSA", "WHA"),
        "over25": first("Avg>2.5", "B365>2.5", "P>2.5"),
        "under25": first("Avg<2.5", "B365<2.5", "P<2.5"),
    }


def market_probabilities(odds):
    vals = [odds.get("home"), odds.get("draw"), odds.get("away")]
    if any(v is None or v <= 1 for v in vals):
        return [np.nan, np.nan, np.nan]
    inv = np.array([1.0 / v for v in vals], dtype=float)
    return (inv / inv.sum()).tolist()  # normalize to remove bookmaker overround


def poisson_pmf(k, lam):
    lam = max(0.05, min(float(lam), 6.0))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_matrix(lam_home, lam_away, max_goals=8, rho=RHO):
    """Independent Poisson score matrix plus a small Dixon-Coles low-score correction."""
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
            # Dixon-Coles tau correction for 0-0, 0-1, 1-0 and 1-1.
            if h == 0 and a == 0:
                tau = 1 - lam_home * lam_away * rho
            elif h == 0 and a == 1:
                tau = 1 + lam_home * rho
            elif h == 1 and a == 0:
                tau = 1 + lam_away * rho
            elif h == 1 and a == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            mat[h, a] = max(0.0, p * tau)
    total = mat.sum()
    return mat / total if total else mat


def score_market(mat):
    home = float(np.tril(mat, -1).sum())  # row home goals > column away goals
    draw = float(np.trace(mat))
    away = float(np.triu(mat, 1).sum())
    over = float(sum(mat[h, a] for h in range(mat.shape[0]) for a in range(mat.shape[1]) if h + a >= 3))
    btts = float(sum(mat[h, a] for h in range(1, mat.shape[0]) for a in range(1, mat.shape[1])))
    # Most likely scoreline within represented score range.
    best = np.unravel_index(np.argmax(mat), mat.shape)
    return {
        "home": home, "draw": draw, "away": away,
        "over25": over, "under25": 1.0 - over, "btts_yes": btts,
        "btts_no": 1.0 - btts, "correct_score": f"{best[0]}-{best[1]}",
        "score_probability": float(mat[best]),
    }


def default_team():
    return {
        "elo": 1500.0, "games": 0, "last_date": None,
        "recent": deque(maxlen=8), "home_recent": deque(maxlen=8),
        "away_recent": deque(maxlen=8),
    }


def average(records, field, default):
    vals = [r[field] for r in records if r.get(field) is not None]
    return float(np.mean(vals)) if vals else default


def points_from_result(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def make_features(home, away, day, market_probs):
    league_gf = 1.45
    hr = list(home["recent"])
    ar = list(away["recent"])
    hhr = list(home["home_recent"])
    aar = list(away["away_recent"])
    def rest(team):
        if team["last_date"] is None:
            return 7.0
        return max(0.0, min(30.0, (day - team["last_date"]).days))
    return {
        "elo_diff": home["elo"] - away["elo"],
        "home_elo": home["elo"], "away_elo": away["elo"],
        "home_form_gf": average(hr, "gf", league_gf),
        "home_form_ga": average(hr, "ga", league_gf),
        "away_form_gf": average(ar, "gf", league_gf),
        "away_form_ga": average(ar, "ga", league_gf),
        "home_home_gf": average(hhr, "gf", league_gf),
        "home_home_ga": average(hhr, "ga", league_gf),
        "away_away_gf": average(aar, "gf", league_gf),
        "away_away_ga": average(aar, "ga", league_gf),
        "home_form_points": average(hr, "points", 1.2),
        "away_form_points": average(ar, "points", 1.2),
        "home_rest_days": rest(home), "away_rest_days": rest(away),
        "market_home": market_probs[0], "market_draw": market_probs[1],
        "market_away": market_probs[2],
    }


def update_team(team, gf, ga, day, is_home):
    team["recent"].append({"gf": gf, "ga": ga, "points": points_from_result(gf, ga)})
    team["home_recent" if is_home else "away_recent"].append({"gf": gf, "ga": ga})
    team["games"] += 1
    team["last_date"] = day


def update_elo(home, away, hg, ag):
    # Standard Elo expected-score update; goal margin modestly scales the K factor.
    eh = 1.0 / (1.0 + 10.0 ** (-(home["elo"] + 55.0 - away["elo"]) / 400.0))
    actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    margin = max(1.0, min(2.0, math.log(abs(hg - ag) + 1.0) + 1.0)) if hg != ag else 1.0
    delta = 20.0 * margin * (actual - eh)
    home["elo"] += delta
    away["elo"] -= delta


def load_historical_matches(leagues, today):
    all_rows = []
    codes = season_codes(today)
    for div in leagues:
        for season in codes:
            url = f"{BASE_URL}/mmz4281/{season}/{div}.csv"
            body = fetch_text(url)
            if not body:
                continue
            for row in read_csv_rows(body):
                day = parse_match_date(row.get("Date"))
                hg = number(row, "FTHG", "HG")
                ag = number(row, "FTAG", "AG")
                home, away = (row.get("HomeTeam") or "").strip(), (row.get("AwayTeam") or "").strip()
                if not day or hg is None or ag is None or not home or not away:
                    continue
                odds = odds_from_row(row)
                all_rows.append({
                    "date": day, "div": div, "home": home, "away": away,
                    "hg": int(hg), "ag": int(ag), "odds": odds,
                })
            time.sleep(0.08)
    # Deduplicate matches in case a season is repeated/overlapping.
    unique = {}
    for r in all_rows:
        unique[(r["div"], r["date"], r["home"], r["away"])] = r
    return sorted(unique.values(), key=lambda r: (r["date"], DIV_ORDER.get(r["div"], 999), r["home"]))


def build_training_data(matches):
    teams_by_div = defaultdict(lambda: defaultdict(default_team))
    X, y, dates, raw = [], [], [], []
    for match in matches:
        div = match["div"]
        teams = teams_by_div[div]
        h, a = teams[match["home"]], teams[match["away"]]
        probs = market_probabilities(match["odds"])
        feats = make_features(h, a, match["date"], probs)
        X.append([feats[k] for k in FEATURES])
        y.append(0 if match["hg"] > match["ag"] else (1 if match["hg"] == match["ag"] else 2))
        dates.append(match["date"])
        raw.append(match)
        update_elo(h, a, match["hg"], match["ag"])
        update_team(h, match["hg"], match["ag"], match["date"], True)
        update_team(a, match["ag"], match["hg"], match["date"], False)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), dates, raw, teams_by_div


def make_models():
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight=None, C=0.7)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=8, max_features=0.85,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=2.0, random_state=42)),
        ]),
    }


def fit_ensemble(X, y):
    models = make_models()
    fitted = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            fitted[name] = model
        except Exception as exc:
            print(f"Warning: model {name} skipped: {exc}", file=sys.stderr)
    if not fitted:
        raise RuntimeError("No model could be trained. Download more historical data.")
    return fitted


def predict_1x2(models, row):
    probs = []
    for model in models.values():
        p = model.predict_proba(row)[0]
        # Handle any absent classes safely.
        aligned = np.full(3, 1.0 / 3.0)
        for i, cls in enumerate(model.named_steps["model"].classes_):
            aligned[int(cls)] = p[i]
        aligned /= aligned.sum()
        probs.append(aligned)
    ml = np.mean(probs, axis=0)
    return ml


def expected_goals(home, away, league_matches):
    # Use current-season/available team form with shrinkage to league means.
    league_home = np.mean([m["hg"] for m in league_matches]) if league_matches else 1.45
    league_away = np.mean([m["ag"] for m in league_matches]) if league_matches else 1.15
    hrecent = list(home["recent"])
    arecent = list(away["recent"])
    hhome = list(home["home_recent"])
    aaway = list(away["away_recent"])
    home_attack = (average(hhome, "gf", league_home) * 4 + league_home * 4) / 8
    home_defence = (average(hhome, "ga", league_away) * 4 + league_away * 4) / 8
    away_attack = (average(aaway, "gf", league_away) * 4 + league_away * 4) / 8
    away_defence = (average(aaway, "ga", league_home) * 4 + league_home * 4) / 8
    lam_h = max(0.2, min(4.5, home_attack * away_defence / max(0.5, league_home)))
    lam_a = max(0.2, min(4.5, away_attack * home_defence / max(0.5, league_away)))
    return lam_h, lam_a


def walk_forward_backtest(X, y, dates, raw, min_train=500):
    """Chronological holdout; never trains on later matches for earlier test rows."""
    n = len(y)
    split = max(min_train, int(n * 0.80))
    if split >= n - 30:
        split = max(int(n * 0.70), n - 100)
    if split < 100 or n - split < 20:
        return {"message": "Not enough matches for a meaningful chronological holdout.", "matches": 0}
    models = fit_ensemble(X[:split], y[:split])
    correct = 0
    brier = []
    logloss = []
    bets = []
    for i in range(split, n):
        p = predict_1x2(models, X[i:i+1])
        actual = int(y[i])
        correct += int(int(np.argmax(p)) == actual)
        onehot = np.zeros(3); onehot[actual] = 1.0
        brier.append(float(np.sum((p - onehot) ** 2)))
        logloss.append(float(-math.log(max(1e-9, p[actual]))))
        odds = raw[i]["odds"]
        for idx, key in enumerate(("home", "draw", "away")):
            odd = odds.get(key)
            if odd and odd > 1.01 and p[idx] * odd - 1.0 >= 0.05:
                profit = (odd - 1.0) if actual == idx else -1.0
                bets.append(profit)
    return {
        "message": "Chronological holdout only; estimates are not proof of future profitability.",
        "matches": n - split,
        "accuracy": correct / (n - split),
        "mean_brier_1x2": float(np.mean(brier)),
        "mean_log_loss": float(np.mean(logloss)),
        "value_bets": len(bets),
        "value_bet_roi": float(sum(bets) / len(bets)) if bets else None,
        "training_matches": split,
        "test_start": str(dates[split]),
        "test_end": str(dates[-1]),
    }


def load_fixtures():
    body = fetch_text(f"{BASE_URL}/fixtures.csv", required=True)
    fixtures = []
    for row in read_csv_rows(body):
        div = (row.get("Div") or "").strip()
        day = parse_match_date(row.get("Date"))
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        if div not in DIV_INFO or not day or not home or not away:
            continue
        dt = local_fixture_datetime(day, row.get("Time"), "Africa/Nairobi")
        fixtures.append({
            "date": dt.date(), "kickoff": dt.strftime("%H:%M"), "div": div,
            "home": home, "away": away, "odds": odds_from_row(row),
        })
    return fixtures


def recent_state_for_future(matches):
    teams_by_div = defaultdict(lambda: defaultdict(default_team))
    league_rows = defaultdict(list)
    for m in matches:
        t = teams_by_div[m["div"]]
        h, a = t[m["home"]], t[m["away"]]
        update_elo(h, a, m["hg"], m["ag"])
        update_team(h, m["hg"], m["ag"], m["date"], True)
        update_team(a, m["ag"], m["hg"], m["date"], False)
        league_rows[m["div"]].append(m)
    return teams_by_div, league_rows


def markets_from_probs(p1x2, score_probs):
    home, draw, away = [float(x) for x in p1x2]
    return {
        "Home win": home, "Draw": draw, "Away win": away,
        "Home or draw (1X)": home + draw,
        "Draw or away (X2)": draw + away,
        "Home or away (12)": home + away,
        "Over 2.5 goals": score_probs["over25"],
        "Under 2.5 goals": score_probs["under25"],
        "BTTS Yes": score_probs["btts_yes"],
        "BTTS No": score_probs["btts_no"],
    }


def prediction_for_fixture(fixture, models, teams_by_div, league_rows, min_edge, min_ev):
    div = fixture["div"]
    teams = teams_by_div[div]
    home = teams[fixture["home"]]
    away = teams[fixture["away"]]
    market = market_probabilities(fixture["odds"])
    feats = make_features(home, away, fixture["date"], market)
    X = np.asarray([[feats[k] for k in FEATURES]], dtype=float)
    ml = predict_1x2(models, X)
    lam_h, lam_a = expected_goals(home, away, league_rows.get(div, []))
    sm = score_market(score_matrix(lam_h, lam_a))
    # Ensemble model and normalized bookmaker market. Use a conservative 65/35 model/market blend
    # when odds exist; otherwise use the model alone. This is a heuristic, not a fitted optimum.
    if all(math.isfinite(v) for v in market):
        ensemble = 0.65 * ml + 0.35 * np.asarray(market)
    else:
        ensemble = ml
    # Blend score-model 1X2 lightly into the ML/market ensemble.
    score_1x2 = np.array([sm["home"], sm["draw"], sm["away"]])
    ensemble = 0.8 * ensemble + 0.2 * score_1x2
    ensemble = ensemble / ensemble.sum()
    market_for_output = fixture["odds"]
    bets = []
    for name, p, key in (
        ("Home win", ensemble[0], "home"), ("Draw", ensemble[1], "draw"),
        ("Away win", ensemble[2], "away"),
        ("Over 2.5 goals", sm["over25"], "over25"),
        ("Under 2.5 goals", sm["under25"], "under25"),
        ("BTTS Yes", sm["btts_yes"], None), ("BTTS No", sm["btts_no"], None),
    ):
        odds = market_for_output.get(key) if key else None
        if odds and odds > 1.01:
            ev = float(p * odds - 1.0)
            implied = 1.0 / odds
            edge = float(p - implied)
            if edge >= min_edge and ev >= min_ev:
                bets.append({
                    "market": name, "probability": round(float(p), 4),
                    "odds": round(float(odds), 2), "implied_probability": round(implied, 4),
                    "edge": round(edge, 4), "expected_value": round(ev, 4),
                })
    bets.sort(key=lambda x: x["expected_value"], reverse=True)
    return {
        "date": str(fixture["date"]), "kickoff_nairobi": fixture["kickoff"],
        "league": DIV_INFO[div][0] + " · " + DIV_INFO[div][1],
        "home": fixture["home"], "away": fixture["away"],
        "expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
        "model_1x2": {
            "home_win": round(float(ensemble[0]), 4),
            "draw": round(float(ensemble[1]), 4),
            "away_win": round(float(ensemble[2]), 4),
        },
        "goal_model": {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in sm.items()},
        "double_chance": {
            "1X": round(float(ensemble[0] + ensemble[1]), 4),
            "X2": round(float(ensemble[1] + ensemble[2]), 4),
            "12": round(float(ensemble[0] + ensemble[2]), 4),
        },
        "correct_score_most_likely": sm["correct_score"],
        "correct_score_probability": round(sm["score_probability"], 4),
        "bookmaker_odds": market_for_output,
        "potential_value_bets": bets,
        "model_note": "Probabilities are model estimates. Only markets with available odds are screened for value.",
    }


def write_outputs(predictions, backtest, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "predictions.json"
    csv_path = outdir / "predictions.csv"
    report_path = outdir / "report.html"
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "data_source": "football-data.co.uk",
        "coverage_note": "Selected leagues only; not worldwide coverage.",
        "backtest": backtest,
        "predictions": predictions,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat = []
    for p in predictions:
        bets = p["potential_value_bets"]
        flat.append({
            "date": p["date"], "kickoff_nairobi": p["kickoff_nairobi"], "league": p["league"],
            "home": p["home"], "away": p["away"],
            "p_home": p["model_1x2"]["home_win"], "p_draw": p["model_1x2"]["draw"],
            "p_away": p["model_1x2"]["away_win"],
            "xg_home": p["expected_goals"]["home"], "xg_away": p["expected_goals"]["away"],
            "over25": p["goal_model"]["over25"], "btts_yes": p["goal_model"]["btts_yes"],
            "most_likely_score": p["correct_score_most_likely"],
            "value_bets": " ; ".join(f'{b["market"]} @ {b["odds"]} (EV {b["expected_value"]:+.1%})' for b in bets),
        })
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    rows = []
    for p in predictions:
        bet_html = "".join(
            f'<li>{b["market"]}: {b["probability"]:.1%} probability, odds {b["odds"]:.2f}, '
            f'edge {b["edge"]:.1%}, estimated EV {b["expected_value"]:+.1%}</li>'
            for b in p["potential_value_bets"]
        ) or "<li>No qualifying value bet based on available odds and thresholds.</li>"
        rows.append(f"""
        <section class="match">
          <div class="league">{p['date']} · {p['kickoff_nairobi']} EAT · {p['league']}</div>
          <h2>{p['home']} vs {p['away']}</h2>
          <p><b>1X2:</b> Home {p['model_1x2']['home_win']:.1%} · Draw {p['model_1x2']['draw']:.1%} · Away {p['model_1x2']['away_win']:.1%}</p>
          <p><b>Expected goals:</b> {p['expected_goals']['home']:.2f} – {p['expected_goals']['away']:.2f}
          &nbsp; <b>O2.5:</b> {p['goal_model']['over25']:.1%}
          &nbsp; <b>BTTS Yes:</b> {p['goal_model']['btts_yes']:.1%}</p>
          <p><b>Double chance:</b> 1X {p['double_chance']['1X']:.1%} · X2 {p['double_chance']['X2']:.1%} · 12 {p['double_chance']['12']:.1%}</p>
          <p><b>Most likely exact score:</b> {p['correct_score_most_likely']} ({p['correct_score_probability']:.1%} model probability)</p>
          <h3>Potential value candidates</h3><ul>{bet_html}</ul>
        </section>""")
    bt = backtest
    bt_html = (f"<p>Holdout matches: {bt.get('matches', 0)}; accuracy: "
               f"{bt.get('accuracy', float('nan')):.1%}; mean Brier score: "
               f"{bt.get('mean_brier_1x2', float('nan')):.3f}; mean log loss: "
               f"{bt.get('mean_log_loss', float('nan')):.3f}; value bets tested: "
               f"{bt.get('value_bets', 0)}; estimated ROI: "
               f"{(bt.get('value_bet_roi') if bt.get('value_bet_roi') is not None else float('nan')):.1%}</p>"
               if bt.get("matches") else f"<p>{bt.get('message', 'Backtest unavailable')}</p>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Soccer Value Research</title><style>
    body{{font:16px system-ui,Arial;margin:0;background:#f4f6f8;color:#17202a}}
    main{{max-width:1000px;margin:auto;padding:18px}} .match{{background:white;padding:16px;margin:14px 0;border-radius:12px}}
    .league{{color:#596579;font-size:.9rem}} h1{{margin-bottom:4px}} .warning{{background:#fff3cd;padding:12px;border-radius:8px}}
    </style></head><body><main><h1>Soccer Value Research</h1>
    <p>Generated {payload['generated_at_utc']}. Data: football-data.co.uk.</p>
    <div class="warning"><b>Research disclaimer:</b> These are estimates, not proven winning tips. Free source coverage is limited to selected leagues, and many markets will lack odds. Backtest ROI is historical and can be affected by overfitting, stale odds and data gaps. Never stake money you cannot afford to lose.</div>
    <h2>Chronological holdout backtest</h2>{bt_html}
    <h2>Fixtures and estimates</h2>{''.join(rows) if rows else '<p>No supported fixtures found for the selected date.</p>'}
    </main></body></html>"""
    report_path.write_text(html, encoding="utf-8")
    return json_path, csv_path, report_path


def main():
    global DEFAULT_SEASONS
    parser = argparse.ArgumentParser(description="Soccer value-bet analysis using free historical data.")
    parser.add_argument("--date", default="today", help="today, tomorrow, or YYYY-MM-DD (Nairobi date)")
    parser.add_argument("--outdir", default="output", help="Directory for HTML/CSV/JSON outputs")
    parser.add_argument("--min-edge", type=float, default=0.04, help="Minimum probability edge vs raw odds, e.g. 0.04")
    parser.add_argument("--min-ev", type=float, default=0.03, help="Minimum expected value, e.g. 0.03")
    parser.add_argument("--leagues", default="all", help="all or comma-separated football-data league codes, e.g. E0,SP1,I1")
    parser.add_argument("--backtest-only", action="store_true", help="Run chronological backtest without upcoming fixtures")
    parser.add_argument("--seasons", type=int, default=DEFAULT_SEASONS, help="Number of seasons to download (default 5; max 8)")
    args = parser.parse_args()
    DEFAULT_SEASONS = max(2, min(8, args.seasons))
    target_text = args.date.strip().lower()
    today = datetime.now().date()
    if target_text == "today":
        target = today
    elif target_text == "tomorrow":
        target = today + timedelta(days=1)
    else:
        try:
            target = datetime.strptime(target_text, "%Y-%m-%d").date()
        except ValueError:
            parser.error("--date must be today, tomorrow or YYYY-MM-DD")
    selected = list(DIV_INFO) if args.leagues.lower() == "all" else [x.strip() for x in args.leagues.split(",")]
    selected = [x for x in selected if x in DIV_INFO]
    if not selected:
        parser.error("No valid league codes supplied. Example: --leagues E0,SP1,I1")
    print(f"Downloading up to {DEFAULT_SEASONS} seasons of results for {len(selected)} supported leagues...")
    matches = load_historical_matches(selected, target)
    if len(matches) < 200:
        raise SystemExit(f"Only {len(matches)} historical matches loaded. Check internet or choose more leagues/seasons.")
    X, y, dates, raw, _ = build_training_data(matches)
    print(f"Historical matches loaded: {len(matches)}")
    backtest = walk_forward_backtest(X, y, dates, raw)
    print("Chronological backtest:", json.dumps(backtest, indent=2))
    if args.backtest_only:
        predictions = []
    else:
        print("Downloading upcoming fixtures...")
        fixtures = [f for f in load_fixtures() if f["date"] == target and f["div"] in selected]
        if not fixtures and args.date in ("today", "tomorrow"):
            later = sorted({f["date"] for f in load_fixtures() if f["date"] >= target and f["div"] in selected})
            if later:
                target = later[0]
                fixtures = [f for f in load_fixtures() if f["date"] == target and f["div"] in selected]
                print(f"No fixtures listed for requested date; showing next listed date: {target}")
        models = fit_ensemble(X, y)
        teams_by_div, league_rows = recent_state_for_future(matches)
        predictions = []
        for fixture in sorted(fixtures, key=lambda f: (DIV_ORDER.get(f["div"], 999), f["kickoff"])):
            try:
                predictions.append(prediction_for_fixture(
                    fixture, models, teams_by_div, league_rows, args.min_edge, args.min_ev))
            except Exception as exc:
                print(f"Skipping {fixture['home']} vs {fixture['away']}: {exc}", file=sys.stderr)
    paths = write_outputs(predictions, backtest, args.outdir)
    print(f"Wrote: {paths[0]}, {paths[1]}, {paths[2]}")
    print("Tip: open output/report.html in a browser. No qualifying bet means the model found no candidate, not that you should force a bet.")


if __name__ == "__main__":
    main()
