"""Pre-match features for every match (Phase 2).

All features of a match are computed ONLY from matches that come before it. The matches are
processed in date order (same-day matches in file order): first the match reads its features
from the current team statistics, then its result is added to them. So the result of a match
can never leak into its own features or into the features of an earlier match.

Feature groups
--------------
Elo rating        team strength, World Football Elo method (eloratings.net)
Recent form       goals for / against and points per match over the last FORM_WINDOW matches
Head-to-head      mean goal difference in the last H2H_WINDOW meetings of the two teams
Venue             neutral ground (0/1) and venue altitude
Match importance  one-hot of the tournament group (Friendly is the reference group)
Year              long-term trend in scoring
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_HOME_ADVANTAGE = 100.0
# K = weight of a match in the Elo update, by tournament group (eloratings.net levels;
# Nations League is counted with the qualifiers - our choice)
ELO_K = {"World Cup": 60, "Continental final": 50, "Qualifier / Nations League": 40,
         "Other tournament": 30, "Friendly": 20}
FORM_WINDOW = 10
H2H_WINDOW = 5

# Approximate city elevations (m) of venues at 1,500 m or higher that occur in the data.
# Keyed by (city, country) because some names exist twice (Cuenca, Guadalajara are also in Spain).
# Values are rounded city elevations (about +-50-100 m); every other venue counts as 0.
VENUE_ALTITUDE_M = {
    ("El Alto", "Bolivia"): 4090, ("Oruro", "Bolivia"): 3735, ("La Paz", "Bolivia"): 3640,
    ("Cusco", "Peru"): 3400, ("Quito", "Ecuador"): 2850, ("Sucre", "Bolivia"): 2810,
    ("Latacunga", "Ecuador"): 2750, ("Toluca", "Mexico"): 2660, ("Bogotá", "Colombia"): 2640,
    ("Ambato", "Ecuador"): 2580, ("Cochabamba", "Bolivia"): 2560, ("Cuenca", "Ecuador"): 2560,
    ("Addis Ababa", "Ethiopia"): 2355, ("Arequipa", "Peru"): 2335, ("Thimphu", "Bhutan"): 2330,
    ("Asmara", "Eritrea"): 2325, ("Sana'a", "Yemen"): 2250, ("Mexico City", "Mexico"): 2240,
    ("Manizales", "Colombia"): 2150, ("Puebla", "Mexico"): 2135, ("Tarija", "Bolivia"): 1850,
    ("Nairobi", "Kenya"): 1795, ("Kabul", "Afghanistan"): 1790, ("Johannesburg", "South Africa"): 1753,
    ("Kigali", "Rwanda"): 1567, ("Guadalajara", "Mexico"): 1566,
}

IMPORTANCE_DUMMIES = {"World Cup": "is_world_cup", "Continental final": "is_continental_final",
                      "Qualifier / Nations League": "is_qualifier", "Other tournament": "is_other_tournament"}

# The feature columns used by the models (all numeric). Note: no elo_diff - it would be an exact
# linear combination of elo_home and elo_away, which makes the normal equation singular.
FEATURES = (["elo_home", "elo_away",
             "form_gf_home", "form_ga_home", "form_pts_home",
             "form_gf_away", "form_ga_away", "form_pts_away",
             "h2h_gd", "h2h_n", "neutral", "altitude_km"]
            + list(IMPORTANCE_DUMMIES.values()) + ["year"])


def goal_multiplier(margin):
    """Elo G factor: 1 for a margin of 0-1 goals, 1.5 for 2 goals, (11 + N) / 8 for N >= 3."""
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def expected_home_result(elo_home, elo_away, neutral):
    """Elo expectation for the home team (1 = sure win, 0.5 = even), incl. home advantage."""
    dr = elo_home - elo_away + (0.0 if neutral else ELO_HOME_ADVANTAGE)
    return 1.0 / (10 ** (-dr / 400) + 1)


def elo_update(elo_home, elo_away, home_score, away_score, neutral, k):
    """Return the new (elo_home, elo_away) after one match. The update is zero-sum."""
    result = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
    change = k * goal_multiplier(home_score - away_score) * (result - expected_home_result(elo_home, elo_away, neutral))
    return elo_home + change, elo_away - change


def build_features(df):
    """Pre-match features for every row of the cleaned, date-sorted match table `df`.

    Returns a DataFrame with the same index as `df` and the columns in FEATURES
    (plus elo_diff and n_prev_home / n_prev_away, which are kept for analysis only).
    """
    if not df["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")

    elo = defaultdict(lambda: ELO_START)
    recent = defaultdict(lambda: deque(maxlen=FORM_WINDOW))      # team -> (goals for, goals against, points)
    meetings = defaultdict(lambda: deque(maxlen=H2H_WINDOW))     # (team_a, team_b) sorted -> goal difference seen by team_a
    rows = []

    for m in df[["home_team", "away_team", "home_score", "away_score", "neutral", "tournament_group"]].itertuples(index=False):
        home, away = m.home_team, m.away_team
        pair = tuple(sorted((home, away)))
        sign = 1 if pair[0] == home else -1                      # converts team_a's view to the home view

        # 1) read the features - only matches before this one have been added so far
        feats = {"elo_home": elo[home], "elo_away": elo[away]}
        for side, team in (("home", home), ("away", away)):
            hist = recent[team]
            feats[f"n_prev_{side}"] = len(hist)
            if hist:
                gf, ga, pts = np.mean(hist, axis=0)
            else:                                                 # first ever match: no form yet
                gf = ga = pts = np.nan
            feats[f"form_gf_{side}"], feats[f"form_ga_{side}"], feats[f"form_pts_{side}"] = gf, ga, pts
        past = meetings[pair]
        feats["h2h_n"] = len(past)
        feats["h2h_gd"] = sign * float(np.mean(past)) if past else 0.0
        rows.append(feats)

        # 2) add this match's result to the statistics (used by LATER matches only)
        hs, as_ = int(m.home_score), int(m.away_score)
        elo[home], elo[away] = elo_update(elo[home], elo[away], hs, as_, bool(m.neutral), ELO_K[m.tournament_group])
        home_pts, away_pts = (3, 0) if hs > as_ else (1, 1) if hs == as_ else (0, 3)
        recent[home].append((hs, as_, home_pts))
        recent[away].append((as_, hs, away_pts))
        meetings[pair].append(sign * (hs - as_))

    feat = pd.DataFrame(rows, index=df.index)
    feat["elo_diff"] = feat["elo_home"] - feat["elo_away"]
    feat["neutral"] = df["neutral"].astype(int)
    altitude_m = pd.Series([VENUE_ALTITUDE_M.get(key, 0) for key in zip(df["city"], df["country"])], index=df.index)
    feat["altitude_km"] = altitude_m / 1000
    for group, column in IMPORTANCE_DUMMIES.items():
        feat[column] = (df["tournament_group"] == group).astype(int)
    feat["year"] = df["date"].dt.year
    return feat[FEATURES + ["elo_diff", "n_prev_home", "n_prev_away"]]
