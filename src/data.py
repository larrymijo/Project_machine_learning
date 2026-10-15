"""Loading and cleaning of the international football results (Mart Jürisoo dataset).

The cleaned table keeps the WHOLE match history: matches before 1994 are not used for
training, but Phase 2 needs them to "warm up" the Elo ratings. Every match gets a `period`
label that decides what it is used for:

    warm-up  date <  1994-01-01                 only used to build features
    train    1994-01-01 <= date < 2023-09-01    training + time-series cross-validation
    test     2023-09-01 <= date <= 2025-09-09   final evaluation (the 2026 qualifying cycle)
    after    date >  2025-09-09                 optional bonus check (incl. the 2026 World Cup)
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RANDOM_STATE = 42
TARGETS = ["home_score", "away_score"]

CONMEBOL = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
            "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela"]
WORLD_CUP = "FIFA World Cup"
WCQ = "FIFA World Cup qualification"

TRAIN_START = pd.Timestamp("1994-01-01")
TEST_START = pd.Timestamp("2023-09-01")   # the 2026 CONMEBOL qualifiers started on 7 Sep 2023
TEST_END = pd.Timestamp("2025-09-09")     # last round of the 2026 CONMEBOL qualifiers

# Continental championship finals (the "50" level of the World Football Elo ratings)
CONTINENTAL_FINALS = ["Copa América", "UEFA Euro", "African Cup of Nations", "AFC Asian Cup",
                      "Gold Cup", "CONCACAF Championship", "Oceania Nations Cup", "Confederations Cup"]
TOURNAMENT_GROUPS = ["World Cup", "Continental final", "Qualifier / Nations League",
                     "Other tournament", "Friendly"]


def load_results(raw_dir=RAW_DIR):
    """results.csv as a DataFrame, sorted by date (stable, so same-day matches keep file order)."""
    df = pd.read_csv(Path(raw_dir) / "results.csv", parse_dates=["date"])
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_former_names(raw_dir=RAW_DIR):
    return pd.read_csv(Path(raw_dir) / "former_names.csv", parse_dates=["start_date", "end_date"])


def apply_former_names(df, former):
    """Rename former team names to the current name, so a team keeps one identity over time.

    Returns (df, n_renamed). With the 13 Oct 2026 download no former name occurs in
    results.csv, so n_renamed is 0 - the function is kept so the check stays visible.
    """
    mapping = dict(zip(former["former"], former["current"]))
    renamed = df["home_team"].isin(mapping) | df["away_team"].isin(mapping)
    df = df.assign(home_team=df["home_team"].replace(mapping),
                   away_team=df["away_team"].replace(mapping))
    return df, int(renamed.sum())


def fifa_teams(df):
    """Teams that have played at least one World Cup or World Cup qualifier.

    Used as a data-driven definition of "FIFA member national team": it excludes regional and
    unrecognised teams (Catalonia, Corsica, Tibet, ...) from tournaments such as the Island
    Games or the CONIFA World Football Cup.
    """
    official = df["tournament"].isin([WORLD_CUP, WCQ])
    return set(df.loc[official, "home_team"]) | set(df.loc[official, "away_team"])


def tournament_group(tournament):
    """Map the 225 tournament names onto 5 groups of match importance."""
    t = pd.Series(tournament)
    group = pd.Series("Other tournament", index=t.index)
    group[t.str.contains("qualification") | t.str.contains("Nations League")] = "Qualifier / Nations League"
    group[t.isin(CONTINENTAL_FINALS)] = "Continental final"
    group[t == WORLD_CUP] = "World Cup"
    group[t == "Friendly"] = "Friendly"
    return pd.Categorical(group, categories=TOURNAMENT_GROUPS)


def assign_period(dates):
    period = pd.Series("after", index=dates.index)
    period[dates <= TEST_END] = "test"
    period[dates < TEST_START] = "train"
    period[dates < TRAIN_START] = "warm-up"
    return pd.Categorical(period, categories=["warm-up", "train", "test", "after"])


def clean_results(df, former=None):
    """Apply the documented cleaning. Returns (clean_df, notes)."""
    notes = [f"Loaded {len(df):,} matches, {df['date'].min().date()} -> {df['date'].max().date()}"]
    notes.append(f"Missing values: {int(df.isna().sum().sum())} | fully duplicated rows: {int(df.duplicated().sum())}")

    if former is not None:
        df, n_renamed = apply_former_names(df, former)
        notes.append(f"Matches with a former team name renamed: {n_renamed}")

    fifa = fifa_teams(df)
    all_teams = set(df["home_team"]) | set(df["away_team"])
    keep = df["home_team"].isin(fifa) & df["away_team"].isin(fifa)
    notes.append(f"Kept the {len(fifa)} teams that have played a World Cup (qualifier); "
                 f"removed {len(all_teams - fifa)} regional/unrecognised teams -> "
                 f"dropped {int((~keep).sum()):,} matches")
    df = df[keep].copy()

    df["tournament_group"] = tournament_group(df["tournament"])
    df["is_conmebol_wcq"] = ((df["tournament"] == WCQ)
                             & df["home_team"].isin(CONMEBOL) & df["away_team"].isin(CONMEBOL))
    df["period"] = assign_period(df["date"])
    df = df.reset_index(drop=True)

    counts = df["period"].value_counts().reindex(["warm-up", "train", "test", "after"])
    notes.append("Matches per period: " + ", ".join(f"{p} {n:,}" for p, n in counts.items()))
    notes.append(f"CONMEBOL World Cup qualifiers in the test period: "
                 f"{int((df['is_conmebol_wcq'] & (df['period'] == 'test')).sum())}")
    return df, notes


def load_clean(raw_dir=RAW_DIR):
    """Convenience: load results.csv + former_names.csv and clean them."""
    return clean_results(load_results(raw_dir), load_former_names(raw_dir))
