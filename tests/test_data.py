"""Checks on the cleaned match table (run with: pytest)."""
import pandas as pd
import pytest

from src.data import (CONMEBOL, TEST_END, TEST_START, TRAIN_START, RAW_DIR, apply_former_names,
                      fifa_teams, load_clean, tournament_group)


def test_former_names_respect_the_date_window():
    df = pd.DataFrame({
        "date": pd.to_datetime(["1950-06-01", "2000-06-01"]),
        "home_team": ["Ireland", "Ireland"],
        "away_team": ["England", "Éire"],
    })
    former = pd.DataFrame({
        "current": ["Northern Ireland", "Republic of Ireland"],
        "former": ["Ireland", "Éire"],
        "start_date": pd.to_datetime(["1882-02-18", "1937-01-01"]),
        "end_date": pd.to_datetime(["1956-04-11", "1953-03-25"]),
    })
    out, n_renamed = apply_former_names(df, former)
    assert n_renamed == 1
    assert out["home_team"].tolist() == ["Northern Ireland", "Ireland"]
    assert out["away_team"].tolist() == ["England", "Éire"]   # Éire is outside its window
    assert df["home_team"].tolist() == ["Ireland", "Ireland"]  # the input is not modified


def test_fifa_filter_ignores_official_matches_after_the_cutoff():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2024-06-01"]),
        "home_team": ["Catalonia", "Catalonia"],
        "away_team": ["Corsica", "Spain"],
        "tournament": ["Friendly", "FIFA World Cup qualification"],
    })
    assert fifa_teams(df, before=TEST_START) == set()
    assert fifa_teams(df) == {"Catalonia", "Spain"}


@pytest.fixture(scope="module")
def clean():
    if not (RAW_DIR / "results.csv").exists():
        pytest.skip("download the dataset into data/raw first (see README)")
    df, _ = load_clean()
    return df


def test_no_missing_values_and_sorted(clean):
    assert clean[["date", "home_team", "away_team", "home_score", "away_score"]].notna().all().all()
    assert clean["date"].is_monotonic_increasing


def test_scores_are_non_negative_integers(clean):
    for col in ["home_score", "away_score"]:
        assert pd.api.types.is_integer_dtype(clean[col])
        assert (clean[col] >= 0).all()


def test_periods_follow_the_dates(clean):
    assert (clean.loc[clean["period"] == "train", "date"] >= TRAIN_START).all()
    assert (clean.loc[clean["period"] == "train", "date"] < TEST_START).all()
    test = clean.loc[clean["period"] == "test", "date"]
    assert test.min() >= TEST_START and test.max() <= TEST_END


def test_the_2026_conmebol_cycle_is_complete(clean):
    cycle = clean[clean["is_conmebol_wcq"] & (clean["period"] == "test")]
    assert len(cycle) == 90                       # 10 teams, double round robin
    assert set(cycle["home_team"]) == set(CONMEBOL)
    assert (cycle.groupby("home_team").size() == 9).all()


def test_only_fifa_teams_remain(clean):
    for name in ["Catalonia", "Tibet", "Corsica"]:
        assert not ((clean["home_team"] == name) | (clean["away_team"] == name)).any()


def test_tournament_groups():
    groups = tournament_group(pd.Series(["FIFA World Cup", "Copa América", "FIFA World Cup qualification",
                                         "UEFA Nations League", "Friendly", "King's Cup"]))
    assert list(groups) == ["World Cup", "Continental final", "Qualifier / Nations League",
                            "Qualifier / Nations League", "Friendly", "Other tournament"]
