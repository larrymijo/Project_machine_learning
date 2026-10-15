"""Checks on the cleaned match table (run with: pytest)."""
import pandas as pd
import pytest

from src.data import CONMEBOL, TEST_END, TEST_START, TRAIN_START, RAW_DIR, load_clean, tournament_group

pytestmark = pytest.mark.skipif(not (RAW_DIR / "results.csv").exists(),
                                reason="download the dataset into data/raw first (see README)")


@pytest.fixture(scope="module")
def clean():
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
