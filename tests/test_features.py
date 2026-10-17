"""Correctness and leakage tests for the pre-match features (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest

from src.data import RAW_DIR, load_clean
from src.features import ELO_START, FEATURES, build_features, elo_update, expected_home_result, goal_multiplier


def toy_matches():
    """Four small matches between three teams, already in date order."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "home_team": ["A", "B", "A", "C"],
        "away_team": ["B", "A", "C", "A"],
        "home_score": [2, 1, 0, 3],
        "away_score": [0, 1, 1, 3],
        "neutral": [False, False, True, False],
        "tournament_group": ["Friendly", "Qualifier / Nations League", "World Cup", "Friendly"],
        "city": ["X", "Y", "Z", "La Paz"],
        "country": ["A", "B", "Q", "Bolivia"],
    })


# ---------- Elo formula ----------

def test_goal_multiplier():
    assert [goal_multiplier(m) for m in (0, 1, -1, 2, 3, 5)] == [1.0, 1.0, 1.0, 1.5, 14 / 8, 16 / 8]


def test_elo_update_by_hand():
    # Equal teams, home advantage +100 -> expected home result 1 / (10^-0.25 + 1) = 0.6401
    assert expected_home_result(1500, 1500, neutral=False) == pytest.approx(0.6401, abs=1e-4)
    assert expected_home_result(1500, 1500, neutral=True) == pytest.approx(0.5)
    # Friendly (K = 20), home wins 1-0: change = 20 * 1 * (1 - 0.6401) = 7.198
    new_home, new_away = elo_update(1500, 1500, 1, 0, neutral=False, k=20)
    assert new_home == pytest.approx(1507.198, abs=1e-3)
    assert new_away == pytest.approx(1492.802, abs=1e-3)


def test_elo_update_is_zero_sum():
    h, a = elo_update(1650, 1420, 0, 3, neutral=True, k=40)
    assert h + a == pytest.approx(1650 + 1420)


# ---------- the feature table ----------

def test_first_match_uses_start_values():
    feat = build_features(toy_matches())
    first = feat.iloc[0]
    assert first["elo_home"] == ELO_START and first["elo_away"] == ELO_START
    assert np.isnan(first["form_gf_home"]) and first["n_prev_home"] == 0
    assert first["h2h_n"] == 0 and first["h2h_gd"] == 0


def test_form_and_head_to_head_from_the_home_view():
    feat = build_features(toy_matches())
    second = feat.iloc[1]                      # B (home) vs A, after A beat B 2-0
    assert second["form_gf_home"] == 0 and second["form_ga_home"] == 2 and second["form_pts_home"] == 0
    assert second["form_gf_away"] == 2 and second["form_pts_away"] == 3
    assert second["h2h_n"] == 1 and second["h2h_gd"] == -2      # B lost the previous meeting by 2


def test_venue_and_importance_columns():
    feat = build_features(toy_matches())
    assert list(feat["neutral"]) == [0, 0, 1, 0]
    assert feat["altitude_km"].iloc[3] == pytest.approx(3.64)   # La Paz, Bolivia
    assert feat["altitude_km"].iloc[0] == 0
    assert list(feat["is_world_cup"]) == [0, 0, 1, 0]
    assert list(feat["is_qualifier"]) == [0, 1, 0, 0]


def test_no_leakage_from_the_match_itself_or_later_matches():
    """Changing the score of match i must not change the features of matches 0..i."""
    base = toy_matches()
    changed = base.copy()
    changed.loc[1, ["home_score", "away_score"]] = [7, 0]
    f_base, f_changed = build_features(base), build_features(changed)
    pd.testing.assert_frame_equal(f_base.iloc[:2], f_changed.iloc[:2])
    assert not f_base.iloc[2:].equals(f_changed.iloc[2:])          # later matches do see it


def test_unsorted_input_is_rejected():
    with pytest.raises(ValueError):
        build_features(toy_matches().iloc[::-1])


# ---------- the real data ----------

@pytest.mark.skipif(not (RAW_DIR / "results.csv").exists(), reason="download the dataset first (see README)")
def test_real_features_have_no_leakage_signature():
    df, _ = load_clean()
    feat = build_features(df)
    assert list(feat.columns[:len(FEATURES)]) == FEATURES
    train = df["period"] == "train"
    # no single feature should almost perfectly predict a target (that would be a leakage signature)
    corr = feat.loc[train, FEATURES].corrwith(df.loc[train, "home_score"]).abs()
    assert corr.max() < 0.8
    # sanity: the team with the higher pre-match Elo should win clearly more often
    t = df[train]
    home_win = t["home_score"] > t["away_score"]
    stronger_home = feat.loc[train, "elo_diff"] > 100
    weaker_home = feat.loc[train, "elo_diff"] < -100
    assert home_win[stronger_home].mean() > home_win[weaker_home].mean() + 0.3
