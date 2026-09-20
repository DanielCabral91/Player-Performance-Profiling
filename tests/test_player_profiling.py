import numpy as np
import pandas as pd
import pytest

from python.generate_synthetic_profiles import (
    ALL_ATTRIBUTES,
    generate_synthetic_player_profiles,
)
from python.player_profiling_analysis import (
    DEFAULT_FRAMEWORK,
    DEFAULT_INPUT,
    EVALUATION_PERIODS,
    add_domain_medians,
    add_observation_coverage,
    build_player_summary,
    latest_available_rows,
    load_observation_framework,
    load_profile_data,
    validate_profile_data,
)


def _valid_demo():
    return validate_profile_data(load_profile_data(DEFAULT_INPUT))


def test_generator_same_seed_is_reproducible():
    a = generate_synthetic_player_profiles(seed=42)
    b = generate_synthetic_player_profiles(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_generator_different_seed_changes_data():
    a = generate_synthetic_player_profiles(seed=42)
    b = generate_synthetic_player_profiles(seed=43)
    assert not a.equals(b)


def test_public_demo_is_synthetic():
    df = _valid_demo()
    assert df["data_source"].eq("synthetic").all()
    assert df["player_id"].nunique() == 24


def test_generator_has_no_forced_complete_period_requirement():
    raw = load_profile_data(DEFAULT_INPUT)
    counts = raw.groupby("player_id")["evaluation_period"].nunique()
    assert counts.between(2, 3).all()


def test_na_ratings_are_accepted():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw.loc[raw.index[0], ALL_ATTRIBUTES[0]] = np.nan
    validated = validate_profile_data(raw)
    assert pd.isna(
        validated.loc[
            validated["player_id"].eq(raw.loc[0, "player_id"]),
            ALL_ATTRIBUTES[0],
        ]
    ).any()


def test_non_numeric_rating_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw[ALL_ATTRIBUTES[0]] = raw[ALL_ATTRIBUTES[0]].astype(object)
    raw.loc[raw.index[0], ALL_ATTRIBUTES[0]] = "bad"
    with pytest.raises(ValueError, match="invalid non-numeric"):
        validate_profile_data(raw)


def test_non_integer_rating_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw[ALL_ATTRIBUTES[0]] = raw[ALL_ATTRIBUTES[0]].astype(object)
    raw.loc[raw.index[0], ALL_ATTRIBUTES[0]] = 3.5
    with pytest.raises(ValueError, match="integer ordinal"):
        validate_profile_data(raw)


def test_out_of_range_rating_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw.loc[raw.index[0], ALL_ATTRIBUTES[0]] = 6
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_profile_data(raw)


def test_zero_observed_sessions_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw.loc[raw.index[0], "observed_sessions"] = 0
    with pytest.raises(ValueError, match="observed_sessions must be > 0"):
        validate_profile_data(raw)


def test_non_integer_observed_sessions_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw["observed_sessions"] = raw["observed_sessions"].astype(float)
    raw.loc[raw.index[0], "observed_sessions"] = 2.5
    with pytest.raises(ValueError, match="observed_sessions must contain integer"):
        validate_profile_data(raw)


def test_invalid_observation_context_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw.loc[raw.index[0], "observation_context"] = "Unknown"
    with pytest.raises(ValueError, match="Invalid observation_context"):
        validate_profile_data(raw)


def test_position_change_is_accepted():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    player = raw["player_id"].iloc[0]
    idx = raw.index[raw["player_id"].eq(player)][-1]
    current = raw.loc[idx, "position_group"]
    raw.loc[idx, "position_group"] = (
        "Forward" if current != "Forward" else "Defender"
    )
    validated = validate_profile_data(raw)
    assert validated[
        validated["player_id"].eq(player)
    ]["position_group"].nunique() >= 1


def test_missing_period_is_accepted():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    player = raw["player_id"].iloc[0]
    indices = raw.index[raw["player_id"].eq(player)]
    if len(indices) == 3:
        raw = raw.drop(indices[1])
    validated = validate_profile_data(raw)
    assert player in set(validated["player_id"])


def test_duplicate_player_period_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate player_id"):
        validate_profile_data(raw)


def test_all_missing_attributes_in_row_is_rejected():
    raw = load_profile_data(DEFAULT_INPUT).copy()
    raw.loc[raw.index[0], list(ALL_ATTRIBUTES)] = np.nan
    with pytest.raises(ValueError, match="at least one observed attribute"):
        validate_profile_data(raw)


def test_observation_coverage_exact():
    df = _valid_demo().head(1).copy()
    df.loc[df.index[0], list(ALL_ATTRIBUTES)] = np.nan
    df.loc[df.index[0], ALL_ATTRIBUTES[:5]] = [1, 2, 3, 4, 5]
    out = add_observation_coverage(df)
    assert out.iloc[0]["observed_attribute_count"] == 5
    assert out.iloc[0]["observation_coverage_pct"] == pytest.approx(29.4, abs=0.1)


def test_domain_median_ignores_na_when_coverage_sufficient():
    df = _valid_demo().head(1).copy()
    technical = [
        "technical_first_touch",
        "technical_passing",
        "technical_receiving_orientation",
        "technical_ball_control",
        "technical_dribbling",
    ]
    df.loc[df.index[0], technical] = [1, np.nan, 3, np.nan, 5]
    out = add_domain_medians(df)
    assert out.iloc[0]["technical_median"] == 3.0


def test_domain_median_becomes_na_when_coverage_insufficient():
    df = _valid_demo().head(1).copy()
    technical = [
        "technical_first_touch",
        "technical_passing",
        "technical_receiving_orientation",
        "technical_ball_control",
        "technical_dribbling",
    ]
    df.loc[df.index[0], technical] = [1, np.nan, np.nan, np.nan, 5]
    out = add_domain_medians(df)
    assert pd.isna(out.iloc[0]["technical_median"])


def test_latest_available_period_is_selected_per_player():
    df = _valid_demo()
    latest = latest_available_rows(df)
    assert latest["player_id"].nunique() == df["player_id"].nunique()
    for player_id, player_rows in df.groupby("player_id"):
        expected = max(
            player_rows["evaluation_period"],
            key=lambda x: EVALUATION_PERIODS.index(str(x)),
        )
        actual = latest.loc[
            latest["player_id"].eq(player_id),
            "evaluation_period",
        ].iloc[0]
        assert str(actual) == str(expected)


def test_summary_has_one_row_per_player():
    summary = build_player_summary(_valid_demo())
    assert len(summary) == 24
    assert summary["player_id"].nunique() == 24


def test_summary_has_no_overall_score_or_rank():
    summary = build_player_summary(_valid_demo())
    forbidden = {
        "overall_score",
        "talent_score",
        "potential_score",
        "player_rank",
        "selection_probability",
    }
    assert forbidden.isdisjoint(summary.columns)


def test_leave_one_out_peer_median_excludes_focal_player():
    rows = []
    for player_id, rating in [
        ("Player_01", 1),
        ("Player_02", 4),
        ("Player_03", 5),
    ]:
        row = {
            "player_id": player_id,
            "evaluation_period": "P3",
            "position_group": "Defender",
            "observation_context": "Mixed",
            "observed_sessions": 5,
            "data_source": "synthetic",
        }
        for attr in ALL_ATTRIBUTES:
            row[attr] = rating
        rows.append(row)

    df = validate_profile_data(pd.DataFrame(rows))
    summary = build_player_summary(df)
    p1 = summary[summary["player_id"].eq("Player_01")].iloc[0]
    assert p1[f"{ALL_ATTRIBUTES[0]}_peer_median"] == 4.5
    assert p1["peer_group_n_excluding_player"] == 2


def test_example_player_fallback_handles_insufficient_preferred_coverage():
    from python.player_profiling_analysis import (
        add_domain_medians,
        add_observation_coverage,
        choose_example_player,
        latest_available_rows,
    )

    validated = _valid_demo()
    enriched = add_domain_medians(add_observation_coverage(validated))
    latest = latest_available_rows(enriched)
    chosen = choose_example_player(latest, "Player_01")

    domain_cols = [
        "technical_median",
        "tactical_median",
        "physical_observation_median",
        "behavioural_observation_median",
    ]
    row = latest[latest["player_id"].eq(chosen)].iloc[0]
    assert row[domain_cols].notna().all()


def test_example_player_fallback_is_deterministic():
    from python.player_profiling_analysis import (
        add_domain_medians,
        add_observation_coverage,
        choose_example_player,
        latest_available_rows,
    )

    validated = _valid_demo()
    enriched = add_domain_medians(add_observation_coverage(validated))
    latest = latest_available_rows(enriched)
    a = choose_example_player(latest, "Nonexistent_Player")
    b = choose_example_player(latest, "Nonexistent_Player")
    assert a == b


def test_peer_reference_matches_focal_evaluation_period():
    rows = []
    fixtures = [
        ("Player_01", "P3", 3),
        ("Player_02", "P3", 5),
        ("Player_03", "P2", 1),
    ]
    for player_id, period, rating in fixtures:
        row = {
            "player_id": player_id,
            "evaluation_period": period,
            "position_group": "Defender",
            "observation_context": "Mixed",
            "observed_sessions": 5,
            "data_source": "synthetic",
        }
        for attr in ALL_ATTRIBUTES:
            row[attr] = rating
        rows.append(row)

    df = validate_profile_data(pd.DataFrame(rows))
    summary = build_player_summary(df)
    focal = summary[summary["player_id"].eq("Player_01")].iloc[0]

    assert focal[f"{ALL_ATTRIBUTES[0]}_peer_median"] == 5.0
    assert focal["peer_group_n_excluding_player"] == 1


def test_peer_reference_matches_current_position_not_historical_position():
    rows = []
    fixtures = [
        ("Player_01", "P2", "Defender", 2),
        ("Player_01", "P3", "Midfielder", 3),
        ("Player_02", "P3", "Midfielder", 5),
        ("Player_03", "P3", "Defender", 1),
    ]
    for player_id, period, position, rating in fixtures:
        row = {
            "player_id": player_id,
            "evaluation_period": period,
            "position_group": position,
            "observation_context": "Mixed",
            "observed_sessions": 5,
            "data_source": "synthetic",
        }
        for attr in ALL_ATTRIBUTES:
            row[attr] = rating
        rows.append(row)

    df = validate_profile_data(pd.DataFrame(rows))
    summary = build_player_summary(df)
    focal = summary[summary["player_id"].eq("Player_01")].iloc[0]

    assert focal["position_group"] == "Midfielder"
    assert focal[f"{ALL_ATTRIBUTES[0]}_peer_median"] == 5.0


def test_framework_has_all_attributes_once():
    framework = load_observation_framework(DEFAULT_FRAMEWORK)
    assert len(framework) == len(ALL_ATTRIBUTES)
    assert framework["attribute"].nunique() == len(ALL_ATTRIBUTES)


def test_framework_has_behavioural_anchors_not_psychological_scores():
    framework = load_observation_framework(DEFAULT_FRAMEWORK)
    combined = " ".join(
        framework.astype(str).fillna("").values.flatten()
    ).lower()
    assert "diagnosis" not in combined
    assert "psychological score" not in combined


def test_no_forbidden_identity_columns_in_public_data():
    raw = load_profile_data(DEFAULT_INPUT)
    forbidden = {
        "name",
        "player_name",
        "date_of_birth",
        "dob",
        "shirt_number",
        "injury",
        "diagnosis",
        "medical_note",
        "coach_comment",
    }
    assert forbidden.isdisjoint(raw.columns)
