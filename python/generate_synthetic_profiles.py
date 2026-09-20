"""Generate a deterministic, fully synthetic player-observation dataset.

Project 03 — Youth Football Player Performance Profiling (V0.2)

The synthetic generator demonstrates data structure and pipeline behaviour only.
It does not reproduce real players, real coach ratings, real positions, or real
development trajectories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_PLAYERS = 24
EVALUATION_PERIODS = ("P1", "P2", "P3")
POSITION_GROUPS = ("Defender", "Midfielder", "Forward")
OBSERVATION_CONTEXTS = ("Training", "Match", "Mixed")

ATTRIBUTE_GROUPS = {
    "technical": (
        "technical_first_touch",
        "technical_passing",
        "technical_receiving_orientation",
        "technical_ball_control",
        "technical_dribbling",
    ),
    "tactical": (
        "tactical_positioning",
        "tactical_scanning",
        "tactical_decision_making",
        "tactical_support",
        "tactical_transition_response",
    ),
    "physical_observation": (
        "physical_acceleration",
        "physical_change_of_direction",
        "physical_repeat_effort",
    ),
    "behavioural_observation": (
        "behaviour_communication",
        "behaviour_concentration",
        "behaviour_training_engagement",
        "behaviour_response_to_feedback",
    ),
}

ALL_ATTRIBUTES = tuple(
    attribute
    for attributes in ATTRIBUTE_GROUPS.values()
    for attribute in attributes
)


def _initial_positions(players: list[str]) -> dict[str, str]:
    groups = list(POSITION_GROUPS)
    return {
        player: groups[index % len(groups)]
        for index, player in enumerate(players)
    }


def generate_synthetic_player_profiles(
    n_players: int = N_PLAYERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create deterministic synthetic structured observations."""
    if n_players < 6:
        raise ValueError("n_players must be >= 6.")

    rng = np.random.default_rng(seed)
    players = [f"Player_{i:02d}" for i in range(1, n_players + 1)]
    initial_positions = _initial_positions(players)

    player_baseline = {player: rng.normal(3.0, 0.42) for player in players}
    domain_bias = {
        player: {
            domain: rng.normal(0.0, 0.30)
            for domain in ATTRIBUTE_GROUPS
        }
        for player in players
    }
    period_domain_shift = {
        player: {
            period: {
                domain: (0.0 if period == "P1" else rng.normal(0.0, 0.28))
                for domain in ATTRIBUTE_GROUPS
            }
            for period in EVALUATION_PERIODS
        }
        for player in players
    }

    rows: list[dict] = []

    for player in players:
        position = initial_positions[player]
        included_periods = list(EVALUATION_PERIODS)

        if rng.random() < 0.25:
            removable = rng.choice(["P1", "P2"])
            included_periods.remove(removable)

        for period in included_periods:
            if period != "P1" and rng.random() < 0.12:
                alternatives = [p for p in POSITION_GROUPS if p != position]
                position = str(rng.choice(alternatives))

            observed_sessions = int(rng.integers(2, 9))
            observation_context = str(
                rng.choice(OBSERVATION_CONTEXTS, p=[0.30, 0.20, 0.50])
            )

            row = {
                "player_id": player,
                "evaluation_period": period,
                "position_group": position,
                "observation_context": observation_context,
                "observed_sessions": observed_sessions,
                "data_source": "synthetic",
            }

            missing_probability = max(0.03, 0.18 - (observed_sessions * 0.018))

            for domain, attributes in ATTRIBUTE_GROUPS.items():
                for attribute in attributes:
                    if rng.random() < missing_probability:
                        row[attribute] = np.nan
                        continue

                    latent = (
                        player_baseline[player]
                        + domain_bias[player][domain]
                        + period_domain_shift[player][period][domain]
                        + rng.normal(0.0, 0.46)
                    )
                    row[attribute] = int(np.clip(np.rint(latent), 1, 5))

            rows.append(row)

    df = pd.DataFrame(rows)

    assert df["player_id"].nunique() == n_players
    assert df["observed_sessions"].ge(1).all()
    assert df["data_source"].eq("synthetic").all()
    for attribute in ALL_ATTRIBUTES:
        observed = df[attribute].dropna()
        assert observed.between(1, 5).all()

    return df.sort_values(["player_id", "evaluation_period"]).reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "synthetic_player_profiles.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_player_profiles()
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} synthetic player-period rows.")
    print(f"Players: {df['player_id'].nunique()}")
    print(f"Periods represented: {df['evaluation_period'].nunique()}")
    print(f"Attributes: {len(ALL_ATTRIBUTES)}")
    print(
        "Missing observation ratings: "
        f"{int(df[list(ALL_ATTRIBUTES)].isna().sum().sum())}"
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
