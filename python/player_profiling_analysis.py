"""Analyse synthetic structured player-observation profiles.

Project 03 — Youth Football Player Performance Profiling (V0.2)

The pipeline structures and visualises coach observations. It is descriptive and
development-oriented. It is not a clinical, psychological, medical, recruitment,
talent-identification, or automated player-selection system.

Ordinal ratings use values 1–5. Missing values mean insufficient observation,
not low performance. Medians are used as the primary aggregation statistic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "synthetic_player_profiles.csv"
DEFAULT_FRAMEWORK = REPO_ROOT / "data" / "reference" / "observation_framework.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

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

REQUIRED_COLUMNS = {
    "player_id",
    "evaluation_period",
    "position_group",
    "observation_context",
    "observed_sessions",
    "data_source",
    *ALL_ATTRIBUTES,
}

MIN_DOMAIN_COVERAGE = 0.50


def load_profile_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Public demo expects a CSV input file.")
    return pd.read_csv(path)


def load_observation_framework(path: str | Path = DEFAULT_FRAMEWORK) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Observation framework not found: {path}")
    framework = pd.read_csv(path)

    required = {
        "domain",
        "attribute",
        "definition",
        "rating_1_anchor",
        "rating_3_anchor",
        "rating_5_anchor",
    }
    missing = required.difference(framework.columns)
    if missing:
        raise ValueError(f"Framework missing required columns: {sorted(missing)}")

    if framework["attribute"].duplicated().any():
        raise ValueError("Observation framework contains duplicate attributes.")

    if set(framework["attribute"]) != set(ALL_ATTRIBUTES):
        raise ValueError(
            "Observation framework attributes do not match the analysis contract."
        )

    return framework


def _require_nonblank_text(df: pd.DataFrame, column: str) -> None:
    if df[column].isna().any():
        raise ValueError(f"{column} must not contain missing values.")
    values = df[column].astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"{column} must not contain blank values.")
    df[column] = values


def _validate_rating_column(series: pd.Series, attribute: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    invalid_nonblank = series.notna() & numeric.isna()
    if invalid_nonblank.any():
        raise ValueError(f"{attribute} contains invalid non-numeric ratings.")

    observed = numeric.dropna()
    if not np.all(np.isclose(observed, np.round(observed))):
        raise ValueError(f"{attribute} must contain integer ordinal ratings or NA.")
    if not observed.between(1, 5).all():
        raise ValueError(f"{attribute} observed ratings must be between 1 and 5.")

    return numeric.astype("Float64")


def validate_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema while allowing real-world-style missing observation data."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    for column in [
        "player_id",
        "evaluation_period",
        "position_group",
        "observation_context",
        "data_source",
    ]:
        _require_nonblank_text(out, column)

    invalid_periods = sorted(set(out["evaluation_period"]) - set(EVALUATION_PERIODS))
    if invalid_periods:
        raise ValueError(f"Invalid evaluation_period values: {invalid_periods}")

    invalid_positions = sorted(set(out["position_group"]) - set(POSITION_GROUPS))
    if invalid_positions:
        raise ValueError(f"Invalid position_group values: {invalid_positions}")

    invalid_contexts = sorted(
        set(out["observation_context"]) - set(OBSERVATION_CONTEXTS)
    )
    if invalid_contexts:
        raise ValueError(f"Invalid observation_context values: {invalid_contexts}")

    if not out["data_source"].eq("synthetic").all():
        raise ValueError(
            "Public demo accepts only rows explicitly marked data_source='synthetic'."
        )

    sessions = pd.to_numeric(out["observed_sessions"], errors="coerce")
    if sessions.isna().any():
        raise ValueError("observed_sessions must contain valid integers.")
    if not np.all(np.isclose(sessions, np.round(sessions))):
        raise ValueError("observed_sessions must contain integer values.")
    sessions = sessions.astype(int)
    if (sessions <= 0).any():
        raise ValueError("observed_sessions must be > 0.")
    out["observed_sessions"] = sessions

    for attribute in ALL_ATTRIBUTES:
        out[attribute] = _validate_rating_column(out[attribute], attribute)

    if out.duplicated(subset=["player_id", "evaluation_period"]).any():
        raise ValueError("Duplicate player_id + evaluation_period rows detected.")

    if out[list(ALL_ATTRIBUTES)].isna().all(axis=1).any():
        raise ValueError("Each player-period row needs at least one observed attribute.")

    period_order = pd.CategoricalDtype(EVALUATION_PERIODS, ordered=True)
    out["evaluation_period"] = out["evaluation_period"].astype(period_order)

    return out.sort_values(
        ["player_id", "evaluation_period"]
    ).reset_index(drop=True)


def add_observation_coverage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["observed_attribute_count"] = out[list(ALL_ATTRIBUTES)].notna().sum(axis=1)
    out["observation_coverage_pct"] = (
        out["observed_attribute_count"] / len(ALL_ATTRIBUTES) * 100.0
    ).round(1)
    return out


def _domain_median_with_coverage(row: pd.Series, attributes: tuple[str, ...]) -> float:
    values = row[list(attributes)].dropna()
    required = int(np.ceil(len(attributes) * MIN_DOMAIN_COVERAGE))
    if len(values) < required:
        return np.nan
    return float(values.median())


def add_domain_medians(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for domain, attributes in ATTRIBUTE_GROUPS.items():
        out[f"{domain}_observed_count"] = out[list(attributes)].notna().sum(axis=1)
        out[f"{domain}_median"] = out.apply(
            lambda row: _domain_median_with_coverage(row, attributes),
            axis=1,
        )
    return out


def latest_available_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return each player's latest available evaluation period."""
    ordered = df.copy()
    codes = ordered["evaluation_period"].cat.codes
    ordered = ordered.assign(_period_code=codes)
    idx = ordered.groupby("player_id")["_period_code"].idxmax()
    return ordered.loc[idx].drop(columns="_period_code").reset_index(drop=True)


def _leave_one_out_peer_medians(
    all_rows: pd.DataFrame,
    player_id: str,
    position_group: str,
    evaluation_period: str,
) -> tuple[pd.Series, int]:
    """Return same-period, same-position peer medians excluding the focal player."""
    period_text = all_rows["evaluation_period"].astype(str)
    peers = all_rows[
        all_rows["position_group"].eq(position_group)
        & period_text.eq(str(evaluation_period))
        & ~all_rows["player_id"].eq(player_id)
    ]
    peer_n = int(len(peers))
    if peer_n == 0:
        return pd.Series({attr: np.nan for attr in ALL_ATTRIBUTES}), 0
    return peers[list(ALL_ATTRIBUTES)].median(skipna=True), peer_n


def build_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_domain_medians(add_observation_coverage(df))
    latest = latest_available_rows(enriched)

    rows: list[dict] = []
    for _, row in latest.iterrows():
        peer_medians, peer_n = _leave_one_out_peer_medians(
            enriched,
            player_id=row["player_id"],
            position_group=row["position_group"],
            evaluation_period=str(row["evaluation_period"]),
        )

        output = {
            "player_id": row["player_id"],
            "latest_evaluation_period": str(row["evaluation_period"]),
            "position_group": row["position_group"],
            "observation_context": row["observation_context"],
            "observed_sessions": int(row["observed_sessions"]),
            "observed_attribute_count": int(row["observed_attribute_count"]),
            "observation_coverage_pct": float(row["observation_coverage_pct"]),
            "peer_group_n_excluding_player": peer_n,
        }

        for domain in ATTRIBUTE_GROUPS:
            output[f"{domain}_median"] = row[f"{domain}_median"]

        for attribute in ALL_ATTRIBUTES:
            output[f"{attribute}_peer_median"] = peer_medians[attribute]

        rows.append(output)

    return pd.DataFrame(rows)


def _domain_labels() -> list[str]:
    return ["Technical", "Tactical", "Physical observation", "Behavioural observation"]


def choose_example_player(
    latest: pd.DataFrame,
    preferred_player: str = "Player_01",
) -> str:
    """Choose a reproducible demo player with complete domain-level coverage."""
    domain_cols = [f"{domain}_median" for domain in ATTRIBUTE_GROUPS]
    eligible = latest.dropna(subset=domain_cols).sort_values("player_id")
    if eligible.empty:
        raise ValueError("No player has sufficient coverage across all four domains.")

    if preferred_player in set(eligible["player_id"]):
        return preferred_player
    return str(eligible.iloc[0]["player_id"])


def choose_peer_comparison_player(
    all_rows: pd.DataFrame,
    latest: pd.DataFrame,
    preferred_player: str = "Player_01",
) -> str:
    """Choose a demo player with domain coverage and at least one matched peer."""
    domain_cols = [f"{domain}_median" for domain in ATTRIBUTE_GROUPS]
    eligible = latest.dropna(subset=domain_cols).sort_values("player_id")

    valid_players: list[str] = []
    for _, row in eligible.iterrows():
        period_text = all_rows["evaluation_period"].astype(str)
        peers = all_rows[
            all_rows["position_group"].eq(row["position_group"])
            & period_text.eq(str(row["evaluation_period"]))
            & ~all_rows["player_id"].eq(row["player_id"])
        ]
        if len(peers) > 0:
            valid_players.append(str(row["player_id"]))

    if not valid_players:
        raise ValueError("No player has sufficient coverage and a matched peer group.")
    if preferred_player in valid_players:
        return preferred_player
    return valid_players[0]


def save_player_domain_radar(
    latest: pd.DataFrame,
    output_dir: Path,
    example_player: str = "Player_01",
) -> None:
    example_player = choose_example_player(latest, example_player)
    player = latest[latest["player_id"].eq(example_player)]
    row = player.iloc[0]

    domain_cols = [f"{domain}_median" for domain in ATTRIBUTE_GROUPS]
    values = [float(row[col]) for col in domain_cols]

    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    values = values + values[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, marker="o", label=example_player)
    ax.fill(angles, values, alpha=0.12)
    ax.set_xticks(angles[:-1], _domain_labels())
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title(f"{example_player} — latest synthetic domain profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "player_domain_radar.png", dpi=160)
    plt.close(fig)


def save_player_attribute_profile(
    latest: pd.DataFrame,
    output_dir: Path,
    example_player: str = "Player_01",
) -> None:
    example_player = choose_example_player(latest, example_player)
    player = latest[latest["player_id"].eq(example_player)]
    row = player.iloc[0]

    labels = [
        attr.replace("technical_", "")
        .replace("tactical_", "")
        .replace("physical_", "")
        .replace("behaviour_", "")
        .replace("_", " ")
        for attr in ALL_ATTRIBUTES
    ]
    values = [row[attr] for attr in ALL_ATTRIBUTES]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    observed_mask = np.array([not pd.isna(v) for v in values])
    observed_y = y[observed_mask]
    observed_values = np.array(
        [float(v) for v in values if not pd.isna(v)]
    )

    ax.scatter(observed_values, observed_y, s=50)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.invert_yaxis()
    ax.set_xlabel("Ordinal observation rating")
    ax.set_title(f"{example_player} — detailed latest-period attribute profile")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "player_attribute_profile.png", dpi=160)
    plt.close(fig)


def save_player_vs_peer_group(
    all_rows: pd.DataFrame,
    latest: pd.DataFrame,
    output_dir: Path,
    example_player: str = "Player_01",
) -> None:
    example_player = choose_peer_comparison_player(
        all_rows, latest, example_player
    )
    player = latest[latest["player_id"].eq(example_player)]
    row = player.iloc[0]

    period_text = all_rows["evaluation_period"].astype(str)
    peers = all_rows[
        all_rows["position_group"].eq(row["position_group"])
        & period_text.eq(str(row["evaluation_period"]))
        & ~all_rows["player_id"].eq(example_player)
    ]

    domain_cols = [f"{domain}_median" for domain in ATTRIBUTE_GROUPS]
    player_values = np.array([row[col] for col in domain_cols], dtype=float)
    peer_values = peers[domain_cols].median(skipna=True).to_numpy(dtype=float)

    x = np.arange(len(domain_cols))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(player_values, x, s=70, label=example_player)
    ax.scatter(
        peer_values,
        x,
        s=70,
        marker="x",
        label="Other same-position peers median",
    )
    ax.set_yticks(x, _domain_labels())
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Median ordinal observation rating")
    ax.set_title(
        f"{example_player} vs other synthetic {row['position_group']} peers"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "player_vs_peer_group.png", dpi=160)
    plt.close(fig)


def save_squad_heatmap(latest: pd.DataFrame, output_dir: Path) -> None:
    matrix = latest.set_index("player_id")[list(ALL_ATTRIBUTES)].sort_index()
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap(None, 5)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

    labels = [
        c.replace("technical_", "T: ")
        .replace("tactical_", "Ta: ")
        .replace("physical_", "P: ")
        .replace("behaviour_", "B: ")
        .replace("_", " ")
        for c in matrix.columns
    ]
    ax.set_xticks(
        range(len(matrix.columns)),
        labels,
        rotation=70,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_xlabel("Observed attribute")
    ax.set_ylabel("Synthetic player")
    ax.set_title("Latest available synthetic squad observation heatmap")
    ax.text(
        0.0,
        -0.20,
        "Blank cells = insufficient observation",
        transform=ax.transAxes,
        fontsize=8,
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[1, 2, 3, 4, 5])
    cbar.set_label("Ordinal observation rating")
    fig.tight_layout()
    fig.savefig(output_dir / "squad_profile_heatmap.png", dpi=160)
    plt.close(fig)


def save_player_change_over_time(
    enriched: pd.DataFrame,
    output_dir: Path,
    example_player: str = "Player_01",
) -> None:
    latest = latest_available_rows(enriched)
    example_player = choose_example_player(latest, example_player)
    player = enriched[enriched["player_id"].eq(example_player)].copy()

    period_codes = {
        period: index
        for index, period in enumerate(EVALUATION_PERIODS)
    }
    x = np.array([
        period_codes[str(period)]
        for period in player["evaluation_period"]
    ])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for domain in ATTRIBUTE_GROUPS:
        y = player[f"{domain}_median"].astype(float).to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=domain.replace("_observation", "").replace("_", " "),
        )

    ax.set_xticks(range(len(EVALUATION_PERIODS)), EVALUATION_PERIODS)
    ax.set_ylim(1, 5)
    ax.set_xlabel("Evaluation period")
    ax.set_ylabel("Median ordinal observation rating")
    ax.set_title(f"{example_player} — change across synthetic observation periods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "player_change_over_time.png", dpi=160)
    plt.close(fig)


def save_observation_coverage(
    latest: pd.DataFrame,
    output_dir: Path,
) -> None:
    ordered = latest.sort_values(
        ["observed_sessions", "player_id"],
        ascending=[True, True],
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(ordered["player_id"], ordered["observed_sessions"])
    ax.set_xlabel("Observed sessions in latest available period")
    ax.set_ylabel("Synthetic player")
    ax.set_title("Observation coverage context")
    fig.tight_layout()
    fig.savefig(output_dir / "observation_coverage.png", dpi=160)
    plt.close(fig)


def save_outputs(
    validated: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    covered = add_observation_coverage(validated)
    enriched = add_domain_medians(covered)
    latest = latest_available_rows(enriched)
    summary = build_player_summary(validated)

    enriched.to_csv(output_dir / "validated_profiles.csv", index=False)
    summary.to_csv(output_dir / "player_profile_summary.csv", index=False)

    save_player_domain_radar(latest, output_dir)
    save_player_attribute_profile(latest, output_dir)
    save_player_vs_peer_group(enriched, latest, output_dir)
    save_squad_heatmap(latest, output_dir)
    save_player_change_over_time(enriched, output_dir)
    save_observation_coverage(latest, output_dir)

    return enriched, latest, summary


def run_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    framework_path: str | Path = DEFAULT_FRAMEWORK,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    load_observation_framework(framework_path)
    raw = load_profile_data(input_path)
    validated = validate_profile_data(raw)
    enriched, latest, summary = save_outputs(validated, output_dir)
    return validated, enriched, latest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--framework", default=str(DEFAULT_FRAMEWORK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validated, enriched, latest, summary = run_analysis(
        input_path=args.input,
        framework_path=args.framework,
        output_dir=args.output_dir,
    )
    print(f"Validated synthetic rows: {len(validated)}")
    print(f"Synthetic players: {validated['player_id'].nunique()}")
    print(f"Latest-player rows: {len(latest)}")
    print(
        "Missing ratings retained as insufficient observation: "
        f"{int(validated[list(ALL_ATTRIBUTES)].isna().sum().sum())}"
    )
    print(f"Summary rows: {len(summary)}")
    print(f"Outputs saved to: {args.output_dir}")
    print(
        "No overall score, ranking, selection probability, "
        "or clinical interpretation is produced."
    )


if __name__ == "__main__":
    main()
