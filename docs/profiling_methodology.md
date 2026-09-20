# Methodology — Project 03 V0.2

## Scope

This project demonstrates a privacy-safe workflow for structuring and visualising
coach observations in youth football using fully synthetic data.

It does not claim to measure a player's complete quality, talent, potential,
psychological state, medical status or selection suitability.

## Observation domains

The public framework contains four domains:

### Technical
First touch, passing, receiving orientation, ball control and dribbling.

### Tactical
Positioning, scanning, decision making, support and transition response.

### Physical observations
Acceleration, change of direction and repeat effort.

These are observational football behaviours, not laboratory, GPS or diagnostic
measurements.

### Behavioural observations
Communication, concentration, training engagement and response to feedback.

`Behavioural observation` is used deliberately. The project does not claim to
measure psychological traits, mental-health constructs or clinical variables.

## Behaviourally anchored ordinal scale

The rating scale is:

1. rarely demonstrated effectively
2. occasionally demonstrated effectively
3. regularly demonstrated effectively
4. frequently demonstrated effectively
5. consistently demonstrated effectively
NA. insufficient observation

The separate `observation_framework.csv` provides a definition and anchor
examples for each attribute.

The scale is ordinal. The intervals between values are not assumed to be equal,
so medians are used as the primary aggregation statistic.

## Missing observation is not poor performance

NA means that the attribute was not observed sufficiently in that player-period.
It is never converted to zero.

Domain medians require at least 50% of the attributes in that domain to be
observed. Otherwise, the domain median remains missing.

## Observation coverage

Each player-period includes:
- `observed_sessions`;
- `observation_context` (`Training`, `Match`, or `Mixed`);
- number of observed attributes;
- percentage attribute coverage.

The project deliberately exposes observation quantity rather than inventing a
confidence score.

## Position and evaluation periods

The pipeline allows:
- missing evaluation periods;
- changes of broad position group across periods.

This is important in developmental football, where observation schedules are
incomplete and player roles may change.

## Peer-group comparison

The latest available player profile can be compared with other synthetic players
in the same broad position group.

The focal player is excluded from the reference calculation. The peer reference
also matches the focal player's evaluation period and current broad position
group. This avoids both self-inclusion and cross-period comparison leakage.

These peer medians are not norms, benchmarks, cut-offs or talent standards.

## Synthetic-data design

V0.2 deliberately removes programmed positional ability advantages.

The generator also removes the previous forced positive P1→P3 trend. Individual
synthetic trajectories vary around zero and can improve, remain stable or
decline by chance.

The purpose is to test and demonstrate the analysis pipeline, not to manufacture
football conclusions.

## Visual outputs

The project uses:
- a four-domain radar for a compact individual overview;
- a horizontal attribute dot plot for detailed inspection;
- player vs leave-one-out peer-group domain comparison;
- a discrete 1–5 squad heatmap;
- longitudinal change across available observation periods;
- observation-session coverage.

The term `change` is preferred to `progression` because change is descriptive and
does not automatically imply development.

## Important limitations

Coach ratings are subjective observations. They depend on the observer, context,
opportunity to display behaviours, role demands and observation duration.

This public demo does not contain inter-rater reliability estimates because it
contains no real raters or repeated independent observations.

The framework should therefore be interpreted as a transparent structure for
organising observations, not as a validated measurement instrument.

## Privacy

All public player-level records are generated synthetically.

The public repository excludes names, real dates of birth, shirt numbers,
identifiable real positions, real coach comments, psychological profiles,
injury/medical information and confidential club identifiers.
