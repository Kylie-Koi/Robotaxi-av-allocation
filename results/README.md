# Results

Equal-total-compute comparison results (every architecture trained on
1,000,000 environment-interaction samples total, evaluated over 10 seeded
episodes). Full writeup, methodology, and discussion: **[`paper.pdf`](paper.pdf)**
("Optimal Allocation Architectures for Robotaxi Fleet Management via
Multi-Agent Reinforcement Learning"), including both comparison charts
(Figures in Section 7).

## `zone_count_sweep.csv`

Single-agent architecture at 15, 30, and 50 geographic (KMeans) zones --
isolates the effect of decomposition granularity, independent of
architecture. Source: [`single-agent-zone-sweep/`](../single-agent-zone-sweep).

## `265zone_architecture_comparison.csv`

Single-agent, flat multi-agent (IPPO), and both hierarchical orderings
(lead-first, workers-first), each scaled to the full 261 active NYC TLC
taxi zones (265 official zones). Sources:
[`single-agent/`](../single-agent) (265-zone config),
[`multi-agent-265zone/`](../multi-agent-265zone),
[`hierarchical-lead-first-265zone/`](../hierarchical-lead-first-265zone),
[`hierarchical-workers-first-265zone/`](../hierarchical-workers-first-265zone).

At this scale the ranking inverts relative to the 7-10 zone results: pure
decentralization (flat multi-agent) collapses (profit roughly 40% below
the others, violations roughly 2x), while both hierarchies -- workers-first
more than lead-first -- slightly *exceed* the 265-zone single-agent
reference on profit, reversing the small-scale pattern where decomposition
only ever caught up to, never beat, the monolithic policy.

Each `*_mean` column is paired with a `*_std` column (population std across
the 10 evaluation episodes).
