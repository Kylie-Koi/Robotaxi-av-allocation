# Results

Equal-total-compute comparison results (every architecture trained on
1,000,000 environment-interaction samples total, evaluated over 10 seeded
episodes).

## Architectures compared

All twelve architecture/zone-count combinations (paper Section 5): zone
count, number of learners, whether policy weights are shared, and how
coordination across zones/agents is achieved. Also in
[`architectures_compared.csv`](architectures_compared.csv).

| Architecture | Zones | Learners | Policy sharing | Coordination mechanism |
|---|---|---|---|---|
| Single-agent, borough | 7 | 1 | n/a | One network scores all 7 zones jointly |
| Single-agent, taxi zone | 261 | 1 | n/a | One network scores all 261 zones jointly |
| Single-agent, K-means 15 zones | 15 (K-means clusters) | 1 | n/a | One network scores all 15 zones jointly |
| Single-agent, K-means 30 zones | 30 (K-means clusters) | 1 | n/a | One network scores all 30 zones jointly |
| Single-agent, K-means 50 zones | 50 (K-means clusters) | 1 | n/a | One network scores all 50 zones jointly |
| Flat multi-agent (IPPO), 7 zones | 7 | 7 | Parameter-shared | One PPO agent per zone; joint softmax over all agents' scores |
| Flat multi-agent (IPPO), 10 zones | 10 (K-means clusters) | 10 | Parameter-shared | Same as above, 10 agents |
| Flat multi-agent (IPPO), 265 zones | 265 (261 active) | 260 | Parameter-shared | Same as above, 260 agents (one per active zone) |
| Hierarchical (feudal), lead + 7 workers | 7 | 8 (1 lead + 7 workers) | Separate lead/worker networks | Lead sets per-zone directives; workers refine them; alternating best-response training over 3 rounds |
| Hierarchical (feudal), lead + 260 workers, 265 zones | 265 (261 active) | 261 (1 lead + 260 workers) | Separate lead/worker networks | Lead sets per-zone directives; workers refine them; alternating best-response training over 3 rounds |
| Hierarchical (feudal), workers-first, 7 workers + lead | 7 | 8 (7 workers + 1 lead) | Separate lead/worker networks | Workers set per-zone preferences first (joint softmax); lead applies a bounded corrective adjustment second; alternating best-response training over 3 rounds, workers first each round |
| Hierarchical (feudal), workers-first, 260 workers + lead, 265 zones | 265 (261 active) | 261 (260 workers + 1 lead) | Separate lead/worker networks | Workers set per-zone preferences first (joint softmax); lead applies a bounded corrective adjustment second; alternating best-response training over 3 rounds, workers first each round |

## Zone-count sweep (single-agent, K=15/30/50)

![Zone-count sweep: single-agent profit and elder-care violations at K=7, 15, 30, 50, and 261 zones](zone_count_sweep.svg)

Table: [`zone_count_sweep.csv`](zone_count_sweep.csv). Source: [`single-agent-zone-sweep/`](../single-agent-zone-sweep).

## 265-zone architecture comparison

![Architecture comparison at 265 official taxi zones: profit and elder-care violations](265zone_architecture_comparison.svg)

Table: [`265zone_architecture_comparison.csv`](265zone_architecture_comparison.csv). Sources:
[`single-agent/`](../single-agent) (265-zone config),
[`multi-agent-265zone/`](../multi-agent-265zone),
[`hierarchical-lead-first-265zone/`](../hierarchical-lead-first-265zone),
[`hierarchical-workers-first-265zone/`](../hierarchical-workers-first-265zone).

Each `*_mean` column in both CSVs is paired with a `*_std` column
(population std across the 10 evaluation episodes).
