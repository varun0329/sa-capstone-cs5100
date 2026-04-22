# CS5100 Capstone - Simulated Annealing

Varun Agarwal | CS5100 Foundations of AI | Spring 2026 | Northeastern University

## What This Is

This is my capstone project for CS5100 where I reproduce and extend the 1983 paper by Kirkpatrick, Gelatt, and Vecchi - "Optimization by Simulated Annealing" (Science, 220(4598), 671-680).

The idea was to take the core SA algorithm, implement it from scratch, and test it on three different NP-hard problems to see if the paper's claims actually hold up. I also compared three cooling schedules and looked at whether smarter (MCMC-style) neighborhood generation helps.

## What's In Here

- `SA_Capstone_Final.ipynb` - the main notebook, runs on Google Colab
- `sa_capstone.py` - standalone Python script version
- `results/` - saved plots from the experiments

## How to Run

Open the notebook in Google Colab, hit Runtime > Run All. Takes about 2-3 minutes. Only needs numpy, matplotlib, and scipy which are already on Colab.

Seeds are fixed (random.seed(42), np.random.seed(42)) so results are reproducible.

## Problems Tested

**0/1 Knapsack** - FSU benchmark instances P01, P02, P06, P07, P08. Known optimal values, so we can measure exact % of optimal.

**Traveling Salesman Problem** - burma14 (14 cities, optimal 30.88), ulysses22 (22 cities, optimal 75.31), and a random 30-city instance.

**Graph Coloring** - Petersen graph (10 nodes, chromatic number 3), Queen5x5 (25 nodes, chromatic number 5), random graph (30 nodes).

## What Was Compared

Three cooling schedules: geometric (T * alpha), linear (T - delta), logarithmic (T0/log(1+k)). Two neighborhood strategies: random perturbation vs MCMC-style (biased toward better moves). Hill climbing with random restarts as a baseline. 15 runs per configuration.

## Main Findings

Geometric cooling is the best choice across all three domains. The difference is huge on TSP - geometric gets 99.1% of optimal on ulysses22 while logarithmic only manages 63.7%. ANOVA confirms this is statistically significant (p < 0.001).

SA beats hill climbing on the harder instances (knapsack P01: 100% vs 94.3%, p = 0.005), which supports the paper's claim about escaping local optima through the Metropolis criterion.

MCMC neighborhoods matter most for graph coloring. On Queen5x5, linear+MCMC finds 11/15 valid colorings while linear+random finds 0/15. Targeting conflicting nodes directly is way more effective than random recoloring.

## Links

- Demo Video: [INSERT YOUTUBE LINK]
- Final Report: submitted separately

## References

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. Science, 220(4598), 671-680.
