# RL-ODE-Bioprocesses
1. What is the best **starting mix** of two microbial species? (called `IOR`)
2. What is the best **feeding schedule over time**? (called `theta`)

If you are new to this topic, think of it like this:
- `IOR` is how you choose your starting lineup.
- `theta` is your game plan over 10 time periods.
- The best game plan depends on the lineup, and the best lineup depends on the game plan.

That is why we use a two-level (bilevel) optimization setup.

## What this repository currently contains

- `bioprocess_optimizer.py` — main script that runs the full optimization and saves plots.
- `bo_posterior.png` — plot showing the model's estimate of yield across different IOR values.
- `optimal_policy.png` — plot showing the best 10-step feed policy for the best IOR found.
- `test.ipynb` — scratch notebook.

## What the script does (high-level)

When you run `python bioprocess_optimizer.py`, the script:

1. Uses a toy differentiable simulator function `simulate(theta, IOR)`.
2. For a fixed IOR, finds the best `theta` using gradient-based optimization (Adam).
3. Uses Bayesian Optimization to pick better IOR values over iterations.
4. Repeats this process for 25 total evaluations.
5. Prints progress and final best settings.
6. Saves two plots.

## Why this architecture is used

This is the key design idea:

- **Inner loop (optimize policy):** For one chosen IOR, we optimize the 10 feed rates (`theta`) with gradients.
- **Outer loop (optimize IOR):** We then decide which IOR to try next using a Bayesian model.

This works well because:
- Policy optimization is smooth and fast with gradients.
- IOR evaluation is expensive (it requires a full inner optimization), so Bayesian Optimization is sample-efficient.
- Bayesian Optimization also gives uncertainty estimates, not just a single guess.

## Plain-language explanation of the two plots

### `bo_posterior.png`

This plot answers: **“Which IOR values seem best, and how sure are we?”**

- Blue line = predicted yield for each IOR.
- Blue shaded region = uncertainty band (90% confidence).
- Red dots = IOR values we actually tested.
- Green vertical dashed line = best IOR found.

Use this plot to understand where the model thinks the sweet spot is, and where uncertainty is still high.

### `optimal_policy.png`

This plot answers: **“Given the best IOR, how should we feed over time?”**

- 10 bars = feed rates for 10 time intervals.
- Red dashed horizontal line = mean feed rate.

Use this plot to see the shape of the recommended feeding schedule.

## Definitions (quick glossary)

- **Yield:** final performance score we want to maximize.
- **IOR (Initial Inoculation Ratio):** starting fraction of species 1 (species 2 is `1 - IOR`).
- **Policy (`theta`):** a list of 10 feed rates, one per time interval.
- **Gradient descent / Adam:** method that improves parameters step by step using slope information.
- **Bayesian Optimization:** method for smartly choosing the next experiment when evaluations are expensive.
- **Gaussian Process (GP):** probabilistic model used inside Bayesian Optimization.
- **Expected Improvement (EI):** rule for choosing the next IOR by balancing exploration and exploitation.

## Current optimization settings

- Policy dimension: 10 feed intervals
- Inner optimizer: Adam
- Inner steps per IOR: 300
- Inner learning rate: 0.05
- Policy bounds: each feed rate clipped to `[0.0, 5.0]`
- BO initial random points: 5
- BO total evaluations: 25
- IOR search bounds: `[0.01, 0.99]`
- GP: `SingleTaskGP` with output standardization
- Acquisition: Expected Improvement

## Requirements

Install dependencies:

```bash
pip install torch botorch gpytorch matplotlib numpy
```

## Run

From this folder:

```bash
python bioprocess_optimizer.py
```

## Console output you should expect

- Per iteration: `BO iter {i}: IOR={val:.4f}, Yield={val:.4f}`
- Final summary: `Best IOR: {val:.4f} | Best Yield: {val:.4f}`
- Final policy: `Optimal policy theta*: [...]`
- Plot message: `Plots saved: bo_posterior.png, optimal_policy.png`

## How this matches the assignment request

The assignment asked for:
- optimize both feed policy and IOR,
- use Bayesian methods,
- quantify uncertainty,
- and demonstrate on a differentiable toy Python function.

This project does all four in one runnable script.

Later, the toy simulator can be replaced by a real differentiable ODE simulator while keeping the same optimization structure.
