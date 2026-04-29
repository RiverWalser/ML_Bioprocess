# rl-ode-bioprocesses

A bilevel optimization toy for a bioprocess: pick the starting mix of two
microbial species (IOR ∈ [0, 1]) on the outside, and a 10-step feed
schedule θ for that IOR on the inside. Yield depends on both, and the two
choices interact, so you can't decouple them.

The simulator here is a hand-rolled differentiable function, not a real
ODE. The point is to get the optimization structure right on something
cheap and with a known answer before plugging in the actual model.

## Layout

- `bioprocess_optimizer.py` — the main run. BO over IOR on the outside,
  gradient-based policy optimization on the inside.
- `sanity_checks.py` — three checks: inner loop converges at every IOR,
  the optimal policy shape actually changes with IOR (so the problem
  isn't trivial), and BO beats random search over 5 seeds.
- `optimizer_benchmark.py` — Adam vs torch L-BFGS vs scipy L-BFGS-B vs
  IPOPT on the inner problem, against a closed-form ground truth.
- PNGs are outputs from those scripts. The notebook is scratch.

## Why bilevel

Each IOR evaluation requires a full inner solve, so it's expensive. That's
exactly the situation BO is good at: sample-efficient, and the GP gives a
posterior so we know where we're still uncertain. Inside, the policy
problem is smooth and only 10-dimensional, so a gradient method handles it
in a handful of iterations.

The two levels really do interact. The simulator weights species 1's
contribution by an early time-window and species 2's by a late one, so the
optimal feed schedule at IOR=0.1 looks nothing like the one at IOR=0.9
(see `sanity_policy_overlay.png`).

## Inner-loop optimizer

The inner objective turns out to be a strictly convex QP with box
constraints: linear coefficients from the species windows, a `0.15·‖θ‖²`
penalty, θ ∈ [0, 5]. It decouples coordinate-wise, which means we have a
closed-form optimum to benchmark against.

We started with Adam in the sanity checks and torch's L-BFGS in the main
loop. The benchmark compares all four candidates on gap-to-optimum and
wall-clock:

| optimizer       | iters to 1e-8 gap | wall time |
|-----------------|-------------------|-----------|
| scipy L-BFGS-B  | 2                 | ~0.3 ms   |
| torch L-BFGS    | 2                 | ~5 ms     |
| IPOPT           | 8–10              | ~7 ms     |
| Adam            | ~190              | ~50 ms    |

scipy's L-BFGS-B wins on overhead: same iteration count as torch's
L-BFGS, no autograd wrapping, native bound handling. IPOPT is overkill on
this problem because the bounds are inactive at the optimum; it'd earn
its keep if we added nonlinear constraints (a total-feed budget, for
instance). Adam works, just ~100× slower than necessary on something this
smooth.

## Running it

```bash
pip install torch botorch gpytorch scipy matplotlib numpy
pip install cyipopt    # only needed for the IPOPT row of the benchmark
                       # on macOS, conda-forge is easier:
                       # conda install -c conda-forge cyipopt

python bioprocess_optimizer.py    # main BO run
python sanity_checks.py           # three sanity figures
python optimizer_benchmark.py     # optimizer comparison
```

The main script saves `bo_posterior.png` (GP posterior over IOR with the
observed points marked) and `optimal_policy.png` (the feed schedule at
the best IOR). The benchmark saves both an iteration-axis and a
wall-clock-axis version of the convergence story.

## What's next

Swap the toy simulator for a real differentiable ODE. The optimization
scaffolding doesn't need to change — that was the point of getting it
right on a problem with a known answer first.
