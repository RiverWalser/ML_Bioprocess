"""Benchmark inner-loop optimizers on the bioprocess yield problem.

Inner objective is strictly convex with box constraints:
    F(theta; IOR) = c(IOR)·theta - 0.15·||theta||^2 + const
    s.t.  0 <= theta_i <= 5
The optimum decouples coordinate-wise, so we have a closed-form ground
truth: theta_i* = clip(c_i / 0.30, 0, 5). We use it to measure
gap-to-optimum exactly.
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize

try:
    import cyipopt
    HAS_IPOPT = True
except ImportError:
    HAS_IPOPT = False

torch.set_default_dtype(torch.float64)

DIM = 10
LB, UB = 0.0, 5.0
QUAD = 0.15


def linear_coeffs(ior):
    t = np.linspace(0.0, 1.0, DIM)
    early = np.exp(-30.0 * (t - 0.15) ** 2)
    late = np.exp(-30.0 * (t - 0.85) ** 2)
    return ior * early + (1.0 - ior) * late


def yield_value(theta, ior):
    c = linear_coeffs(ior)
    diversity = 9.0 * ior * (1.0 - ior)
    return float(c @ theta - QUAD * (theta @ theta) + diversity)


def closed_form(ior):
    c = linear_coeffs(ior)
    theta_star = np.clip(c / (2.0 * QUAD), LB, UB)
    return theta_star, yield_value(theta_star, ior)


def bench_adam(ior, n_steps=500, lr=0.05):
    torch.manual_seed(42)
    theta = torch.zeros(DIM, requires_grad=True)
    optim = torch.optim.Adam([theta], lr=lr)
    c_t = torch.tensor(linear_coeffs(ior))
    div = 9.0 * ior * (1.0 - ior)
    yields = [yield_value(np.zeros(DIM), ior)]
    times = [0.0]
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optim.zero_grad()
        loss = -(c_t @ theta - QUAD * (theta @ theta) + div)
        loss.backward()
        optim.step()
        with torch.no_grad():
            theta.clamp_(LB, UB)
            yields.append(yield_value(theta.detach().numpy(), ior))
            times.append(time.perf_counter() - t0)
    return np.array(yields), np.array(times)


def bench_torch_lbfgs(ior, n_steps=50, lr=1.0):
    torch.manual_seed(42)
    theta = torch.zeros(DIM, requires_grad=True)
    optim = torch.optim.LBFGS(
        [theta], lr=lr, max_iter=1, history_size=20,
        line_search_fn="strong_wolfe",
    )
    c_t = torch.tensor(linear_coeffs(ior))
    div = 9.0 * ior * (1.0 - ior)
    yields = [yield_value(np.zeros(DIM), ior)]
    times = [0.0]
    t0 = time.perf_counter()
    for _ in range(n_steps):
        def closure():
            optim.zero_grad()
            loss = -(c_t @ theta - QUAD * (theta @ theta) + div)
            loss.backward()
            return loss
        optim.step(closure)
        with torch.no_grad():
            theta.clamp_(LB, UB)
            yields.append(yield_value(theta.detach().numpy(), ior))
            times.append(time.perf_counter() - t0)
    return np.array(yields), np.array(times)


def bench_scipy_lbfgsb(ior):
    c = linear_coeffs(ior)
    div = 9.0 * ior * (1.0 - ior)

    def neg_F(x):
        return -(c @ x - QUAD * (x @ x) + div)

    def neg_grad(x):
        return -(c - 2.0 * QUAD * x)

    yields = [yield_value(np.zeros(DIM), ior)]
    times = [0.0]
    t0 = time.perf_counter()

    def cb(xk):
        x = xk.x if hasattr(xk, "x") else xk
        yields.append(yield_value(np.asarray(x), ior))
        times.append(time.perf_counter() - t0)

    scipy_minimize(
        neg_F, np.zeros(DIM), jac=neg_grad, method="L-BFGS-B",
        bounds=[(LB, UB)] * DIM, callback=cb,
        options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 200},
    )
    return np.array(yields), np.array(times)


class _IpoptInner:
    def __init__(self, c, div, ior):
        self.c = c
        self.div = div
        self.ior = ior
        self.yields = []
        self.times = []
        self.t0 = None
        self._x_last = None

    def objective(self, x):
        self._x_last = np.asarray(x).copy()
        return float(-(self.c @ x - QUAD * (x @ x) + self.div))

    def gradient(self, x):
        return -(self.c - 2.0 * QUAD * np.asarray(x))

    def constraints(self, x):
        return np.array([])

    def jacobian(self, x):
        return np.array([])

    def intermediate(self, *args, **kwargs):
        if self._x_last is not None:
            self.yields.append(yield_value(self._x_last, self.ior))
            self.times.append(time.perf_counter() - self.t0)


def bench_ipopt(ior):
    if not HAS_IPOPT:
        return None, None
    c = linear_coeffs(ior)
    div = 9.0 * ior * (1.0 - ior)
    prob = _IpoptInner(c, div, ior)
    nlp = cyipopt.Problem(
        n=DIM, m=0, problem_obj=prob,
        lb=[LB] * DIM, ub=[UB] * DIM,
        cl=[], cu=[],
    )
    nlp.add_option("print_level", 0)
    nlp.add_option("sb", "yes")
    nlp.add_option("tol", 1e-12)
    nlp.add_option("max_iter", 200)
    prob.t0 = time.perf_counter()
    prob.yields.append(yield_value(np.zeros(DIM), ior))
    prob.times.append(0.0)
    nlp.solve(np.zeros(DIM))
    return np.array(prob.yields), np.array(prob.times)


def run_all(IORs=(0.1, 0.3, 0.5, 0.7, 0.9), gap_thresh=1e-8):
    optimizers = [
        ("Adam",           bench_adam,         "C3"),
        ("torch-LBFGS",    bench_torch_lbfgs,  "C0"),
        ("scipy L-BFGS-B", bench_scipy_lbfgsb, "C2"),
        ("IPOPT",          bench_ipopt,        "C4"),
    ]

    rows = []
    fig_it, axes_it = plt.subplots(1, len(IORs), figsize=(4 * len(IORs), 4.2),
                                   sharey=True)
    fig_wt, axes_wt = plt.subplots(1, len(IORs), figsize=(4 * len(IORs), 4.2),
                                   sharey=True)

    for ax_it, ax_wt, ior in zip(axes_it, axes_wt, IORs):
        _, F_star = closed_form(ior)
        for name, fn, color in optimizers:
            ys, ts = fn(ior)
            if ys is None:
                continue
            gaps = np.maximum(F_star - ys, 1e-16)
            ax_it.semilogy(np.arange(len(gaps)), gaps, label=name, color=color,
                           linewidth=1.6)
            ax_wt.semilogy(ts * 1000.0, gaps, label=name, color=color,
                           linewidth=1.6)
            iters_to_thresh = next(
                (i for i, g in enumerate(gaps) if g < gap_thresh),
                None,
            )
            rows.append({
                "IOR": ior,
                "optimizer": name,
                "iters_to_1e-8": iters_to_thresh,
                "final_gap": float(gaps[-1]),
                "wall_time_ms": float(ts[-1] * 1000.0),
                "n_iter": len(gaps) - 1,
            })
        for ax in (ax_it, ax_wt):
            ax.set_title(f"IOR={ior:.2f}")
            ax.grid(alpha=0.3, which="both")
        ax_it.set_xlabel("iteration")
        ax_wt.set_xlabel("wall-clock time (ms)")

    axes_it[0].set_ylabel(r"gap  $F^* - F(\theta)$")
    axes_wt[0].set_ylabel(r"gap  $F^* - F(\theta)$")
    axes_it[-1].legend(loc="upper right", fontsize=9)
    axes_wt[-1].legend(loc="upper right", fontsize=9)
    fig_it.suptitle("Inner-loop optimizer benchmark — gap vs iteration")
    fig_wt.suptitle("Inner-loop optimizer benchmark — gap vs wall-clock")
    fig_it.tight_layout()
    fig_wt.tight_layout()
    fig_it.savefig("optimizer_benchmark_iters.png", dpi=150)
    fig_wt.savefig("optimizer_benchmark_walltime.png", dpi=150)
    plt.close(fig_it)
    plt.close(fig_wt)

    print(f"\n{'IOR':>5} {'optimizer':>16} {'iters→1e-8':>12} "
          f"{'final gap':>14} {'wall (ms)':>12} {'n_iter':>8}")
    print("-" * 74)
    for r in rows:
        it = "—" if r["iters_to_1e-8"] is None else str(r["iters_to_1e-8"])
        print(f"{r['IOR']:>5.2f} {r['optimizer']:>16} {it:>12} "
              f"{r['final_gap']:>14.3e} {r['wall_time_ms']:>12.2f} "
              f"{r['n_iter']:>8}")
    return rows


if __name__ == "__main__":
    if not HAS_IPOPT:
        print("WARNING: cyipopt not found — skipping IPOPT.\n")
    run_all()
    print("\nSaved: optimizer_benchmark_iters.png, optimizer_benchmark_walltime.png")
