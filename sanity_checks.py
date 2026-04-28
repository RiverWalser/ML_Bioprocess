import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from bioprocess_optimizer import simulate, optimize_policy

torch.set_default_dtype(torch.float64)


def inner_convergence(IORs=(0.1, 0.3, 0.5, 0.7, 0.9), n_steps=300, lr=0.05):
    curves = {}
    for ior in IORs:
        torch.manual_seed(42)
        theta = torch.zeros(10, requires_grad=True)
        optim = torch.optim.Adam([theta], lr=lr)
        yields = []
        for _ in range(n_steps):
            optim.zero_grad()
            loss = -simulate(theta, ior)
            loss.backward()
            optim.step()
            with torch.no_grad():
                theta.clamp_(0.0, 5.0)
                yields.append(float(simulate(theta, ior).item()))
        curves[ior] = np.array(yields)

    plt.figure(figsize=(9, 5.5))
    for ior, y in curves.items():
        plt.plot(y, linewidth=1.8, label=f"IOR={ior:.2f}")
    plt.xlabel("Adam iteration")
    plt.ylabel("Inner-loop yield F(theta; IOR)")
    plt.title("Sanity check 1: inner-loop gradient descent converges at every IOR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sanity_inner_convergence.png", dpi=150)
    plt.close()
    return curves


def policy_shape_vs_ior(IORs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    policies = {}
    for ior in IORs:
        theta_star, _ = optimize_policy(ior)
        policies[ior] = theta_star

    t = np.linspace(0.0, 1.0, 10)
    fig, axes = plt.subplots(1, len(IORs), figsize=(3.2 * len(IORs), 4.2), sharey=True)
    for ax, (ior, theta) in zip(axes, policies.items()):
        ax.bar(np.arange(1, 11), theta, color="steelblue")
        ax.set_title(f"IOR={ior:.2f}")
        ax.set_xlabel("time interval")
    axes[0].set_ylabel("Feed rate theta*")
    fig.suptitle("Sanity check 2: optimal policy shape changes with IOR")
    plt.tight_layout()
    plt.savefig("sanity_policy_shapes.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for ior, theta in policies.items():
        plt.plot(t, theta, marker="o", linewidth=1.8, label=f"IOR={ior:.2f}")
    plt.xlabel("time t")
    plt.ylabel("theta*(t)")
    plt.title("Optimal feed policy as a function of time, overlaid across IOR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sanity_policy_overlay.png", dpi=150)
    plt.close()
    return policies


def random_baseline(n_iter=25, seed=0):
    torch.manual_seed(seed)
    iors, yields = [], []
    for _ in range(n_iter):
        ior = float((0.01 + 0.98 * torch.rand(1)).item())
        _, y = optimize_policy(ior)
        iors.append(ior)
        yields.append(y)
    return iors, yields


def bo_run(n_init=5, n_iter=25, seed=0):
    torch.manual_seed(seed)
    bounds = torch.tensor([[0.01], [0.99]])
    iors, yields = [], []
    for _ in range(n_init):
        ior = float((0.01 + 0.98 * torch.rand(1)).item())
        _, y = optimize_policy(ior)
        iors.append(ior)
        yields.append(y)

    while len(iors) < n_iter:
        train_X = torch.tensor(iors).unsqueeze(-1)
        train_Y = torch.tensor(yields).unsqueeze(-1)
        try:
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()
            ei = ExpectedImprovement(model, best_f=train_Y.max())
            candidate, _ = optimize_acqf(ei, bounds=bounds, q=1, num_restarts=20, raw_samples=200)
            ior = float(candidate.view(-1)[0].item())
        except Exception:
            ior = float((0.01 + 0.98 * torch.rand(1)).item())
        _, y = optimize_policy(ior)
        iors.append(ior)
        yields.append(y)
    return iors, yields


def bo_vs_random(n_iter=25, n_seeds=5):
    bo_curves, rand_curves = [], []
    for s in range(n_seeds):
        _, bo_y = bo_run(n_init=5, n_iter=n_iter, seed=s)
        _, rd_y = random_baseline(n_iter=n_iter, seed=100 + s)
        bo_curves.append(np.maximum.accumulate(bo_y))
        rand_curves.append(np.maximum.accumulate(rd_y))
    bo_arr = np.array(bo_curves)
    rd_arr = np.array(rand_curves)

    x = np.arange(1, n_iter + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(x, bo_arr.mean(0), color="C0", linewidth=2.2, label="Bayesian optimization")
    plt.fill_between(x, bo_arr.mean(0) - bo_arr.std(0), bo_arr.mean(0) + bo_arr.std(0),
                     color="C0", alpha=0.18)
    plt.plot(x, rd_arr.mean(0), color="C3", linewidth=2.2, label="Random search")
    plt.fill_between(x, rd_arr.mean(0) - rd_arr.std(0), rd_arr.mean(0) + rd_arr.std(0),
                     color="C3", alpha=0.18)
    plt.xlabel("Evaluation #")
    plt.ylabel("Best yield found so far")
    plt.title(f"Sanity check 3: BO beats random search (mean ± std over {n_seeds} seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sanity_bo_vs_random.png", dpi=150)
    plt.close()
    return bo_arr, rd_arr


if __name__ == "__main__":
    print("Running sanity check 1: inner-loop convergence across IORs...")
    inner_convergence()
    print("  -> sanity_inner_convergence.png")

    print("Running sanity check 2: optimal policy shape vs IOR...")
    policy_shape_vs_ior()
    print("  -> sanity_policy_shapes.png, sanity_policy_overlay.png")

    print("Running sanity check 3: BO vs random search (5 seeds, 25 evals each)...")
    bo_arr, rd_arr = bo_vs_random(n_iter=25, n_seeds=5)
    print(f"  BO final mean best yield:     {bo_arr[:, -1].mean():.4f} (std {bo_arr[:, -1].std():.4f})")
    print(f"  Random final mean best yield: {rd_arr[:, -1].mean():.4f} (std {rd_arr[:, -1].std():.4f})")
    print("  -> sanity_bo_vs_random.png")
