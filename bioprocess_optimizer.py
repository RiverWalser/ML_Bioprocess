import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

def simulate(theta, IOR):
    t = torch.linspace(0.0, 1.0, 10, dtype=theta.dtype, device=theta.device)
    early_window = torch.exp(-30.0 * (t - 0.15) ** 2)
    late_window = torch.exp(-30.0 * (t - 0.85) ** 2)
    ior = torch.as_tensor(IOR, dtype=theta.dtype, device=theta.device)
    yield_species1 = (theta * early_window).sum()
    yield_species2 = (theta * late_window).sum()
    productivity = ior * yield_species1 + (1.0 - ior) * yield_species2
    diversity_bonus = 9.0 * ior * (1.0 - ior)
    feed_cost = 0.15 * (theta ** 2).sum()
    return productivity + diversity_bonus - feed_cost

def optimize_policy(IOR_val, n_steps=300, lr=0.05):
    torch.manual_seed(42)
    fixed_ior = float(IOR_val)
    theta = torch.zeros(10, dtype=torch.get_default_dtype(), requires_grad=True)
    optim = torch.optim.LBFGS([theta], lr=lr, max_iter=1, history_size=20, line_search_fn="strong_wolfe")
    best_yield = float("-inf")
    best_theta = theta.detach().clone()

    for _ in range(n_steps):
        def closure():
            optim.zero_grad()
            loss = -simulate(theta, fixed_ior)
            loss.backward()
            return loss

        optim.step(closure)
        with torch.no_grad():
            theta.clamp_(0.0, 5.0)
            current_yield = float(simulate(theta, fixed_ior).item())
            if current_yield > best_yield:
                best_yield = current_yield
                best_theta = theta.detach().clone()

    return best_theta.cpu().numpy(), float(best_yield)


def fit_gp(train_X, train_Y):
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model, model.likelihood


def run_bo(n_init=5, n_iter=25):
    bounds = torch.tensor([[0.01], [0.99]], dtype=torch.get_default_dtype())
    all_iors = []
    all_yields = []
    all_thetas = []
    gp_model = None
    likelihood = None

    for _ in range(n_init):
        ior_val = float((0.01 + 0.98 * torch.rand(1, dtype=torch.get_default_dtype())).item())
        theta_star, yield_star = optimize_policy(ior_val)
        all_iors.append(ior_val)
        all_yields.append(yield_star)
        all_thetas.append(theta_star)
        print(f"BO iter {len(all_iors)}: IOR={ior_val:.4f}, Yield={yield_star:.4f}")

    while len(all_iors) < n_iter:
        train_X = torch.tensor(all_iors, dtype=torch.get_default_dtype()).unsqueeze(-1)
        train_Y = torch.tensor(all_yields, dtype=torch.get_default_dtype()).unsqueeze(-1)

        try:
            gp_model, likelihood = fit_gp(train_X, train_Y)
            ei = ExpectedImprovement(gp_model, best_f=train_Y.max())
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=200,
            )
            ior_val = float(candidate.view(-1)[0].item())
        except Exception:
            ior_val = float((0.01 + 0.98 * torch.rand(1, dtype=torch.get_default_dtype())).item())

        theta_star, yield_star = optimize_policy(ior_val)
        all_iors.append(ior_val)
        all_yields.append(yield_star)
        all_thetas.append(theta_star)
        print(f"BO iter {len(all_iors)}: IOR={ior_val:.4f}, Yield={yield_star:.4f}")

    train_X = torch.tensor(all_iors, dtype=torch.get_default_dtype()).unsqueeze(-1)
    train_Y = torch.tensor(all_yields, dtype=torch.get_default_dtype()).unsqueeze(-1)
    try:
        gp_model, likelihood = fit_gp(train_X, train_Y)
    except Exception:
        gp_model, likelihood = None, None

    best_idx = int(np.argmax(all_yields))
    best_ior = float(all_iors[best_idx])
    best_yield = float(all_yields[best_idx])
    best_theta = all_thetas[best_idx]
    return best_ior, best_yield, best_theta, all_iors, all_yields, gp_model, likelihood


def plot_results(best_IOR, best_yield, best_theta, all_IORs, all_yields, gp_model, likelihood):
    grid = torch.linspace(0.0, 1.0, 200, dtype=torch.get_default_dtype()).unsqueeze(-1)

    if gp_model is not None:
        gp_model.eval()
        if likelihood is not None:
            likelihood.eval()
        with torch.no_grad():
            posterior = gp_model.posterior(grid)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
    else:
        sorted_pairs = sorted(zip(all_IORs, all_yields), key=lambda pair: pair[0])
        sorted_x = np.array([pair[0] for pair in sorted_pairs], dtype=float)
        sorted_y = np.array([pair[1] for pair in sorted_pairs], dtype=float)
        x_grid = grid.squeeze(-1).cpu().numpy()
        mean = np.interp(x_grid, sorted_x, sorted_y)
        base_std = np.std(sorted_y) if len(sorted_y) > 1 else 0.1
        std = np.full_like(mean, 0.2 * base_std)

    x_grid = grid.squeeze(-1).cpu().numpy()
    lower = mean - 1.645 * std
    upper = mean + 1.645 * std

    plt.figure(figsize=(9, 5.5))
    plt.plot(x_grid, mean, color="blue", linewidth=2, label="GP Posterior Mean")
    plt.fill_between(x_grid, lower, upper, color="blue", alpha=0.2, label="90% Confidence Interval")
    plt.scatter(all_IORs, all_yields, color="red", s=40, label="Observed Points", zorder=3)
    plt.axvline(best_IOR, color="green", linestyle="--", linewidth=2, label="Optimal IOR")
    plt.title("Bayesian Optimization: IOR vs Yield (GP Posterior)")
    plt.xlabel("Initial Inoculation Ratio (IOR)")
    plt.ylabel("Predicted Yield")
    plt.xlim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bo_posterior.png", dpi=150)
    plt.close()

    intervals = np.arange(1, 11)
    theta_vals = np.asarray(best_theta, dtype=float)
    mean_feed = float(theta_vals.mean())

    plt.figure(figsize=(10, 5.5))
    plt.bar(intervals, theta_vals, color="steelblue")
    plt.axhline(mean_feed, color="red", linestyle="--", linewidth=1.8, label="Mean feed rate")
    plt.title(f"Optimal Feeding Policy at IOR={best_IOR:.4f}")
    plt.xlabel("Time Interval")
    plt.ylabel("Feed Rate (g/L/h)")
    plt.xticks(intervals, [f"Time Interval {i}" for i in intervals], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_policy.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    best_IOR, best_yield, best_theta, all_IORs, all_yields, gp_model, likelihood = run_bo(n_init=5, n_iter=25)
    plot_results(best_IOR, best_yield, best_theta, all_IORs, all_yields, gp_model, likelihood)
    rounded_theta = [round(float(value), 4) for value in np.asarray(best_theta).tolist()]
    print(f"Best IOR: {best_IOR:.4f} | Best Yield: {best_yield:.4f}")
    print(f"Optimal policy theta*: {rounded_theta}")
    print("Plots saved: bo_posterior.png, optimal_policy.png")
