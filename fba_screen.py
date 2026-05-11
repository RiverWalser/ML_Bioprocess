"""
fba_screen.py
-------------
Stage 3

Receives a CommunityModel from Stage 2 (gem_adder) and evaluates
whether the species pair is worth forwarding to Stage 4 (the
differentiable JAX/Diffrax simulator).

Authors: Owen Lee + Jiayi Fu!

Key fixes vs original:
  - _run_cooperative_fba now runs ONE LP with all floors active
    simultaneously, instead of N separate LPs (one per species).
  - Solo growth is recomputed under DEFAULT_CONDITIONS (same medium as
    the community run) so tradeoff fractions are meaningful ratios.
  - Naive FBA typo (bconditions) fixed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import cobra

from gem_adder import CommunityModel


# ─────────────────────────────────────────────────────────────────────────────
# 1. ScreenResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenResult:
    """
    Output of fba_screen().
    ----------
    passed : bool
        Whether the pair passes all screening thresholds.
    species : list[str]
        Species prefixes in the community.
    naive_growth : dict[str, float]
        Per-species growth under naive weighted-sum FBA.
        Degenerate — included for reference only.
    tradeoff_growth : dict[str, float]
        Per-species growth under cooperative-tradeoff FBA (one LP).
    solo_growth : dict[str, float]
        Maximum solo growth for each species under DEFAULT_CONDITIONS.
        Computed in fba_screen — NOT the rich-medium value from gem_adder.
    tradeoff_fractions : dict[str, float]
        tradeoff_growth[sp] / solo_growth[sp]
        Fraction of solo potential each species achieves in community.
        1.0 = no sacrifice. Both numerator and denominator are in the
        same medium so this ratio is meaningful.
    community_score : float
        Mean of tradeoff growth rates (h⁻¹). Primary ranking metric.
    shared_metabolites : list[str]
        Extracellular metabolites coupling the two species.
    failure_reasons : list[str]
        Human-readable reasons the pair was rejected (empty if passed).
    conditions : dict
        The substrate conditions used for screening.
    """
    passed:               bool
    species:              list[str]
    naive_growth:         dict[str, float] = field(default_factory=dict)
    tradeoff_growth:      dict[str, float] = field(default_factory=dict)
    solo_growth:          dict[str, float] = field(default_factory=dict)
    tradeoff_fractions:   dict[str, float] = field(default_factory=dict)
    community_score:      float            = 0.0
    shared_metabolites:   list[str]        = field(default_factory=list)
    failure_reasons:      list[str]        = field(default_factory=list)
    conditions:           dict             = field(default_factory=dict)

    def summary(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"  ScreenResult: {' + '.join(self.species)}  {status}")
        print(f"  Community score (tradeoff) : {self.community_score:.4f} h⁻¹")
        print(f"\n  Per-species (cooperative tradeoff FBA):")
        for sp in self.species:
            solo = self.solo_growth.get(sp, 0.0)
            co   = self.tradeoff_growth.get(sp, 0.0)
            frac = self.tradeoff_fractions.get(sp, 0.0)
            print(f"    {sp:15s}  solo={solo:.4f}  community={co:.4f}  "
                  f"({frac*100:.1f}% of solo max)")
        if not self.passed:
            print(f"\n  Rejection reasons:")
            for r in self.failure_reasons:
                print(f"    • {r}")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Substrate condition helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches H&H validation case: glucose + xylose aerobic batch
DEFAULT_CONDITIONS = {
    "glc__D_e": -10.0,   # glucose uptake bound (mmol/gDW/h)
    "xyl__D_e":  -5.0,   # xylose
    "o2_e":     -20.0,   # oxygen (aerobic)
    "nh4_e":    -10.0,   # nitrogen source
    "pi_e":     -10.0,   # phosphate
}

def _apply_conditions(model: cobra.Model, conditions: dict[str, float]) -> None:
    """Set exchange reaction lower bounds to simulate substrate availability."""
    for met_base, lb in conditions.items():
        ex_id = f"EX_{met_base}"
        if ex_id in model.reactions:
            model.reactions.get_by_id(ex_id).lower_bound = lb


# ─────────────────────────────────────────────────────────────────────────────
# 3. Naive FBA (diagnostic only)
# ─────────────────────────────────────────────────────────────────────────────

def _run_naive_fba(cm: CommunityModel,
                   conditions: dict[str, float]) -> dict[str, float]:
    """
    Run standard weighted-sum FBA on the community model.
    Known to produce a degenerate result (one species starved).
    Included for comparison only — never forwarded to Stage 4.
    """
    with cm.cobra_model:
        _apply_conditions(cm.cobra_model, conditions)
        sol = cm.cobra_model.optimize()
        if sol.status != "optimal":
            return {sp: 0.0 for sp in cm.species}
        return cm.get_species_growth(sol)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 Solo growth under screening conditions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_solo_growth(cm: CommunityModel,
                          conditions: dict[str, float]) -> dict[str, float]:
    """
    Compute the maximum growth rate of each species individually,
    under the same substrate conditions used for community screening.

    This is the correct denominator for tradeoff fractions — both
    community growth (numerator) and solo growth (denominator) are now
    in the same medium, so the ratio is a meaningful comparison.

    Method: for each species, zero out all other species' biomass
    reaction upper bounds so they cannot grow, then run FBA.
    """
    solo_growth = {}

    for sp in cm.species:
        bm_id = cm.biomass_reaction_ids.get(sp)
        if bm_id is None:
            solo_growth[sp] = 0.0
            continue

        with cm.cobra_model as model:
            _apply_conditions(model, conditions)

            # Silence all other species so this one is effectively alone
            for other_sp in cm.species:
                if other_sp == sp:
                    continue
                other_bm_id = cm.biomass_reaction_ids.get(other_sp)
                if other_bm_id:
                    model.reactions.get_by_id(other_bm_id).upper_bound = 0.0

            sol = model.optimize()
            solo_growth[sp] = (
                sol.objective_value if sol.status == "optimal" else 0.0
            )

    return solo_growth


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cooperative tradeoff FBA
# ─────────────────────────────────────────────────────────────────────────────

def _run_cooperative_fba(cm: CommunityModel,
                          conditions: dict[str, float],
                          mu: float = 0.5,
                          solo_growth: Optional[dict[str, float]] = None
                          ) -> dict[str, float]:
    """
    True cooperative tradeoff FBA — ONE LP with all floors active.

    All species floors are set before calling optimize() once.
    Both species must satisfy mu * solo_max simultaneously in the same
    flux distribution. This is the correct implementation — previously
    we ran N separate LPs (one per species) which produced growth rates
    from inconsistent flux distributions that couldn't both be true at
    the same time.

    Parameters
    ----------
    cm : CommunityModel
    conditions : dict
        Substrate bounds (same format as DEFAULT_CONDITIONS).
    mu : float
        Cooperative tradeoff fraction. Each species must grow at
        >= mu * solo_max. Default 0.5.
    solo_growth : dict, optional
        Pre-computed solo growth rates under screening conditions.
        If None, falls back to cm.solo_growth_rates (placeholder zeros).
    """
    if solo_growth is None:
        solo_growth = cm.solo_growth_rates

    tradeoff_growth = {}
    old_lbs = {}

    with cm.cobra_model as model:
        _apply_conditions(model, conditions)

        # Step 1: set ALL floors first — do not solve yet
        for sp in cm.species:
            bm_id = cm.biomass_reaction_ids.get(sp)
            if bm_id is None:
                continue
            bm_rxn = model.reactions.get_by_id(bm_id)
            old_lbs[sp] = bm_rxn.lower_bound
            bm_rxn.lower_bound = mu * solo_growth.get(sp, 0.0)

        # Step 2: one solve with ALL floors active simultaneously
        sol = model.optimize()

        if sol.status == "optimal":
            tradeoff_growth = {
                sp: sol.fluxes.get(cm.biomass_reaction_ids[sp], 0.0)
                for sp in cm.species
                if cm.biomass_reaction_ids.get(sp) is not None
            }
        else:
            warnings.warn(
                f"Cooperative FBA infeasible at mu={mu}. Try lowering mu."
            )
            tradeoff_growth = {sp: 0.0 for sp in cm.species}

        # Step 3: restore all lower bounds
        for sp, lb in old_lbs.items():
            bm_id = cm.biomass_reaction_ids.get(sp)
            if bm_id:
                model.reactions.get_by_id(bm_id).lower_bound = lb

    return tradeoff_growth


# ─────────────────────────────────────────────────────────────────────────────
# 5. Screening thresholds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreeningThresholds:
    """
    Configurable pass/fail criteria.
    Defaults are starting points — empirical validation needed.
    mu=0.5 backed by MICOM gut microbiome literature (Heinken et al. 2021)
    but not yet validated for industrial bioprocess co-culture.
    min_tradeoff_fraction and min_community_score are not literature-backed.
    """
    min_tradeoff_fraction: float = 0.10   # each species >= 10% of solo max
    min_community_score:   float = 0.05   # mean tradeoff growth >= 0.05 h⁻¹
    min_shared_metabolites: int  = 1      # at least 1 shared metabolite
    mu:                    float = 0.50   # cooperative tradeoff parameter


DEFAULT_THRESHOLDS = ScreeningThresholds()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main screening function (public API)
# ─────────────────────────────────────────────────────────────────────────────

def fba_screen(cm: CommunityModel,
               conditions: Optional[dict[str, float]] = None,
               thresholds: Optional[ScreeningThresholds] = None,
               verbose: bool = True) -> ScreenResult:
    """
    Stage 3 FBA screener.

    Parameters
    ----------
    cm : CommunityModel
        Output of gem_adder().
    conditions : dict, optional
        Substrate availability. Defaults to DEFAULT_CONDITIONS.
    thresholds : ScreeningThresholds, optional
        Pass/fail criteria. Defaults to DEFAULT_THRESHOLDS.
    verbose : bool
        Print summary after screening.

    Returns
    -------
    ScreenResult
        passed=True → forward to Stage 4.
        passed=False → drop this pair.
    """
    if conditions is None:
        conditions = DEFAULT_CONDITIONS
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    failure_reasons = []

    # ---- check 1: shared metabolites ----
    if len(cm.shared_metabolites) < thresholds.min_shared_metabolites:
        failure_reasons.append(
            f"Only {len(cm.shared_metabolites)} shared extracellular "
            f"metabolites (need ≥ {thresholds.min_shared_metabolites})"
        )
        return ScreenResult(
            passed             = False,
            species            = cm.species,
            shared_metabolites = cm.shared_metabolites,
            failure_reasons    = failure_reasons,
            conditions         = conditions,
        )

    # ---- compute solo growth under screening conditions (correct denominator) ----
    solo_growth = _compute_solo_growth(cm, conditions)

    # ---- run naive FBA (degenerate, for reference) ----
    naive_growth = _run_naive_fba(cm, conditions)

    # ---- run cooperative tradeoff FBA (one LP, both floors active) ----
    tradeoff_growth = _run_cooperative_fba(cm, conditions, thresholds.mu, solo_growth)

    # ---- compute community score ----
    n = len(cm.species)
    community_score = sum(tradeoff_growth.values()) / n if n > 0 else 0.0

    # ---- compute per-species tradeoff fractions ----
    # Both numerator and denominator now in the same medium — ratio is meaningful
    tradeoff_fractions = {}
    for sp in cm.species:
        solo = solo_growth.get(sp, 0.0)
        co   = tradeoff_growth.get(sp, 0.0)
        frac = (co / solo) if solo > 1e-9 else 0.0
        tradeoff_fractions[sp] = frac

    # ---- check 2: per-species minimum fraction ----
    for sp, frac in tradeoff_fractions.items():
        if frac < thresholds.min_tradeoff_fraction:
            failure_reasons.append(
                f"{sp} achieves only {frac*100:.1f}% of solo max "
                f"(need ≥ {thresholds.min_tradeoff_fraction*100:.0f}%)"
            )

    # ---- check 3: community score floor ----
    if community_score < thresholds.min_community_score:
        failure_reasons.append(
            f"Community score {community_score:.4f} h⁻¹ below minimum "
            f"{thresholds.min_community_score} h⁻¹"
        )

    passed = len(failure_reasons) == 0

    result = ScreenResult(
        passed             = passed,
        species            = cm.species,
        naive_growth       = naive_growth,
        tradeoff_growth    = tradeoff_growth,
        solo_growth        = solo_growth,
        tradeoff_fractions = tradeoff_fractions,
        community_score    = community_score,
        shared_metabolites = cm.shared_metabolites,
        failure_reasons    = failure_reasons,
        conditions         = conditions,
    )

    if verbose:
        result.summary()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Batch screening (multiple pairs)
# ─────────────────────────────────────────────────────────────────────────────

def screen_pairs(pairs: list[tuple[str, str]],
                 conditions: Optional[dict[str, float]] = None,
                 thresholds: Optional[ScreeningThresholds] = None
                 ) -> list[ScreenResult]:
    """
    Screen multiple species pairs and return ranked ScreenResults.
    Results sorted by community_score descending.
    """
    from gem_adder import gem_adder

    results = []
    for src_a, src_b in pairs:
        print(f"\n── Screening {src_a} + {src_b} ──")
        try:
            cm = gem_adder(src_a, src_b)
            result = fba_screen(cm, conditions, thresholds, verbose=True)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(ScreenResult(
                passed          = False,
                species         = [src_a, src_b],
                failure_reasons = [f"Exception during loading/merging: {e}"],
                conditions      = conditions or {},
            ))

    results.sort(key=lambda r: r.community_score, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from gem_adder import gem_adder

    print("=== Stage 3 FBA Screen: E. coli + Salmonella ===\n")
    cm = gem_adder("textbook", "salmonella")

    result = fba_screen(cm)

    print(f"Forward to Stage 4? {result.passed}")
    print(f"Community score    : {result.community_score:.4f} h⁻¹")

    print("\n--- Naive FBA (degenerate, for reference) ---")
    for sp, gr in result.naive_growth.items():
        print(f"  {sp}: {gr:.4f} h⁻¹")

    print("\n--- Cooperative Tradeoff FBA (one LP — Stage 4 input) ---")
    for sp, gr in result.tradeoff_growth.items():
        solo = result.solo_growth.get(sp, 0.0)
        frac = result.tradeoff_fractions.get(sp, 0.0)
        print(f"  {sp}: {gr:.4f} h⁻¹  ({frac*100:.1f}% of solo max {solo:.4f})")
