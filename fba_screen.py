"""
fba_screen.py
-------------
Stage 3 

Receives a CommunityModel from Stage 2 (gem_adder) and evaluates
whether the species pair is worth forwarding to Stage 4 (The
differentiable JAX/Diffrax simulator).

Authors: Owen Lee + Jiayi Fu!
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import cobra
from cobra import Model

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
        Whether the pair passes all screening thresholds (the big test)
    species : list[str]
        Species prefixes in the community.
    naive_growth : dict[str, float]
        Per-species growth under naive weighted-sum FBA. This is just for reference, we use cooperative tradeoff FBA isntead. 
    tradeoff_growth : dict[str, float]
        Per-species growth under cooperative-tradeoff FBA.
    solo_growth : dict[str, float]
        Maximum solo growth for each species (from Stage 2 gem_adder.py)
    tradeoff_fractions : dict[str, float]
        tradeoff_growth[sp] / solo_growth[sp] — how much of solo max
        each species achieves in community, basically how much coexisting in community forces species to sacrifice growth. 1.0 = nothing sacrificed.
    community_score : float
        Weighted sum of tradeoff growth rates (primary ranking metric).
    shared_metabolites : list[str]
        Extracellular metabolites coupling the two species.
    failure_reasons : list[str]
        Reasons the pair was rejected (empty if passed).
    conditions : dict
        The substrate conditions used for screening.
    """
    passed:               bool
    species:              list[str]
    naive_growth:         dict[str, float]       = field(default_factory=dict)
    tradeoff_growth:      dict[str, float]       = field(default_factory=dict)
    solo_growth:          dict[str, float]       = field(default_factory=dict)
    tradeoff_fractions:   dict[str, float]       = field(default_factory=dict)
    community_score:      float                  = 0.0
    shared_metabolites:   list[str]              = field(default_factory=list)
    failure_reasons:      list[str]              = field(default_factory=list)
    conditions:           dict                   = field(default_factory=dict)

    def summary(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"  ScreenResult: {' + '.join(self.species)}  {status}")
        print(f"  Community score (tradeoff) : {self.community_score:.4f} h⁻¹")
        print(f"\n  Per-species (cooperative tradeoff FBA):")
        for sp in self.species:
            solo  = self.solo_growth.get(sp, 0.0)
            co    = self.tradeoff_growth.get(sp, 0.0)
            frac  = self.tradeoff_fractions.get(sp, 0.0)
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

# Default conditions matching the H&H validation case: glucose + xylose aerobic batch
DEFAULT_CONDITIONS = {
    "glc__D_e": -10.0,   # glucose uptake bound (mmol/gDW/h), negative = uptake
    "xyl__D_e": -5.0,    # xylose
    "o2_e":     -20.0,   # oxygen (aerobic)
    "nh4_e":    -10.0,   # nitrogen source
    "pi_e":     -10.0,   # phosphate
}

def _apply_conditions(model: cobra.Model, conditions: dict[str, float]) -> None:
    for met_base, lb in conditions.items():
        ex_id = f"EX_{met_base}"
        if ex_id in model.reactions:
            model.reactions.get_by_id(ex_id).lower_bound = lb


# ─────────────────────────────────────────────────────────────────────────────
# 3. Naive FBA 
# ─────────────────────────────────────────────────────────────────────────────

def _run_naive_fba(cm: CommunityModel, conditions: dict[str, float]) -> dict[str, float]:
    with cm.cobra_model:
        _apply_conditions(cm.cobra_model, conditions)
        sol = cm.cobra_model.optimize()
        if sol.status != "optimal":
            return {sp: 0.0 for sp in cm.species}
        return cm.get_species_growth(sol)

# ─────────────────────────────────────────────────────────────────────────────
# 3.5 Solo Growth 
# ─────────────────────────────────────────────────────────────────────────────

def _compute_solo_growth(cm: CommunityModel, conditions: dict[str, float]) -> dict[str, float]:

    solo_growth = {}

    for sp in cm.species:
        bm_id = cm.biomass_reaction_ids.get(sp)
        if bm_id is None:
            solo_growth[sp] = 0.0
            continue

        with cm.cobra_model as model:
            _apply_conditions(model, conditions)

            for other_sp in cm.species:
                if other_sp == sp:
                    continue
                other_bm_id = cm.biomass_reaction_ids.get(other_sp)
                if other_bm_id:
                    model.reactions.get_by_id(other_bm_id).upper_bound = 0.0

            sol = model.optimize()
            solo_growth[sp] = sol.objective_value if sol.status == "optimal" else 0.0

    return solo_growth

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cooperative tradeoff FBA  
# ─────────────────────────────────────────────────────────────────────────────
def _run_cooperative_fba(cm: CommunityModel, conditions: dict[str, float], mu: float = 0.5) -> dict[str, float]:
    tradeoff_growth = {}
    old_lbs = {}
        if solo_growth is None:
            solo_growth = solo_growth
    with cm.cobra_model as model:
        _apply_conditions(model, conditions)

        # Setting all the floors first
        for sp in cm.species:
            bm_id = cm.biomass_reaction_ids.get(sp)
            if bm_id is None:
                continue
            bm_rxn = model.reactions.get_by_id(bm_id)
            old_lbs[sp] = bm_rxn.lower_bound
            bm_rxn.lower_bound = mu * solo_growth.get(sp, 0.0)

        # Linear Program Solver
        sol = model.optimize()

        if sol.status == "optimal":
            tradeoff_growth = {
                sp: sol.fluxes.get(cm.biomass_reaction_ids[sp], 0.0)
                for sp in cm.species
                if cm.biomass_reaction_ids.get(sp) is not None
            }
        else:
            warnings.warn(f"Cooperative FBA infeasible at mu={mu}. Try lowering mu.")
            tradeoff_growth = {sp: 0.0 for sp in cm.species}

        # Restore all Lower boun ds
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
    # Minimum tradeoff fraction each species must achieve (0-1)
    min_tradeoff_fraction: float = 0.10

    # Minimum absolute community growth rate (weighted sum, h⁻¹)
    min_community_score: float = 0.05

    # Minimum number of shared extracellular metabolites
    min_shared_metabolites: int = 1

    # Cooperative tradeoff mu parameter (see _run_cooperative_fba)
    mu: float = 0.50


DEFAULT_THRESHOLDS = ScreeningThresholds()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main screening function  (public API)
# ─────────────────────────────────────────────────────────────────────────────

def fba_screen(cm: CommunityModel,
               conditions: Optional[dict[str, float]] = None,
               thresholds: Optional[ScreeningThresholds] = None,
               verbose: bool = True
               ) -> ScreenResult:
    """
    Parameters
    ----------
    cm : CommunityModel
        Output of gem_adder().
    conditions : dict, optional
        Substrate availability (exchange lower bounds).
        Defaults to DEFAULT_CONDITIONS (glucose + xylose aerobic batch).
    thresholds : ScreeningThresholds, optional
        Pass/fail criteria. Defaults to DEFAULT_THRESHOLDS.
    verbose : bool
        Print summary after screening.
    Returns
    -------
    ScreenResult
        Passed=True → forward to Stage 4.
        Passed=False → drop this pair.
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
        # Early return — can't run meaningful FBA
        return ScreenResult(
            passed             = False,
            species            = cm.species,
            shared_metabolites = cm.shared_metabolites,
            failure_reasons    = failure_reasons,
            conditions         = conditions,
        )

    # ---- compute solo growth under screening conditions (correct denominator) ----
    solo_growth = _compute_solo_growth(cm, conditions)

    # ---- run naive FBA ----
    naive_growth = _run_naive_fba(cm, conditions)

    # ---- run cooperative tradeoff FBA ----
    tradeoff_growth = _run_cooperative_fba(cm, conditions, thresholds.mu, solo_growth)

    # ---- compute community score ----
    # Weighted sum of tradeoff growth rates (equal weights for now)
    n = len(cm.species)
    community_score = sum(tradeoff_growth.values()) / n if n > 0 else 0.0

    # ---- compute per-species tradeoff fractions ----
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
# 7. Batch screening  (multiple pairs)
# ─────────────────────────────────────────────────────────────────────────────

def screen_pairs(pairs: list[tuple[str, str]],
                 conditions: Optional[dict[str, float]] = None,
                 thresholds: Optional[ScreeningThresholds] = None
                 ) -> list[ScreenResult]:

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
                passed         = False,
                species        = [src_a, src_b],
                failure_reasons= [f"Exception during loading/merging: {e}"],
                conditions     = conditions or {},
            ))

    # sort by community score descending
    results.sort(key=lambda r: r.community_score, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from gem_adder import gem_adder

    print("=== Stage 3 FBA Screen: E. coli + Salmonella ===\n")
    cm = gem_adder("textbook", "salmonella")

    # Default conditions (glucose + xylose aerobic)
    result = fba_screen(cm)

    print(f"Forward to Stage 4? {result.passed}")
    print(f"Community score    : {result.community_score:.4f} h⁻¹")

    # Show naive vs tradeoff comparison
    print("\n--- Naive FBA (degenerate, for reference) ---")
    for sp, gr in result.naive_growth.items():
        print(f"  {sp}: {gr:.4f} h⁻¹")

    print("\n--- Cooperative Tradeoff FBA (Stage 4 input) ---")
    for sp, gr in result.tradeoff_growth.items():
        solo = result.solo_growth.get(sp, 0.0)
        frac = result.tradeoff_fractions.get(sp, 0.0)
        print(f"  {sp}: {gr:.4f} h⁻¹  ({frac*100:.1f}% of solo max {solo:.4f})")
