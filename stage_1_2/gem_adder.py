"""
gem_adder.py
------------
Stage 1 + 2 of the Bioprocess Consortium Optimization pipeline.

Loads GEMs from BiGG (via COBRApy), merges them into a community model
with a shared extracellular medium compartment, and exposes a clean
interface for Stage 3 (FBA screening).

Architecture decisions (aligned with River's Stage 4 presentation):
- Shared medium is a flat namespace (matches Wilson's operator-split model)
- Namespace prefixing prevents reaction/metabolite ID collisions
- Exchange reactions are preserved and standardised for Stage 3
- CommunityModel dataclass is the contract between Stage 2 and Stage 3

Authors: Jiayi Fu + Owen Lee  |  YCRG Smart Therapeutics Track
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import cobra
from cobra import Model, Reaction, Metabolite
from cobra.io import read_sbml_model

import warnings
warnings.filterwarnings("ignore", message="Ignoring reaction")

# ── COBRApy ships the H&H validation species; use these IDs when loading ──
BUILTIN_MODELS = {
    "e_coli_core": "textbook",          # COBRApy builtin alias
    "salmonella":  "salmonella",        # COBRApy builtin alias
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gem(source: str) -> cobra.Model:
    """
    Load a GEM from:
      - a file path to an SBML (.xml / .xml.gz) file
      - a BiGG model ID string (downloads from bigg.ucsd.edu)
      - a COBRApy builtin alias ('textbook', 'salmonella')

    Returns a cobra.Model with its id set to the source string.
    """
    # COBRApy builtins
    if source in ("textbook", "e_coli_core"):
        model = cobra.io.load_model("textbook")
        model.id = "e_coli_core"
        return model
    if source == "salmonella":
        model = cobra.io.load_model("salmonella")
        model.id = "salmonella"
        return model

    # SBML file path
    if source.endswith(".xml") or source.endswith(".xml.gz"):
        model = read_sbml_model(source)
        return model

    # BiGG HTTP download
    import urllib.request, gzip, tempfile, os
    url = f"http://bigg.ucsd.edu/static/models/{source}.xml.gz"
    print(f"[load_gem] Downloading {url} …")
    with tempfile.NamedTemporaryFile(suffix=".xml.gz", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        tmp_path = tmp.name
    try:
        model = read_sbml_model(tmp_path)
    finally:
        os.unlink(tmp_path)
    model.id = source
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared-medium detection
# ─────────────────────────────────────────────────────────────────────────────

_EX_PATTERN = re.compile(r"^EX_(.+)_e$")   # standard COBRApy exchange ID

def _extracellular_metabolite_ids(model: cobra.Model) -> set[str]:
    """
    Return the base IDs of all metabolites present in the extracellular
    compartment ('e') of a model.

    We strip the trailing compartment tag so the IDs are comparable
    across models regardless of internal naming conventions.
    """
    ids = set()
    for met in model.metabolites:
        if met.compartment == "e" or met.id.endswith("_e"):
            # normalise: strip trailing _e
            base = met.id[:-2] if met.id.endswith("_e") else met.id
            ids.add(base)
    return ids


def find_shared_metabolites(model_a: cobra.Model,
                             model_b: cobra.Model) -> list[str]:
    """
    Return base IDs of metabolites in the shared extracellular medium
    (present as exchange-accessible in both models).
    """
    shared = _extracellular_metabolite_ids(model_a) & \
             _extracellular_metabolite_ids(model_b)
    return sorted(shared)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Namespace helper
# ─────────────────────────────────────────────────────────────────────────────

def _safe_prefix(model_id: str) -> str:
    """Alphanumeric prefix derived from a model's ID (≤8 chars)."""
    clean = re.sub(r"[^a-zA-Z0-9]", "_", model_id)[:8]
    return clean.lower()


def _apply_namespace(model: cobra.Model,
                     prefix: str,
                     shared_ext_ids: set[str]) -> cobra.Model:
    """
    Return a *copy* of model with all internal IDs prefixed.

    Rules:
    - Shared extracellular metabolites keep their original IDs so they
      collapse into one shared pool after merging.
    - All intracellular metabolites, reactions, and genes get the prefix.
    - Exchange reactions for shared metabolites keep their original IDs.
    """
    m = model.copy()

    # -- metabolites --
    for met in m.metabolites:
        base = met.id[:-2] if met.id.endswith("_e") else met.id
        is_shared_ext = (met.id.endswith("_e") and base in shared_ext_ids)
        if not is_shared_ext:
            met.id = f"{prefix}__{met.id}"

    # -- reactions --
    for rxn in m.reactions:
        # Exchange reactions for shared metabolites: leave ID alone
        # so both models share the same EX_ handle in the merged model
        is_shared_exchange = (
            rxn.id.startswith("EX_") and
            any(rxn.id == f"EX_{sid}_e" for sid in shared_ext_ids)
        )
        if not is_shared_exchange:
            rxn.id = f"{prefix}__{rxn.id}"

    # -- genes --
    for gene in m.genes:
        gene.id = f"{prefix}__{gene.id}"

    m.id = prefix
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 4. Merger
# ─────────────────────────────────────────────────────────────────────────────

def _merge_models(namespaced_a: cobra.Model,
                  namespaced_b: cobra.Model,
                  shared_ext_ids: set[str],
                  weight_a: float = 0.5,
                  weight_b: float = 0.5) -> cobra.Model:
    """
    Merge two namespace-isolated models into a single cobra.Model.

    Shared extracellular metabolites are deduplicated — both models'
    exchange reactions point to the same metabolite object, which is
    the mechanistic coupling (resource competition / cross-feeding).

    The objective is set to a weighted sum of both biomass reactions,
    consistent with River's Stage 4 formulation.
    """
    community = cobra.Model("community")

    # ---- add model A ----
    community.add_reactions([rxn.copy() for rxn in namespaced_a.reactions])

    # ---- add model B (deduplicate shared extracellular metabolites) ----
    for rxn in namespaced_b.reactions:
        rxn_copy = rxn.copy()
        # remap metabolites that already exist in the community model
        new_stoich: dict[Metabolite, float] = {}
        for met, coeff in rxn_copy.metabolites.items():
            base = met.id[:-2] if met.id.endswith("_e") else met.id
            is_shared = met.id.endswith("_e") and base in shared_ext_ids
            if is_shared and met.id in community.metabolites:
                # reuse existing shared metabolite object
                existing = community.metabolites.get_by_id(met.id)
                new_stoich[existing] = coeff
            else:
                new_stoich[met] = coeff
        rxn_copy._metabolites = new_stoich
        community.add_reactions([rxn_copy])

    # ---- weighted biomass objective ----
    biomass_rxns = {}
    prefix_a = namespaced_a.id
    prefix_b = namespaced_b.id

    # find biomass reactions (heuristic: id contains 'biomass' or 'BIOMASS')
    def _find_biomass(model_id: str) -> Optional[Reaction]:
        for rxn in community.reactions:
            if rxn.id.startswith(model_id) and \
               "biomass" in rxn.id.lower():
                return rxn
        return None

    bm_a = _find_biomass(prefix_a)
    bm_b = _find_biomass(prefix_b)

    if bm_a and bm_b:
        community.objective = {bm_a: weight_a, bm_b: weight_b}
    elif bm_a:
        community.objective = bm_a
    elif bm_b:
        community.objective = bm_b
    else:
        warnings.warn("Could not identify biomass reactions; objective not set.")

    return community


# ─────────────────────────────────────────────────────────────────────────────
# 5. CommunityModel dataclass  (Stage 2 → Stage 3 contract)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommunityModel:
    """
    Output of gem_adder(). This is the data contract between Stage 2
    and Stage 3 (FBA screening).

    Attributes
    ----------
    cobra_model : cobra.Model
        The merged community model ready for FBA.
    species : list[str]
        Prefixes / IDs of the member species (same order as input).
    shared_metabolites : list[str]
        Base IDs of extracellular metabolites shared between species.
    biomass_reaction_ids : dict[str, str]
        Maps species prefix → biomass reaction ID in the merged model.
    solo_growth_rates : dict[str, float]
        Maximum solo growth rate of each species (used by Stage 3 for
        cooperative-tradeoff threshold computation).
    """
    cobra_model:         cobra.Model
    species:             list[str]
    shared_metabolites:  list[str]
    biomass_reaction_ids: dict[str, str]        = field(default_factory=dict)
    solo_growth_rates:   dict[str, float]       = field(default_factory=dict)

    # ---- convenience methods ----

    def summary(self) -> None:
        n_rxn = len(self.cobra_model.reactions)
        n_met = len(self.cobra_model.metabolites)
        print(f"\n{'='*55}")
        print(f"  CommunityModel: {' + '.join(self.species)}")
        print(f"  Reactions : {n_rxn}   Metabolites : {n_met}")
        print(f"  Shared extracellular metabolites ({len(self.shared_metabolites)}):")
        for sid in self.shared_metabolites[:8]:
            print(f"    {sid}_e")
        if len(self.shared_metabolites) > 8:
            print(f"    … and {len(self.shared_metabolites) - 8} more")
        print(f"\n  Solo growth rates:")
        for sp, gr in self.solo_growth_rates.items():
            print(f"    {sp}: {gr:.4f} h⁻¹")
        print(f"{'='*55}\n")

    def run_fba(self) -> cobra.Solution:
        """Run FBA on the merged model and return the solution object."""
        with self.cobra_model:
            sol = self.cobra_model.optimize()
        return sol

    def get_species_growth(self, solution: cobra.Solution) -> dict[str, float]:
        """Extract per-species growth rates from a solution."""
        rates = {}
        for sp, bm_id in self.biomass_reaction_ids.items():
            rates[sp] = solution.fluxes.get(bm_id, 0.0)
        return rates


# ─────────────────────────────────────────────────────────────────────────────
# 6. Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def gem_adder(source_a: str,
              source_b: str,
              weight_a: float = 0.5,
              weight_b: float = 0.5) -> CommunityModel:
    """
    Load two GEMs and merge them into a CommunityModel.

    Parameters
    ----------
    source_a, source_b : str
        Source strings accepted by load_gem() — file paths, BiGG IDs,
        or COBRApy builtin aliases.
    weight_a, weight_b : float
        Objective weights for each species' biomass (default 50/50).

    Returns
    -------
    CommunityModel
        Merged model ready for Stage 3 FBA screening.

    Example
    -------
    >>> cm = gem_adder("textbook", "salmonella")
    >>> cm.summary()
    >>> sol = cm.run_fba()
    """
    # 1. load
    model_a = load_gem(source_a)
    model_b = load_gem(source_b)

    prefix_a = _safe_prefix(model_a.id)
    prefix_b = _safe_prefix(model_b.id)

    # 2. compute solo growth rates (needed by Stage 3 cooperative tradeoff)
    with model_a:
        sol_a = model_a.optimize()
        solo_a = sol_a.objective_value if sol_a.status == "optimal" else 0.0

    with model_b:
        sol_b = model_b.optimize()
        solo_b = sol_b.objective_value if sol_b.status == "optimal" else 0.0

    # 3. find shared extracellular metabolites
    shared_ids = find_shared_metabolites(model_a, model_b)
    shared_set = set(shared_ids)

    # 4. apply namespaces
    ns_a = _apply_namespace(model_a, prefix_a, shared_set)
    ns_b = _apply_namespace(model_b, prefix_b, shared_set)

    # 5. merge
    community = _merge_models(ns_a, ns_b, shared_set, weight_a, weight_b)

    # 6. collect biomass reaction IDs
    bm_ids = {}
    for prefix in (prefix_a, prefix_b):
        for rxn in community.reactions:
            if rxn.id.startswith(prefix) and "biomass" in rxn.id.lower():
                bm_ids[prefix] = rxn.id
                break

    return CommunityModel(
        cobra_model         = community,
        species             = [prefix_a, prefix_b],
        shared_metabolites  = shared_ids,
        biomass_reaction_ids= bm_ids,
        solo_growth_rates   = {prefix_a: solo_a, prefix_b: solo_b},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading E. coli core + Salmonella …")
    cm = gem_adder("textbook", "salmonella")
    cm.summary()

    sol = cm.run_fba()
    print(f"Community FBA status : {sol.status}")
    print(f"Community objective  : {sol.objective_value:.4f}")
    per_species = cm.get_species_growth(sol)
    for sp, gr in per_species.items():
        print(f"  {sp}: {gr:.4f} h⁻¹")
