# -*- coding: utf-8 -*-
"""
§7.1 directional constraints: Barjamovic et al. Appendix A's 11 rules
("Durhumit is north of Kanes", etc.), expressed as a differentiable
hinge penalty in km, addable to the node-distance loss:

    L = L_node + gamma * L_constraint

Each rule contributes 0 once satisfied and grows linearly (in km) with
the violation -- see constraint_loss() below for the exact formula.
This is new functionality (not a bug fix): it changes what the model
is trained to do, so nothing in experiments/ calls this by default.
See experiments/ancient_lost_dyadic_constrained.py for the one place
it is actually wired in and run, as a new, additional result -- it
does not touch or replace the unconstrained ancient_lost_dyadic run.

NOTE on Mamma: Barjamovic et al.'s 11 rules include Mamma ("south of
Kanes"), but in this dataset Mamma is one of the 15 KNOWN cities (has
real coordinates), not one of the 10 actually-lost cities. So this
constraint can only ever bind during a LOO fold where Mamma is
fictitiously held out -- never during the final 10-lost-city
prediction, where Mamma's coordinates are fixed ground truth and never
predicted. Kept in CONSTRAINTS for fidelity to Appendix A; flagging
here because it's easy to misread "11 rules" as "11 of the 10 lost
cities."
"""

import torch

from . import data

# ('city', 'N'|'S'|'E'|'W') -- cities with two rules (Hahhum, Zalpa) appear twice.
CONSTRAINTS = [
    ('Durhumit', 'N'), ('Hahhum', 'S'), ('Hahhum', 'E'),
    ('Kuburnat', 'N'), ('Mamma', 'S'), ('Ninassa', 'W'),
    ('Purushaddum', 'W'), ('Sinahuttum', 'N'), ('Suppiluliya', 'N'),
    ('Tuhpiya', 'N'), ('Washaniya', 'W'),
    ('Zalpa', 'S'), ('Zalpa', 'E'),
]

CONSTRAINED_CITY_NAMES = sorted({city for city, _ in CONSTRAINTS})


def kanes_reference_point():
    """(lon, lat) for Kanes, read from data.py rather than restated as a
    literal, so it can never drift from the coordinate every other module
    uses."""
    known_coords, known_names = data.ancient_known_coords()
    idx = known_names.index('Kanes')
    lon, lat = known_coords[idx]
    return float(lon), float(lat)


def constraint_loss(pred_unscaled, name_to_node_idx, km_per_degree, cos_ref,
                     kanes_lon, kanes_lat, constraints=CONSTRAINTS):
    """pred_unscaled: (num_nodes, 2) tensor of [lon, lat] in degrees --
    i.e. already run through the Bug-4-fixed unscaling, absolute
    coordinates, not the [0,1]-scaled model output.
    name_to_node_idx: dict, city name -> row index into pred_unscaled.
        Only rules whose city is present are applied (so this also works
        mid-LOO-fold, when one known city's row is temporarily absent).
    Returns a scalar km penalty, 0 when every applicable rule is
    satisfied, averaged over the number of distinct cities actually
    constrained this call (paper's |V_c|)."""
    terms = []
    cities_applied = set()

    for city, direction in constraints:
        if city not in name_to_node_idx:
            continue
        lon = pred_unscaled[name_to_node_idx[city], 0]
        lat = pred_unscaled[name_to_node_idx[city], 1]
        cities_applied.add(city)

        if direction == 'N':
            terms.append(km_per_degree * torch.relu(kanes_lat - lat))
        elif direction == 'S':
            terms.append(km_per_degree * torch.relu(lat - kanes_lat))
        elif direction == 'E':
            terms.append(km_per_degree * cos_ref * torch.relu(kanes_lon - lon))
        elif direction == 'W':
            terms.append(km_per_degree * cos_ref * torch.relu(lon - kanes_lon))
        else:
            raise ValueError(f"unknown direction: {direction}")

    if not terms:
        return torch.zeros(1).sum()  # differentiable zero, same device/dtype family
    return torch.stack(terms).sum() / len(cities_applied)


def count_binding_constraints(pred_unscaled, name_to_node_idx, kanes_lon, kanes_lat,
                               constraints=CONSTRAINTS, tol_deg=0.0):
    """Diagnostic: which applicable rules are violated at these predicted
    coordinates (tol_deg as a small slack in degrees, default none). Used
    to tune gamma -- HANDOFF.md §7.1 says to raise gamma until this is
    empty, since Barjamovic et al. report none of their own constraints
    bind near their point estimates."""
    binding = []
    for city, direction in constraints:
        if city not in name_to_node_idx:
            continue
        lon = float(pred_unscaled[name_to_node_idx[city], 0])
        lat = float(pred_unscaled[name_to_node_idx[city], 1])
        if direction == 'N' and lat < kanes_lat - tol_deg:
            binding.append((city, direction))
        elif direction == 'S' and lat > kanes_lat + tol_deg:
            binding.append((city, direction))
        elif direction == 'E' and lon < kanes_lon - tol_deg:
            binding.append((city, direction))
        elif direction == 'W' and lon > kanes_lon + tol_deg:
            binding.append((city, direction))
    return binding
