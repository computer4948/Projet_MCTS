"""Statistiques d'inférence pour les expériences.

Inclut l'intervalle de confiance binomial de Wilson (à préférer à
Wald pour de petits échantillons et $p$ proche de 0 ou 1).
"""
from __future__ import annotations

import math
from typing import Tuple


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Renvoie ``(p_hat, lo, hi)`` où ``[lo, hi]`` est l'intervalle de
    confiance de Wilson au niveau correspondant à ``z`` (1.96 = 95 %).

    Référence : Wilson, E.B. (1927).
    """
    if n == 0:
        return 0.5, 0.0, 1.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def aggregate_winrates(wins_list, n_list, z: float = 1.96):
    """Moyenne pondérée de winrates avec CI sur la moyenne.

    Utilisé pour combiner plusieurs seeds : on traite tous les jeux
    comme un seul échantillon agrégé.
    """
    total_w = sum(wins_list)
    total_n = sum(n_list)
    return wilson_ci(total_w, total_n, z)
