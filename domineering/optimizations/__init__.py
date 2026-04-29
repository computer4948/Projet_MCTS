"""Optimisations (heavy playouts, heuristiques, priors)."""

from .heuristics import (
    move_advantage,
    move_advantage_prior,
    biased_playout,
    epsilon_greedy_playout,
)

__all__ = [
    "move_advantage",
    "move_advantage_prior",
    "biased_playout",
    "epsilon_greedy_playout",
]
