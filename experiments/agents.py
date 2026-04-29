"""Construction d'agents pour les expériences.

Chaque fonction renvoie une lambda Board -> Move utilisable directement
par ``domineering.tournament``.
"""
from __future__ import annotations

from functools import partial

from domineering.algorithms import (
    flat_best_move,
    ucb_best_move,
    uct_best_move,
    rave_best_move,
    grave_best_move,
    nmcs_best_move,
    nrpa_best_move,
    puct_best_move,
    sh_best_move,
    ppatcs_best_move,
    mast_best_move,
    shot_best_move,
    gnrpa_best_move,
    nrpa2p_best_move,
)
from domineering.optimizations import biased_playout, epsilon_greedy_playout


def make_random_agent():
    import random
    from domineering.board import Board, VERTICAL

    def agent(b):
        return b.legal_moves()[random.randrange(len(b.legal_moves()))]

    return agent


def make_flat(budget=200):
    return lambda b: flat_best_move(b, budget)


def make_ucb(budget=200, c=0.4):
    return lambda b: ucb_best_move(b, budget, c=c)


def make_uct(budget=200, c=0.4, heavy=None):
    if heavy is None:
        return lambda b: uct_best_move(b, budget, c=c)
    elif heavy == "biased":
        return lambda b: uct_best_move(b, budget, c=c, playout_fn=biased_playout)
    elif heavy == "epsilon":
        return lambda b: uct_best_move(b, budget, c=c, playout_fn=epsilon_greedy_playout)
    raise ValueError(heavy)


def make_rave(budget=200, heavy=None):
    if heavy is None:
        return lambda b: rave_best_move(b, budget)
    return lambda b: rave_best_move(b, budget, playout_strategy=heavy)


def make_grave(budget=200, ref_min=50, heavy=None):
    if heavy is None:
        return lambda b: grave_best_move(b, budget, ref_min=ref_min)
    return lambda b: grave_best_move(b, budget, ref_min=ref_min, playout_strategy=heavy)


def make_nmcs(level=1, discounted=False):
    return lambda b: nmcs_best_move(b, level=level, discounted=discounted)


def make_nrpa(budget=64, level=1):
    return lambda b: nrpa_best_move(b, budget=budget, level=level)


def make_puct(budget=200, c_puct=1.5):
    return lambda b: puct_best_move(b, budget, c_puct=c_puct)


def make_sh(budget=200):
    return lambda b: sh_best_move(b, budget)


def make_ppatcs(budget=200, level=2, alpha=1.0, feature_set="simple"):
    return lambda b: ppatcs_best_move(b, budget=budget, level=level,
                                       alpha=alpha, feature_set=feature_set)


def make_mast(budget=200, tau=1.0):
    return lambda b: mast_best_move(b, budget, tau=tau)


def make_shot(budget=200, c=0.4):
    return lambda b: shot_best_move(b, budget, c=c)


def make_gnrpa(budget=200, level=2, alpha=1.0, beta_scale=1.0):
    return lambda b: gnrpa_best_move(b, budget=budget, level=level,
                                     alpha=alpha, beta_scale=beta_scale)


def make_nrpa2p(budget=200, level=2, alpha=1.0):
    return lambda b: nrpa2p_best_move(b, budget=budget, level=level, alpha=alpha)
