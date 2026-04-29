"""MAST - Move-Average Sampling Technique.

Finnsson \\& Bjornsson, *Simulation-Based Approach to General Game
Playing*, AAAI 2008. Le cours liste MAST parmi les politiques de
playout en ligne (à côté de PPA et de Pool-RAVE).

L'idée : maintenir, pour chaque code de coup, la moyenne du résultat
des playouts dans lesquels ce coup est apparu. Pendant les playouts on
échantillonne softmax sur ces moyennes (température $\\tau$).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np

from ..board import Board, Move, MaxCodeMoves, VERTICAL


class MASTStats:
    """Statistiques globales par code de coup."""

    __slots__ = ("sums", "counts")

    def __init__(self) -> None:
        self.sums = [0.0] * MaxCodeMoves
        self.counts = [0] * MaxCodeMoves

    def mean(self, code: int) -> float:
        n = self.counts[code]
        return 0.5 if n == 0 else self.sums[code] / n

    def update(self, codes: List[int], result: float) -> None:
        for c in codes:
            self.sums[c] += result
            self.counts[c] += 1


def mast_playout(
    board: Board,
    stats: MASTStats,
    tau: float = 1.0,
) -> float:
    """Playout MAST : softmax sur la moyenne par code de coup.
    Renvoie le score (POV VERTICAL) et met à jour les stats."""
    codes_played: List[int] = []
    while True:
        moves = board.legal_moves()
        if not moves:
            score = 0.0 if board.turn == VERTICAL else 1.0
            stats.update(codes_played, score)
            return score
        if len(moves) == 1:
            board.play(moves[0])
            codes_played.append(moves[0].code())
            continue
        means = np.array([stats.mean(m.code()) for m in moves])
        # Inverse pour Horizontal qui veut minimiser le score POV V
        if board.turn != VERTICAL:
            means = 1.0 - means
        logits = means / max(1e-9, tau)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(np.random.choice(len(moves), p=probs))
        codes_played.append(moves[idx].code())
        board.play(moves[idx])


def mast_best_move(
    board: Board,
    budget: int,
    tau: float = 1.0,
) -> Move:
    """Flat-MAST : on alloue le budget en playouts MAST, on choisit le
    coup dont la moyenne empirique est la meilleure du POV joueur courant.
    """
    stats = MASTStats()
    moves = board.legal_moves()
    if len(moves) == 1:
        return moves[0]
    me = board.turn
    visits = [0] * len(moves)
    sums = [0.0] * len(moves)
    n_per = max(1, budget // len(moves))
    for i, m in enumerate(moves):
        for _ in range(n_per):
            b = board.copy()
            b.play(m)
            r = mast_playout(b, stats, tau=tau)
            if me != VERTICAL:
                r = 1.0 - r
            sums[i] += r
            visits[i] += 1
    best = max(range(len(moves)), key=lambda i: sums[i] / max(1, visits[i]))
    return moves[best]
