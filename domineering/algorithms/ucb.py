"""Bandit UCB1 à la racine (sans arbre).

Variante "racine seulement" : on alloue les playouts selon UCB1 entre les
coups directs ; aucun arbre n'est construit, la suite de la partie reste
des playouts aléatoires.
"""
from __future__ import annotations

import math

from ..board import Board, Move, VERTICAL


def ucb_best_move(board: Board, budget: int, c: float = 0.4, playout=None) -> Move:
    moves = board.legal_moves()
    if not moves:
        raise ValueError("Aucun coup légal")
    n = len(moves)
    sums = [0.0] * n
    visits = [0] * n
    me = board.turn
    INF = float("inf")
    for i in range(budget):
        # Forcer une visite par bras avant d'utiliser UCB
        best = 0
        best_val = -INF
        for k in range(n):
            if visits[k] == 0:
                best = k
                break
            mean = sums[k] / visits[k]
            ucb = mean + c * math.sqrt(math.log(i + 1) / visits[k])
            if ucb > best_val:
                best_val = ucb
                best = k
        else:
            pass
        b = board.copy()
        b.play(moves[best])
        r = playout(b) if playout is not None else b.playout()
        if me != VERTICAL:
            r = 1.0 - r
        sums[best] += r
        visits[best] += 1
    # Robust child : on retourne le coup le plus visité.
    best_idx = max(range(n), key=lambda k: visits[k])
    return moves[best_idx]
