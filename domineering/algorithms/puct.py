"""PUCT (Predictor + UCT), variante popularisée par AlphaGo Zero.

Score d'exploration : ``Q + c_puct * P * sqrt(N_total) / (1 + N_a)``.

Sans réseau de neurones, ``P`` est calculé via une heuristique dépendante
du domaine (par défaut, l'heuristique ``move_advantage`` qui compte la
différence entre coups adverses bloqués et coups propres détruits).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

from ..board import Board, Move, VERTICAL
from ..optimizations.heuristics import move_advantage_prior

Entry = list  # [n_total, [n_a], [sum_q], [prior]]


def _new_entry(n: int, priors: List[float]) -> Entry:
    return [0, [0] * n, [0.0] * n, list(priors)]


def puct_descent(
    board: Board,
    table: Dict[int, Entry],
    c_puct: float,
    prior_fn: Callable[[Board, List[Move]], List[float]],
) -> float:
    if board.terminal():
        return board.score()
    entry = table.get(board.h)
    if entry is not None:
        moves = board.legal_moves()
        n_total = entry[0]
        visits = entry[1]
        sums = entry[2]
        priors = entry[3]
        sqrt_total = math.sqrt(max(1, n_total))
        best = 0
        best_val = -1e9
        for i in range(len(moves)):
            ni = visits[i]
            q = 0.0 if ni == 0 else sums[i] / ni
            if board.turn != VERTICAL:
                q = 1.0 - q
            u = c_puct * priors[i] * sqrt_total / (1 + ni)
            val = q + u
            if val > best_val:
                best_val = val
                best = i
        board.play(moves[best])
        res = puct_descent(board, table, c_puct, prior_fn)
        entry[0] += 1
        entry[1][best] += 1
        entry[2][best] += res
        return res
    moves = board.legal_moves()
    priors = prior_fn(board, moves)
    table[board.h] = _new_entry(len(moves), priors)
    return board.playout()


def puct_best_move(
    board: Board,
    budget: int,
    c_puct: float = 1.5,
    prior_fn: Callable[[Board, List[Move]], List[float]] = move_advantage_prior,
) -> Move:
    table: Dict[int, Entry] = {}
    for _ in range(budget):
        b = board.copy()
        puct_descent(b, table, c_puct, prior_fn)
    moves = board.legal_moves()
    entry = table.get(board.h)
    if entry is None:
        return moves[0]
    visits = entry[1]
    return moves[max(range(len(moves)), key=lambda i: visits[i])]
