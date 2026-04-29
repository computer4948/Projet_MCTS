"""UCT (Upper Confidence bounds applied to Trees) avec table de transposition.

Implémentation directement transposée du squelette pédagogique du cours
(notebook Breakthrough), adaptée à Domineering.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

from ..board import Board, Move, VERTICAL


Entry = list  # [n_total, [n_a], [sum_q]]


def _new_entry(size: int) -> Entry:
    return [0, [0] * size, [0.0] * size]


def _default_playout(board: Board) -> float:
    return board.playout()


def uct_descent(
    board: Board,
    table: Dict[int, Entry],
    c: float,
    playout_fn: Callable[[Board], float],
) -> float:
    if board.terminal():
        return board.score()
    entry = table.get(board.h)
    if entry is not None:
        moves = board.legal_moves()
        n_total = entry[0]
        visits = entry[1]
        sums = entry[2]
        best = 0
        best_val = -1.0
        for i in range(len(moves)):
            ni = visits[i]
            if ni == 0:
                val = 1e9
            else:
                q = sums[i] / ni
                if board.turn != VERTICAL:
                    q = 1.0 - q
                val = q + c * math.sqrt(math.log(max(1, n_total)) / ni)
            if val > best_val:
                best_val = val
                best = i
        board.play(moves[best])
        res = uct_descent(board, table, c, playout_fn)
        entry[0] += 1
        entry[1][best] += 1
        entry[2][best] += res
        return res
    moves = board.legal_moves()
    table[board.h] = _new_entry(len(moves))
    return playout_fn(board)


def uct_best_move(
    board: Board,
    budget: int,
    c: float = 0.4,
    playout_fn: Optional[Callable[[Board], float]] = None,
) -> Move:
    table: Dict[int, Entry] = {}
    pf = playout_fn or _default_playout
    for _ in range(budget):
        b = board.copy()
        uct_descent(b, table, c, pf)
    moves = board.legal_moves()
    entry = table.get(board.h)
    if entry is None:
        return moves[0]
    visits = entry[1]
    return moves[max(range(len(moves)), key=lambda i: visits[i])]
