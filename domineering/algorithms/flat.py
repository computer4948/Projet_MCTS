"""Flat Monte Carlo.

On distribue les playouts uniformément entre les coups légaux du joueur
courant et l'on choisit celui qui maximise la valeur empirique du joueur.
"""
from __future__ import annotations

from ..board import Board, Move, VERTICAL


def flat_best_move(board: Board, budget: int, playout=None) -> Move:
    moves = board.legal_moves()
    if not moves:
        raise ValueError("Aucun coup légal")
    n_per_move = max(1, budget // len(moves))
    best_score = -1.0
    best_move = moves[0]
    me = board.turn
    for m in moves:
        s = 0.0
        for _ in range(n_per_move):
            b = board.copy()
            b.play(m)
            r = playout(b) if playout is not None else b.playout()
            if me == VERTICAL:
                s += r
            else:
                s += 1.0 - r
        if s > best_score:
            best_score = s
            best_move = m
    return best_move
