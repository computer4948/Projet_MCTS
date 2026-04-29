"""NRPA (Nested Rollout Policy Adaptation).

Adaptation à un jeu à deux joueurs (Domineering) : la politique
softmax-paramétrique est partagée entre les deux camps et son score est
calculé du point de vue du joueur au trait au moment de l'appel.

Référence : Rosin, "Nested Rollout Policy Adaptation for Monte Carlo Tree
Search", IJCAI 2011.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from ..board import Board, Move, VERTICAL


def _random_move(board: Board, policy: Dict[int, float]) -> Move:
    moves = board.legal_moves()
    weights = [math.exp(policy.get(m.code(), 0.0)) for m in moves]
    total = sum(weights)
    pick = random.random() * total
    cum = 0.0
    for w, m in zip(weights, moves):
        cum += w
        if cum >= pick:
            return m
    return moves[-1]


def _playout_policy(
    board: Board, policy: Dict[int, float], root_player: int
) -> Tuple[float, List[int]]:
    sequence: List[int] = []
    while not board.terminal():
        m = _random_move(board, policy)
        sequence.append(m.code())
        board.play(m)
    s = board.score()
    if root_player != VERTICAL:
        s = 1.0 - s
    return s, sequence


def _adapt(
    policy: Dict[int, float],
    sequence: List[int],
    initial: Board,
    alpha: float,
) -> Dict[int, float]:
    polp = dict(policy)
    state = initial.copy()
    for code_played in sequence:
        moves = state.legal_moves()
        codes = [m.code() for m in moves]
        weights = [math.exp(policy.get(c, 0.0)) for c in codes]
        z = sum(weights)
        for c, w in zip(codes, weights):
            polp[c] = polp.get(c, 0.0) - alpha * w / z
        polp[code_played] = polp.get(code_played, 0.0) + alpha
        # Rejouer le coup
        idx = codes.index(code_played)
        state.play(moves[idx])
    return polp


def _nrpa(
    initial: Board,
    level: int,
    policy: Dict[int, float],
    n_iter: int,
    alpha: float,
    root_player: int,
) -> Tuple[float, List[int]]:
    if level == 0:
        b = initial.copy()
        return _playout_policy(b, policy, root_player)
    best_score = -1e9
    best_seq: List[int] = []
    cur_policy = dict(policy)
    for _ in range(n_iter):
        s, seq = _nrpa(initial, level - 1, cur_policy, n_iter, alpha, root_player)
        if s > best_score:
            best_score = s
            best_seq = seq
        cur_policy = _adapt(cur_policy, best_seq, initial, alpha)
    return best_score, best_seq


def nrpa_best_move(
    board: Board,
    budget: int = 1000,
    level: int = 2,
    alpha: float = 1.0,
) -> Move:
    """``budget`` détermine le nombre d'itérations par niveau : on choisit
    ``n_iter = round(budget ** (1 / level))`` pour que le coût total
    approxime ``budget`` playouts."""
    n_iter = max(2, int(round(budget ** (1.0 / max(1, level)))))
    _, seq = _nrpa(board, level, {}, n_iter, alpha, board.turn)
    if not seq:
        return board.legal_moves()[0]
    moves = board.legal_moves()
    for m in moves:
        if m.code() == seq[0]:
            return m
    return moves[0]
