"""GNRPA - Generalized Nested Rollout Policy Adaptation.

Cazenave (2020) introduit un *biais* $\\beta(s, a)$ ajouté au logit
d'échantillonnage, équivalent à initialiser $\\theta$ avec $\\beta$.
Pour Domineering nous utilisons $\\beta = $ ``move_advantage`` normalisée :
les bons coups (au sens de la différence de libertés) ont un score
softmax initial supérieur, ce qui accélère grandement la convergence
de la politique.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from ..board import Board, Move, VERTICAL
from ..optimizations.heuristics import move_advantage


def _beta(board: Board, move: Move, scale: float = 1.0) -> float:
    """Biais $\\beta$ : ``move_advantage`` normalisée par la taille du
    plateau, multipliée par ``scale``."""
    dx, dy = board.board.shape
    return scale * move_advantage(board, move) / max(dx, dy)


def _random_move(
    board: Board, policy: Dict[int, float], beta_scale: float
) -> Move:
    moves = board.legal_moves()
    weights = [
        math.exp(policy.get(m.code(), 0.0) + _beta(board, m, beta_scale))
        for m in moves
    ]
    total = sum(weights)
    pick = random.random() * total
    cum = 0.0
    for w, m in zip(weights, moves):
        cum += w
        if cum >= pick:
            return m
    return moves[-1]


def _playout_policy(
    board: Board,
    policy: Dict[int, float],
    root_player: int,
    beta_scale: float,
) -> Tuple[float, List[int]]:
    sequence: List[int] = []
    while not board.terminal():
        m = _random_move(board, policy, beta_scale)
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
    beta_scale: float,
) -> Dict[int, float]:
    polp = dict(policy)
    state = initial.copy()
    for code_played in sequence:
        moves = state.legal_moves()
        codes = [m.code() for m in moves]
        weights = [
            math.exp(policy.get(c, 0.0) + _beta(state, m, beta_scale))
            for c, m in zip(codes, moves)
        ]
        z = sum(weights)
        for c, w in zip(codes, weights):
            polp[c] = polp.get(c, 0.0) - alpha * w / z
        polp[code_played] = polp.get(code_played, 0.0) + alpha
        idx = codes.index(code_played)
        state.play(moves[idx])
    return polp


def _gnrpa(
    initial: Board,
    level: int,
    policy: Dict[int, float],
    n_iter: int,
    alpha: float,
    root_player: int,
    beta_scale: float,
) -> Tuple[float, List[int]]:
    if level == 0:
        b = initial.copy()
        return _playout_policy(b, policy, root_player, beta_scale)
    best_score = -1e9
    best_seq: List[int] = []
    cur = dict(policy)
    for _ in range(n_iter):
        s, seq = _gnrpa(initial, level - 1, cur, n_iter, alpha,
                        root_player, beta_scale)
        if s > best_score:
            best_score = s
            best_seq = seq
        cur = _adapt(cur, best_seq, initial, alpha, beta_scale)
    return best_score, best_seq


def gnrpa_best_move(
    board: Board,
    budget: int = 200,
    level: int = 2,
    alpha: float = 1.0,
    beta_scale: float = 1.0,
) -> Move:
    n_iter = max(2, int(round(budget ** (1.0 / max(1, level)))))
    _, seq = _gnrpa(board, level, {}, n_iter, alpha, board.turn, beta_scale)
    if not seq:
        return board.legal_moves()[0]
    moves = board.legal_moves()
    for m in moves:
        if m.code() == seq[0]:
            return m
    return moves[0]
