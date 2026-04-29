"""Heuristiques spécifiques à Domineering.

L'idée centrale est l'heuristique dite *move advantage* (différentielle de
liberté) : le nombre de coups adverses tués par un coup, moins le nombre
de coups propres tués (hors le coup lui-même). C'est exactement la
quantité que cherchent à maximiser les politiques expert décrites dans la
littérature de Domineering.

Trois familles d'utilisations :
    * priors PUCT  -> ``move_advantage_prior``
    * heavy playouts softmax -> ``biased_playout``
    * heavy playouts epsilon-greedy -> ``epsilon_greedy_playout``
"""
from __future__ import annotations

import math
import random
from typing import List

import numpy as np

from ..board import Board, Move, EMPTY, VERTICAL, HORIZONTAL


def _count_killed(board: Board, color: int, cells_filled) -> int:
    """Nombre de coups de couleur ``color`` rendus illégaux si ``cells_filled``
    sont remplies (et qui étaient légaux avant)."""
    bd = board.board
    dx, dy = bd.shape
    candidates = set()
    for (cr, cc) in cells_filled:
        if color == VERTICAL:
            for top_r in (cr - 1, cr):
                if 0 <= top_r and top_r + 1 < dx:
                    candidates.add((top_r, cc))
        else:
            for left_c in (cc - 1, cc):
                if 0 <= left_c and left_c + 1 < dy:
                    candidates.add((cr, left_c))
    n = 0
    for (r, c) in candidates:
        if color == VERTICAL:
            if bd[r, c] == EMPTY and bd[r + 1, c] == EMPTY:
                n += 1
        else:
            if bd[r, c] == EMPTY and bd[r, c + 1] == EMPTY:
                n += 1
    return n


def move_advantage(board: Board, move: Move) -> int:
    """Différence (coups adverses tués) - (coups propres tués hors le coup joué)."""
    me = move.color
    opp = HORIZONTAL if me == VERTICAL else VERTICAL
    cells = move.cells()
    opp_killed = _count_killed(board, opp, cells)
    me_killed = _count_killed(board, me, cells)
    return opp_killed - (me_killed - 1)


def move_advantage_prior(
    board: Board, moves: List[Move], temperature: float = 1.0
) -> List[float]:
    """Distribution softmax sur les ``moves`` d'après ``move_advantage``."""
    if not moves:
        return []
    advs = np.array([move_advantage(board, m) for m in moves], dtype=float)
    advs = (advs - advs.max()) / max(1e-9, temperature)
    expv = np.exp(advs)
    return (expv / expv.sum()).tolist()


def biased_playout(board: Board, temperature: float = 1.0) -> float:
    """Playout *heavy* : à chaque étape on échantillonne softmax sur
    ``move_advantage``."""
    while True:
        moves = board.legal_moves()
        if not moves:
            return 0.0 if board.turn == VERTICAL else 1.0
        if len(moves) == 1:
            board.play(moves[0])
            continue
        advs = np.array([move_advantage(board, m) for m in moves], dtype=float)
        advs = (advs - advs.max()) / max(1e-9, temperature)
        probs = np.exp(advs)
        probs /= probs.sum()
        idx = int(np.random.choice(len(moves), p=probs))
        board.play(moves[idx])


def epsilon_greedy_playout(board: Board, epsilon: float = 0.7) -> float:
    """Playout *heavy* : avec probabilité ``epsilon`` joue le coup de meilleur
    ``move_advantage``, sinon joue au hasard."""
    while True:
        moves = board.legal_moves()
        if not moves:
            return 0.0 if board.turn == VERTICAL else 1.0
        if len(moves) == 1:
            board.play(moves[0])
            continue
        if random.random() < epsilon:
            best_i = 0
            best_v = -1e9
            for i, m in enumerate(moves):
                v = move_advantage(board, m)
                if v > best_v:
                    best_v = v
                    best_i = i
            board.play(moves[best_i])
        else:
            board.play(moves[random.randrange(len(moves))])
