"""RAVE (Rapid Action Value Estimation).

Combine la valeur UCT classique (Q) avec la valeur AMAF (All Moves As First)
selon une pondération beta(n).

Adapté du notebook Breakthrough du cours.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np

from ..board import Board, Move, VERTICAL, MaxCodeMoves
from ..optimizations.heuristics import move_advantage


# Entrée de la table : [n_total, n_visits[i], sum_q[i], n_amaf[code], sum_amaf[code]]
Entry = list


def _new_entry(n_moves: int) -> Entry:
    return [
        0,
        [0] * n_moves,
        [0.0] * n_moves,
        [0] * MaxCodeMoves,
        [0.0] * MaxCodeMoves,
    ]


def playout_amaf(
    board: Board,
    played: List[int],
    strategy: str = "random",
    temperature: float = 1.0,
    epsilon: float = 0.7,
) -> float:
    while True:
        moves = board.legal_moves()
        if not moves:
            return 0.0 if board.turn == VERTICAL else 1.0
        if strategy == "random" or len(moves) == 1:
            m = moves[random.randrange(len(moves))]
        elif strategy == "biased":
            advs = np.array([move_advantage(board, mm) for mm in moves], dtype=float)
            advs = (advs - advs.max()) / max(1e-9, temperature)
            probs = np.exp(advs)
            probs /= probs.sum()
            idx = int(np.random.choice(len(moves), p=probs))
            m = moves[idx]
        elif strategy == "epsilon_greedy":
            if random.random() < epsilon:
                best_i = 0
                best_v = -1e9
                for i, mm in enumerate(moves):
                    v = move_advantage(board, mm)
                    if v > best_v:
                        best_v = v
                        best_i = i
                m = moves[best_i]
            else:
                m = moves[random.randrange(len(moves))]
        else:
            m = moves[random.randrange(len(moves))]
        played.append(m.code())
        board.play(m)


def update_amaf(entry: Entry, played: List[int], res: float) -> None:
    seen = set()
    for code in played:
        if code in seen:
            continue
        seen.add(code)
        entry[3][code] += 1
        entry[4][code] += res


BIAS = 1e-5


def rave_descent(
    board: Board,
    table: Dict[int, Entry],
    played: List[int],
    playout_kwargs: dict = None,
) -> float:
    if board.terminal():
        return board.score()
    entry = table.get(board.h)
    if entry is not None:
        moves = board.legal_moves()
        visits = entry[1]
        sums = entry[2]
        n_amaf = entry[3]
        s_amaf = entry[4]
        best = 0
        best_val = -1.0
        best_code = moves[0].code()
        for i in range(len(moves)):
            code = moves[i].code()
            ni = visits[i]
            nai = n_amaf[code]
            if nai > 0:
                beta = nai / (ni + nai + BIAS * ni * nai)
                if ni > 0:
                    q = sums[i] / ni
                    if board.turn != VERTICAL:
                        q = 1.0 - q
                else:
                    q = 1.0
                amaf = s_amaf[code] / nai
                if board.turn != VERTICAL:
                    amaf = 1.0 - amaf
                val = (1.0 - beta) * q + beta * amaf
            else:
                val = 1e9
            if val > best_val:
                best_val = val
                best = i
                best_code = code
        board.play(moves[best])
        played.append(best_code)
        res = rave_descent(board, table, played, playout_kwargs)
        entry[0] += 1
        entry[1][best] += 1
        entry[2][best] += res
        update_amaf(entry, played, res)
        return res
    moves = board.legal_moves()
    table[board.h] = _new_entry(len(moves))
    return playout_amaf(board, played, **(playout_kwargs or {}))


def rave_best_move(
    board: Board,
    budget: int,
    playout_strategy: str = "random",
    temperature: float = 1.0,
    epsilon: float = 0.7,
) -> Move:
    table: Dict[int, Entry] = {}
    pkw = {"strategy": playout_strategy, "temperature": temperature, "epsilon": epsilon}
    for _ in range(budget):
        b = board.copy()
        rave_descent(b, table, [], pkw)
    entry = table.get(board.h)
    moves = board.legal_moves()
    if entry is None:
        return moves[0]
    visits = entry[1]
    return moves[max(range(len(moves)), key=lambda i: visits[i])]
