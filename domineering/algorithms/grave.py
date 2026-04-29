"""GRAVE (Generalized RAVE).

Au lieu d'utiliser la statistique AMAF du noeud courant, on utilise celle
du dernier ancêtre ayant suffisamment de visites (paramètre ``ref_min``).
Très efficace lorsque les noeuds profonds n'ont pas eu le temps
d'accumuler assez de statistiques AMAF locales.

Adapté du notebook Breakthrough du cours.
"""
from __future__ import annotations

from typing import Dict, List

from ..board import Board, Move, VERTICAL
from .rave import _new_entry, playout_amaf, update_amaf, BIAS, Entry


def grave_descent(
    board: Board,
    table: Dict[int, Entry],
    played: List[int],
    ref: Entry,
    ref_min: int,
    playout_kwargs: dict = None,
) -> float:
    if board.terminal():
        return board.score()
    entry = table.get(board.h)
    if entry is not None:
        cur_ref = ref
        if entry[0] > ref_min:
            cur_ref = entry
        moves = board.legal_moves()
        visits = entry[1]
        sums = entry[2]
        ref_n = cur_ref[3]
        ref_s = cur_ref[4]
        best = 0
        best_val = -1.0
        best_code = moves[0].code()
        for i in range(len(moves)):
            code = moves[i].code()
            ni = visits[i]
            nai = ref_n[code]
            if nai > 0:
                beta = nai / (ni + nai + BIAS * ni * nai)
                if ni > 0:
                    q = sums[i] / ni
                    if board.turn != VERTICAL:
                        q = 1.0 - q
                else:
                    q = 1.0
                amaf = ref_s[code] / nai
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
        res = grave_descent(board, table, played, cur_ref, ref_min, playout_kwargs)
        entry[0] += 1
        entry[1][best] += 1
        entry[2][best] += res
        update_amaf(entry, played, res)
        return res
    moves = board.legal_moves()
    table[board.h] = _new_entry(len(moves))
    return playout_amaf(board, played, **(playout_kwargs or {}))


def grave_best_move(
    board: Board,
    budget: int,
    ref_min: int = 50,
    playout_strategy: str = "random",
    temperature: float = 1.0,
    epsilon: float = 0.7,
) -> Move:
    table: Dict[int, Entry] = {}
    moves = board.legal_moves()
    table[board.h] = _new_entry(len(moves))
    pkw = {"strategy": playout_strategy, "temperature": temperature, "epsilon": epsilon}
    for _ in range(budget):
        b = board.copy()
        grave_descent(b, table, [], table[board.h], ref_min, pkw)
    entry = table[board.h]
    visits = entry[1]
    return moves[max(range(len(moves)), key=lambda i: visits[i])]
