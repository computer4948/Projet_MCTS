"""SHOT - Sequential Halving Applied to Trees.

Cazenave, *Sequential Halving Applied to Trees*, IEEE TCIAIG 2015.
Le cours résume : « SHOT près de la racine, UCT plus profond ».

Principe : à la racine on alloue les playouts par Sequential Halving ;
les playouts effectuent ensuite une descente UCT classique. Lorsqu'on
descend dans un sous-arbre déjà connu, on réutilise les statistiques
(le budget cumulé n'est pas perdu).

Implémentation simplifiée : SH à la racine, UCT en dessous (les
statistiques par sous-arbre sont partagées via la table de
transposition d'UCT).
"""
from __future__ import annotations

import math
from typing import Dict

from ..board import Board, Move, VERTICAL
from .uct import _new_entry, uct_descent, _default_playout, Entry


def shot_best_move(
    board: Board,
    budget: int,
    c: float = 0.4,
) -> Move:
    moves = board.legal_moves()
    K = len(moves)
    if K == 1:
        return moves[0]
    me = board.turn
    table: Dict[int, Entry] = {}
    sums = [0.0] * K
    visits = [0] * K
    active = list(range(K))
    n_phases = max(1, int(math.ceil(math.log2(K))))
    for _ in range(n_phases):
        per_arm = max(1, budget // (len(active) * n_phases))
        for i in active:
            for _ in range(per_arm):
                b = board.copy()
                b.play(moves[i])
                # Descente UCT depuis l'enfant ; les statistiques sont
                # partagées via ``table`` (toutes les phases bénéficient
                # des recherches précédentes).
                r = uct_descent(b, table, c, _default_playout)
                if me != VERTICAL:
                    r = 1.0 - r
                sums[i] += r
                visits[i] += 1
        if len(active) <= 1:
            break
        active.sort(key=lambda i: -(sums[i] / max(1, visits[i])))
        active = active[: max(1, len(active) // 2)]
    return moves[max(active, key=lambda i: sums[i] / max(1, visits[i]))]
