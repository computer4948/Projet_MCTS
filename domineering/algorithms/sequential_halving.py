"""Sequential Halving (SH) à la racine.

Au lieu d'allouer les playouts via UCB, on divise le budget en
$\\lceil \\log_2 K \\rceil$ phases. À chaque phase, on alloue à parts
égales les playouts entre les bras encore actifs, puis on en élimine la
moitié inférieure (selon la moyenne empirique).

Cette stratégie a de meilleures garanties théoriques que UCB1 pour
identifier le meilleur bras (best-arm identification, Karnin~\\emph{et
al.} 2013). Sa version *Sequential Halving Using Scores* avec lissage
spécifique aux jeux est étudiée par Fabiano et
Cazenave~\\cite{fabiano2021shuss}. Notre implémentation est la version
classique sans lissage.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from ..board import Board, Move, VERTICAL


def _default_playout(board: Board) -> float:
    return board.playout()


def sh_best_move(
    board: Board,
    budget: int,
    playout_fn: Optional[Callable[[Board], float]] = None,
) -> Move:
    moves = board.legal_moves()
    K = len(moves)
    if K == 1:
        return moves[0]
    me = board.turn
    pf = playout_fn or _default_playout
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
                r = pf(b)
                if me != VERTICAL:
                    r = 1.0 - r
                sums[i] += r
                visits[i] += 1
        if len(active) <= 1:
            break
        active.sort(key=lambda i: -(sums[i] / max(1, visits[i])))
        active = active[: max(1, len(active) // 2)]
    return moves[max(active, key=lambda i: sums[i] / max(1, visits[i]))]
