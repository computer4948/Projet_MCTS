"""NRPA à deux politiques (co-évolutif).

Variante personnelle de NRPA conçue pour adresser la limitation
identifiée empiriquement (Exp.~6, 7, 12) du NRPA-mono-politique sur
Domineering : la politique partagée fait coévoluer V et H vers le même
comportement, ce qui empêche l'émergence de stratégies adversariales.

Ici je maintiens **deux politiques séparées** $\\theta_V$ et $\\theta_H$.
Pendant chaque playout, le joueur au trait échantillonne softmax sur
son propre $\\theta$. Après chaque playout, je suis :

* la \\emph{meilleure} séquence pour V (\\emph{i.e.} score maximal,
  V gagne),
* la \\emph{meilleure} séquence pour H (\\emph{i.e.} score minimal,
  H gagne).

L'\\emph{adapt} est ensuite fait \\emph{séparément} :

* $\\theta_V$ est adapté vers la meilleure séquence-V, en ne
  considérant que les coups joués par V dans cette séquence ;
* $\\theta_H$ est adapté vers la meilleure séquence-H, en ne
  considérant que les coups joués par H.

Au moment de proposer le meilleur coup, je retourne le premier coup
(qui appartient au joueur au trait, donc ``root\\_player``) de la
meilleure séquence trouvée pour ce joueur.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from ..board import Board, Move, VERTICAL, HORIZONTAL


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


def _playout(
    board: Board,
    theta_v: Dict[int, float],
    theta_h: Dict[int, float],
) -> Tuple[float, List[Tuple[int, int]]]:
    """Playout co-évolutif. Renvoie (score POV V, [(player, code), ...])."""
    seq: List[Tuple[int, int]] = []
    while not board.terminal():
        theta = theta_v if board.turn == VERTICAL else theta_h
        m = _random_move(board, theta)
        seq.append((board.turn, m.code()))
        board.play(m)
    return board.score(), seq


def _adapt_player(
    theta: Dict[int, float],
    seq: List[Tuple[int, int]],
    initial: Board,
    player: int,
    alpha: float,
) -> Dict[int, float]:
    """Pousse ``theta`` vers les coups de ``player`` dans ``seq``."""
    polp = dict(theta)
    state = initial.copy()
    for cur_player, code_played in seq:
        moves = state.legal_moves()
        codes = [m.code() for m in moves]
        # On ne modifie ``theta`` que pour les coups du joueur qu'on entraîne.
        if cur_player == player:
            weights = [math.exp(theta.get(c, 0.0)) for c in codes]
            z = sum(weights)
            for c, w in zip(codes, weights):
                polp[c] = polp.get(c, 0.0) - alpha * w / z
            polp[code_played] = polp.get(code_played, 0.0) + alpha
        # Toujours rejouer le coup pour avancer dans l'état.
        idx = codes.index(code_played)
        state.play(moves[idx])
    return polp


def _nrpa2p(
    initial: Board,
    level: int,
    theta_v: Dict[int, float],
    theta_h: Dict[int, float],
    n_iter: int,
    alpha: float,
) -> Tuple[float, List[Tuple[int, int]], float, List[Tuple[int, int]]]:
    """Renvoie (best_score_V, best_seq_V, best_score_H_inverse, best_seq_H).

    Note : ``best_score_V`` est le maximum (V veut gagner, score=1) ;
    ``best_score_H_inverse = 1 - min_score`` est le maximum vu de H
    (H veut score=0, donc 1-score=1)."""
    if level == 0:
        s, seq = _playout(initial.copy(), theta_v, theta_h)
        return s, seq, 1.0 - s, seq
    best_v_score = -1e9
    best_v_seq: List[Tuple[int, int]] = []
    best_h_score = -1e9
    best_h_seq: List[Tuple[int, int]] = []
    cur_v = dict(theta_v)
    cur_h = dict(theta_h)
    for _ in range(n_iter):
        sv, seqv, shv, seqh = _nrpa2p(initial, level - 1, cur_v, cur_h,
                                       n_iter, alpha)
        if sv > best_v_score:
            best_v_score = sv
            best_v_seq = seqv
        if shv > best_h_score:
            best_h_score = shv
            best_h_seq = seqh
        # Co-adaptation : V apprend de la meilleure-V, H de la meilleure-H.
        cur_v = _adapt_player(cur_v, best_v_seq, initial, VERTICAL, alpha)
        cur_h = _adapt_player(cur_h, best_h_seq, initial, HORIZONTAL, alpha)
    return best_v_score, best_v_seq, best_h_score, best_h_seq


def nrpa2p_best_move(
    board: Board,
    budget: int = 200,
    level: int = 2,
    alpha: float = 1.0,
) -> Move:
    n_iter = max(2, int(round(budget ** (1.0 / max(1, level)))))
    sv, seqv, sh, seqh = _nrpa2p(
        board, level, {}, {}, n_iter, alpha,
    )
    # Selon le joueur au trait, on prend la meilleure séquence pour lui.
    if board.turn == VERTICAL:
        seq = seqv
    else:
        seq = seqh
    if not seq:
        return board.legal_moves()[0]
    first_player, first_code = seq[0]
    moves = board.legal_moves()
    for m in moves:
        if m.code() == first_code:
            return m
    return moves[0]
