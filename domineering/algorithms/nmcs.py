"""Nested Monte-Carlo Search (NMCS) version 2 joueurs.

Au niveau 0 c'est un playout aléatoire, au niveau ``L`` chaque coup est
choisi en simulant tous les enfants au niveau ``L-1`` et en gardant celui
qui maximise le score du joueur au trait.

Références :
* Cazenave, *Nested Monte-Carlo Search*, IJCAI 2009.
* Cazenave, Saffidine, Schofield, Thielscher, *Nested Monte Carlo Search
  for Two-player Games*, AAAI 2016.

Cette dernière introduit l'idée d'un **playout *discounté***
$v(s_t) / (t + 1)$ qui transforme le résultat binaire victoire/défaite
en un signal continu favorisant les victoires courtes et les défaites
longues. On peut activer cette variante en passant
``discounted=True``.
"""
from __future__ import annotations

from ..board import Board, Move, VERTICAL


def _discounted(score: float, length: int) -> float:
    """Transformation du score en : 0,5 + (score - 0,5) / (1 + length).

    On préserve la convention POV VERTICAL : 0,5 reste neutre, score
    proche de 1 (V gagne vite) -> 0,5 + 0,5/(1+length), tandis que le
    score normal restait à 1,0. Plus la partie est longue, plus le
    signal s'amortit, ce qui favorise les victoires courtes."""
    return 0.5 + (score - 0.5) / (1.0 + length)


def _nmcs_playout(board: Board, level: int, discounted: bool) -> float:
    b = board.copy()
    plies = 0
    while not b.terminal():
        b.play(_nmcs_choose(b, level, discounted))
        plies += 1
    s = b.score()
    return _discounted(s, plies) if discounted else s


def _nmcs_choose(board: Board, level: int, discounted: bool) -> Move:
    moves = board.legal_moves()
    cp = board.turn
    best_score = -1.0
    best_m = moves[0]
    for m in moves:
        child = board.copy()
        child.play(m)
        if level <= 1:
            end = child.copy()
            s = end.playout()
            if discounted:
                # On approxime ``length`` par le nombre de cases occupées
                length = int((child.board != 0).sum() // 2)
                s = _discounted(s, length)
        else:
            s = _nmcs_playout(child, level - 1, discounted)
        cs = s if cp == VERTICAL else 1.0 - s
        if cs > best_score:
            best_score = cs
            best_m = m
    return best_m


def nmcs_best_move(
    board: Board, budget: int = 1, level: int = 1, discounted: bool = False,
) -> Move:
    return _nmcs_choose(board, max(1, level), discounted)
