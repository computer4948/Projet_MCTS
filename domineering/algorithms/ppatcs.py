"""PPATCS - Playout Policy Adaptation with Move Features.

Cf.\\ Cazenave, *Playout Policy Adaptation with Move Features*, TCS 2016.

Au lieu d'apprendre un poids par couple ``(état, code de coup)`` comme
NRPA classique, PPATCS apprend un vecteur de poids ``theta`` partagé,
chaque coup étant représenté par un vecteur de descripteurs (features).
La politique d'échantillonnage est softmax linéaire :
``pi(a | s) propto exp(theta . phi(s, a))``.

Trois jeux de features sont implémentés (ablation comparative) :

* ``"simple"`` (mon design initial, 5 features denses) :
    f1 : ``move_advantage`` normalisé (heuristique de différence de libertés)
    f2 : -distance au centre (normalisée)
    f3 : indicateur de coin
    f4 : indicateur de bord (hors coins)
    f5 : ratio de cases voisines vides (8-voisinage)

* ``"pattern"`` (recommandé par le cours, slide PPATCS Domineering :
  *« cells next to the domino played »*). Pour chaque coup on encode
  la configuration binaire vide/occupée des 8 cases adjacentes au domino
  joué (10 cases pour un domino vertical, 10 pour un horizontal,
  bordures comptées comme « occupées »). Cela donne $2^{10} = 1024$
  patterns possibles, chaque pattern étant une feature one-hot.

* ``"combined"`` : concaténation des deux, soit $5 + 1024 = 1029$
  features.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

from ..board import Board, Move, EMPTY, VERTICAL
from ..optimizations.heuristics import move_advantage


N_FEATURES_SIMPLE = 5
N_PATTERNS = 1 << 10  # 1024 patterns (10 cases voisines binaires)
N_FEATURES_PATTERN = N_PATTERNS
N_FEATURES_COMBINED = N_FEATURES_SIMPLE + N_PATTERNS


def _features_simple(board: Board, move: Move) -> np.ndarray:
    bd = board.board
    dx, dy = bd.shape
    (r1, c1), (r2, c2) = move.cells()
    mx = 0.5 * (r1 + r2)
    my = 0.5 * (c1 + c2)
    cx = 0.5 * (dx - 1)
    cy = 0.5 * (dy - 1)
    norm = max(dx, dy)
    f1 = move_advantage(board, move) / norm
    dist = math.sqrt((mx - cx) ** 2 + (my - cy) ** 2) / norm
    f2 = -dist
    on_top_or_bottom = (r1 == 0) or (r2 == dx - 1)
    on_left_or_right = (c1 == 0) or (c2 == dy - 1)
    is_corner = float(on_top_or_bottom and on_left_or_right)
    is_edge = float(on_top_or_bottom or on_left_or_right) - is_corner
    cnt_empty = 0
    total = 0
    seen = {(r1, c1), (r2, c2)}
    for (r, c) in [(r1, c1), (r2, c2)]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if (rr, cc) in seen:
                    continue
                if 0 <= rr < dx and 0 <= cc < dy:
                    total += 1
                    if bd[rr, cc] == EMPTY:
                        cnt_empty += 1
    f5 = cnt_empty / max(1, total)
    return np.array([f1, f2, is_corner, is_edge, f5], dtype=float)


def _pattern_code(board: Board, move: Move) -> int:
    """Code binaire des 10 cases adjacentes à la position du domino.

    Ordre des bits (du moins au plus significatif) pour un coup VERTICAL
    en (r, c) (couvrant (r,c) et (r+1,c)) :

        bit 0 : (r-1, c-1)   bit 1 : (r-1, c)     bit 2 : (r-1, c+1)
        bit 3 : (r,   c-1)   bit 4 : (r,   c+1)
        bit 5 : (r+1, c-1)   bit 6 : (r+1, c+1)
        bit 7 : (r+2, c-1)   bit 8 : (r+2, c)     bit 9 : (r+2, c+1)

    Pour un coup HORIZONTAL en (r, c), on tourne le motif de 90°.
    Une case hors plateau est comptée comme « occupée » (bit = 1).
    """
    bd = board.board
    dx, dy = bd.shape
    (r1, c1), (r2, c2) = move.cells()
    if move.color == VERTICAL:
        offsets = [
            (r1 - 1, c1 - 1), (r1 - 1, c1), (r1 - 1, c1 + 1),
            (r1, c1 - 1), (r1, c1 + 1),
            (r2, c1 - 1), (r2, c1 + 1),
            (r2 + 1, c1 - 1), (r2 + 1, c1), (r2 + 1, c1 + 1),
        ]
    else:
        offsets = [
            (r1 - 1, c1 - 1), (r1 - 1, c1), (r1 - 1, c2), (r1 - 1, c2 + 1),
            (r1, c1 - 1), (r1, c2 + 1),
            (r1 + 1, c1 - 1), (r1 + 1, c1), (r1 + 1, c2), (r1 + 1, c2 + 1),
        ]
    code = 0
    for i, (rr, cc) in enumerate(offsets):
        if rr < 0 or rr >= dx or cc < 0 or cc >= dy or bd[rr, cc] != 0:
            code |= (1 << i)
    return code


def _features_pattern(board: Board, move: Move) -> np.ndarray:
    feats = np.zeros(N_PATTERNS, dtype=float)
    feats[_pattern_code(board, move)] = 1.0
    return feats


def _features_combined(board: Board, move: Move) -> np.ndarray:
    return np.concatenate([
        _features_simple(board, move),
        _features_pattern(board, move),
    ])


def features(board: Board, move: Move,
             feature_set: str = "simple") -> np.ndarray:
    if feature_set == "simple":
        return _features_simple(board, move)
    if feature_set == "pattern":
        return _features_pattern(board, move)
    if feature_set == "combined":
        return _features_combined(board, move)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def n_features(feature_set: str = "simple") -> int:
    return {
        "simple": N_FEATURES_SIMPLE,
        "pattern": N_FEATURES_PATTERN,
        "combined": N_FEATURES_COMBINED,
    }[feature_set]


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max()
    e = np.exp(logits - m)
    return e / e.sum()


def _playout_policy(
    board: Board, theta: np.ndarray, root_player: int, feature_set: str,
) -> Tuple[float, List[Tuple[np.ndarray, int]]]:
    """Renvoie ``(score, [(features_par_coup, idx_choisi)])``."""
    record: List[Tuple[np.ndarray, int]] = []
    while not board.terminal():
        moves = board.legal_moves()
        feats = np.stack([features(board, m, feature_set) for m in moves])
        logits = feats @ theta
        probs = _stable_softmax(logits)
        idx = int(np.random.choice(len(moves), p=probs))
        record.append((feats, idx))
        board.play(moves[idx])
    s = board.score()
    if root_player != VERTICAL:
        s = 1.0 - s
    return s, record


def _adapt(
    theta: np.ndarray,
    record: List[Tuple[np.ndarray, int]],
    alpha: float,
) -> np.ndarray:
    new_theta = theta.copy()
    for feats, idx in record:
        logits = feats @ theta
        probs = _stable_softmax(logits)
        expected = (probs[:, None] * feats).sum(axis=0)
        new_theta += alpha * (feats[idx] - expected)
    return new_theta


def _ppatcs(
    initial: Board,
    level: int,
    theta: np.ndarray,
    n_iter: int,
    alpha: float,
    root_player: int,
    feature_set: str,
) -> Tuple[float, List[Tuple[np.ndarray, int]]]:
    if level == 0:
        return _playout_policy(initial.copy(), theta.copy(),
                               root_player, feature_set)
    best_score = -1e9
    best_record: List[Tuple[np.ndarray, int]] = []
    cur = theta.copy()
    for _ in range(n_iter):
        s, record = _ppatcs(initial, level - 1, cur, n_iter, alpha,
                            root_player, feature_set)
        if s > best_score:
            best_score = s
            best_record = record
        cur = _adapt(cur, best_record, alpha)
    return best_score, best_record


def ppatcs_best_move(
    board: Board,
    budget: int = 200,
    level: int = 2,
    alpha: float = 1.0,
    feature_set: str = "simple",
    init_theta: np.ndarray = None,
) -> Move:
    F = n_features(feature_set)
    theta = init_theta if init_theta is not None else np.zeros(F)
    n_iter = max(2, int(round(budget ** (1.0 / max(1, level)))))
    _, record = _ppatcs(board, level, theta, n_iter, alpha,
                        board.turn, feature_set)
    if not record:
        return board.legal_moves()[0]
    _, idx0 = record[0]
    moves = board.legal_moves()
    return moves[min(idx0, len(moves) - 1)]
