"""Outils d'auto-jeu : faire jouer deux agents l'un contre l'autre.

Un *agent* est une fonction (Board -> Move). Les utilitaires ci-dessous
gèrent l'alternance des couleurs et calculent des taux de victoire.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .board import Board, Move, VERTICAL, HORIZONTAL


Agent = Callable[[Board], Move]


@dataclass
class GameRecord:
    moves: List[str] = field(default_factory=list)
    score: float = 0.5  # POV VERTICAL
    duration_a: float = 0.0
    duration_b: float = 0.0
    a_color: int = VERTICAL


def play_game(
    agent_a: Agent,
    agent_b: Agent,
    a_plays_vertical: bool,
    record_moves: bool = False,
) -> GameRecord:
    b = Board()
    rec = GameRecord(a_color=VERTICAL if a_plays_vertical else HORIZONTAL)
    while not b.terminal():
        if (b.turn == VERTICAL and a_plays_vertical) or (
            b.turn == HORIZONTAL and not a_plays_vertical
        ):
            t0 = time.perf_counter()
            m = agent_a(b)
            rec.duration_a += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            m = agent_b(b)
            rec.duration_b += time.perf_counter() - t0
        if record_moves:
            rec.moves.append(repr(m))
        b.play(m)
    rec.score = b.score()
    return rec


@dataclass
class MatchResult:
    a_name: str
    b_name: str
    n_games: int
    a_wins: int = 0
    b_wins: int = 0
    a_wins_as_vertical: int = 0
    a_wins_as_horizontal: int = 0
    games_as_vertical: int = 0
    games_as_horizontal: int = 0
    avg_time_a: float = 0.0
    avg_time_b: float = 0.0

    @property
    def a_winrate(self) -> float:
        return self.a_wins / max(1, self.n_games)

    def as_dict(self) -> dict:
        return {
            "a": self.a_name,
            "b": self.b_name,
            "n_games": self.n_games,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "a_wins_as_vertical": self.a_wins_as_vertical,
            "a_wins_as_horizontal": self.a_wins_as_horizontal,
            "games_as_vertical": self.games_as_vertical,
            "games_as_horizontal": self.games_as_horizontal,
            "avg_time_a": self.avg_time_a,
            "avg_time_b": self.avg_time_b,
            "a_winrate": self.a_winrate,
        }


def play_match(
    a_name: str,
    agent_a: Agent,
    b_name: str,
    agent_b: Agent,
    n_games: int = 20,
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> MatchResult:
    """Joue ``n_games`` parties en alternant les couleurs.

    Couleurs : moitié des parties A est VERTICAL, moitié HORIZONTAL. Si
    ``n_games`` est impair, la dernière partie est tirée au sort."""
    if seed is not None:
        random.seed(seed)
        try:
            import numpy as _np

            _np.random.seed(seed)
        except ImportError:
            pass
    res = MatchResult(a_name=a_name, b_name=b_name, n_games=n_games)
    total_t_a = 0.0
    total_t_b = 0.0
    for k in range(n_games):
        a_vertical = (k < n_games // 2)
        if n_games % 2 == 1 and k == n_games - 1:
            a_vertical = bool(random.getrandbits(1))
        rec = play_game(agent_a, agent_b, a_vertical)
        total_t_a += rec.duration_a
        total_t_b += rec.duration_b
        if a_vertical:
            res.games_as_vertical += 1
            if rec.score == 1.0:  # VERTICAL win = A win
                res.a_wins += 1
                res.a_wins_as_vertical += 1
            else:
                res.b_wins += 1
        else:
            res.games_as_horizontal += 1
            if rec.score == 0.0:  # HORIZONTAL win = A win
                res.a_wins += 1
                res.a_wins_as_horizontal += 1
            else:
                res.b_wins += 1
        if progress is not None:
            progress(k + 1, n_games)
    res.avg_time_a = total_t_a / n_games
    res.avg_time_b = total_t_b / n_games
    return res


def round_robin(
    agents: Dict[str, Agent],
    n_games: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[MatchResult]:
    """Round-robin de tous les agents (sauf auto-matches)."""
    names = list(agents.keys())
    results: List[MatchResult] = []
    pairs: List[Tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    for k, (a, b) in enumerate(pairs):
        if verbose:
            print(f"[{k+1}/{len(pairs)}] {a} vs {b} ...", flush=True)
        m = play_match(a, agents[a], b, agents[b], n_games=n_games, seed=seed)
        results.append(m)
        if verbose:
            print(f"   -> {a} {m.a_wins}-{m.b_wins} {b}  "
                  f"(t/coup A={m.avg_time_a:.2f}s, B={m.avg_time_b:.2f}s)")
    return results


def winrate_matrix(results: List[MatchResult], names: List[str]) -> Dict[str, Dict[str, float]]:
    """Construit une matrice symétrique des taux de victoire (NaN sur la diagonale)."""
    mat: Dict[str, Dict[str, float]] = {n: {} for n in names}
    for n in names:
        mat[n][n] = float("nan")
    for r in results:
        mat[r.a_name][r.b_name] = r.a_winrate
        mat[r.b_name][r.a_name] = 1.0 - r.a_winrate
    return mat
