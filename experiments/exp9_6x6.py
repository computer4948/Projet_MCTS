"""Expérience 9 : round-robin sur plateau 6x6 pour confirmer la
hiérarchie observée en 5x5.

Sous-ensemble d'agents (les plus pertinents) à budget réduit pour rester
dans une enveloppe temps raisonnable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domineering.board import set_board_size  # noqa: E402
from domineering.tournament import round_robin  # noqa: E402
from domineering.stats import wilson_ci  # noqa: E402
from experiments.agents import (  # noqa: E402
    make_flat,
    make_uct,
    make_rave,
    make_grave,
    make_puct,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp9_6x6.json")
    args = parser.parse_args()

    set_board_size(6, 6)

    agents = {
        "Flat":  make_flat(args.budget),
        "UCT":   make_uct(args.budget),
        "RAVE":  make_rave(args.budget),
        "GRAVE": make_grave(args.budget),
        "PUCT":  make_puct(args.budget),
    }

    t0 = time.perf_counter()
    results = round_robin(agents, n_games=args.games, seed=args.seed, verbose=True)
    duration = time.perf_counter() - t0

    out = {
        "params": vars(args),
        "duration_s": duration,
        "agents": list(agents.keys()),
        "matches": [],
    }
    for r in results:
        p, lo, hi = wilson_ci(r.a_wins, r.n_games)
        out["matches"].append({
            "a": r.a_name, "b": r.b_name,
            "wins_a": r.a_wins, "wins_b": r.b_wins, "n": r.n_games,
            "winrate": p, "ci_lo": lo, "ci_hi": hi,
            "avg_time_a": r.avg_time_a, "avg_time_b": r.avg_time_b,
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {duration:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
