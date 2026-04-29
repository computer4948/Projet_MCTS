"""Expérience 2 : effet du budget de simulations.

Chaque algorithme (UCT, RAVE, GRAVE, PUCT) joue contre un Flat MC à
budget fixe (le baseline) pour différents budgets. On trace ensuite le
taux de victoire en fonction du budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domineering.board import set_board_size  # noqa: E402
from domineering.tournament import play_match  # noqa: E402
from experiments.agents import (  # noqa: E402
    make_flat,
    make_uct,
    make_rave,
    make_grave,
    make_puct,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--baseline-budget", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[50, 100, 200, 400, 800])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp2_budget.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)
    baseline = make_flat(args.baseline_budget)

    factories = {
        "UCT": lambda b: make_uct(b),
        "RAVE": lambda b: make_rave(b),
        "GRAVE": lambda b: make_grave(b),
        "PUCT": lambda b: make_puct(b),
    }

    out = {"params": vars(args), "results": {}}
    t0 = time.perf_counter()
    for name, fac in factories.items():
        out["results"][name] = []
        for b in args.budgets:
            t1 = time.perf_counter()
            print(f"{name} budget={b} vs Flat-{args.baseline_budget} ...", flush=True)
            m = play_match(name, fac(b), "Flat", baseline,
                           n_games=args.games, seed=args.seed)
            dur = time.perf_counter() - t1
            print(f"   -> winrate={m.a_winrate:.2f}  ({dur:.1f}s)")
            out["results"][name].append({
                "budget": b,
                "winrate": m.a_winrate,
                "wins": m.a_wins,
                "losses": m.b_wins,
                "avg_time_per_move": m.avg_time_a,
                "duration_s": dur,
            })
    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
