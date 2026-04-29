"""Expérience 4 : sensibilité de UCT à la constante d'exploration ``c``.

UCT avec différentes valeurs de ``c`` joue contre un adversaire fixe
(Flat MC à budget identique). On retient la valeur qui maximise le taux
de victoire empirique.
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
from experiments.agents import make_uct, make_flat  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--c-values", type=float, nargs="+",
                        default=[0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0])
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp4_uct_c.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)
    baseline = make_flat(args.budget)

    out = {"params": vars(args), "results": []}
    t0 = time.perf_counter()
    for c in args.c_values:
        agent = make_uct(args.budget, c=c)
        t1 = time.perf_counter()
        print(f"UCT c={c} vs Flat ...", flush=True)
        m = play_match(f"UCT_c{c}", agent, "Flat", baseline,
                       n_games=args.games, seed=args.seed)
        dur = time.perf_counter() - t1
        print(f"   -> winrate={m.a_winrate:.2f}  ({dur:.1f}s)")
        out["results"].append({
            "c": c,
            "winrate": m.a_winrate,
            "wins": m.a_wins,
            "losses": m.b_wins,
            "duration_s": dur,
        })
    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
