"""Expérience 5 : passage à l'échelle (taille du plateau).

Pour des tailles 4x4, 5x5, 6x6, 7x7 on mesure le temps moyen par coup et
le taux de victoire de GRAVE contre Flat (à budget identique).
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
from experiments.agents import make_grave, make_flat  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp5_size.json")
    args = parser.parse_args()

    out = {"params": vars(args), "results": []}
    t0 = time.perf_counter()
    for s in args.sizes:
        set_board_size(s, s)
        a = make_grave(args.budget)
        b = make_flat(args.budget)
        print(f"Size {s}x{s} ...", flush=True)
        t1 = time.perf_counter()
        m = play_match("GRAVE", a, "Flat", b,
                       n_games=args.games, seed=args.seed)
        dur = time.perf_counter() - t1
        print(f"   -> winrate={m.a_winrate:.2f}  t/coup GRAVE={m.avg_time_a:.3f}s "
              f"Flat={m.avg_time_b:.3f}s  ({dur:.1f}s)")
        out["results"].append({
            "size": s,
            "winrate": m.a_winrate,
            "avg_time_grave": m.avg_time_a,
            "avg_time_flat": m.avg_time_b,
            "duration_s": dur,
        })
    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
