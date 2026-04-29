"""Expérience 3 : ablation des heavy playouts.

On compare {UCT, RAVE, GRAVE} en versions *random*, *biased* (softmax sur
``move_advantage``) et *epsilon-greedy*. Trois versions de chaque algo se
mesurent en duel direct contre la version *random* à budget identique.
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
from experiments.agents import make_uct, make_rave, make_grave  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp3_heavy.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)

    duels = []
    duels.append(("UCT-biased", make_uct(args.budget, heavy="biased"),
                  "UCT-random", make_uct(args.budget)))
    duels.append(("UCT-epsilon", make_uct(args.budget, heavy="epsilon"),
                  "UCT-random", make_uct(args.budget)))
    duels.append(("RAVE-biased", make_rave(args.budget, heavy="biased"),
                  "RAVE-random", make_rave(args.budget)))
    duels.append(("RAVE-epsilon", make_rave(args.budget, heavy="epsilon_greedy"),
                  "RAVE-random", make_rave(args.budget)))
    duels.append(("GRAVE-biased", make_grave(args.budget, heavy="biased"),
                  "GRAVE-random", make_grave(args.budget)))
    duels.append(("GRAVE-epsilon", make_grave(args.budget, heavy="epsilon_greedy"),
                  "GRAVE-random", make_grave(args.budget)))

    out = {"params": vars(args), "results": []}
    t0 = time.perf_counter()
    for a_name, a, b_name, b in duels:
        print(f"{a_name} vs {b_name} ...", flush=True)
        t1 = time.perf_counter()
        m = play_match(a_name, a, b_name, b, n_games=args.games, seed=args.seed)
        dur = time.perf_counter() - t1
        print(f"   -> {m.a_wins}-{m.b_wins} (winrate {m.a_winrate:.2f}, {dur:.1f}s)")
        out["results"].append({
            "a": a_name, "b": b_name,
            "winrate": m.a_winrate,
            "wins": m.a_wins,
            "losses": m.b_wins,
            "avg_time_a": m.avg_time_a,
            "avg_time_b": m.avg_time_b,
            "duration_s": dur,
        })
    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nRésultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
