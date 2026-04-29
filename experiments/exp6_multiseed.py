"""Expérience 6 : tournoi round-robin multi-seed avec intervalles de
confiance Wilson 95 %.

Pour chaque appariement, on joue ``n_games`` parties pour chacun des
``n_seeds`` seeds, puis on agrège (chaque match contribue ``n_games``
parties au total combiné).
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
from domineering.stats import wilson_ci  # noqa: E402
from experiments.agents import (  # noqa: E402
    make_flat,
    make_ucb,
    make_uct,
    make_rave,
    make_grave,
    make_nmcs,
    make_nrpa,
    make_puct,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=10,
                        help="parties par seed et par appariement")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2026, 4242, 9001])
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp6_multiseed.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)

    def fresh_agents():
        return {
            "Flat": make_flat(args.budget),
            "UCB": make_ucb(args.budget),
            "UCT": make_uct(args.budget),
            "RAVE": make_rave(args.budget),
            "GRAVE": make_grave(args.budget),
            "PUCT": make_puct(args.budget),
            "NRPA-L2": make_nrpa(budget=args.budget, level=2),
        }

    names = list(fresh_agents().keys())
    pairs = [(names[i], names[j]) for i in range(len(names))
             for j in range(i + 1, len(names))]

    pair_stats = {p: {"wins_a": 0, "wins_b": 0, "n": 0} for p in pairs}
    t0 = time.perf_counter()
    for s_idx, seed in enumerate(args.seeds):
        agents = fresh_agents()
        print(f"\n=== Seed {seed} ({s_idx+1}/{len(args.seeds)}) ===")
        for k, (a, b) in enumerate(pairs):
            t1 = time.perf_counter()
            m = play_match(a, agents[a], b, agents[b],
                           n_games=args.games, seed=seed)
            pair_stats[(a, b)]["wins_a"] += m.a_wins
            pair_stats[(a, b)]["wins_b"] += m.b_wins
            pair_stats[(a, b)]["n"] += m.n_games
            dt = time.perf_counter() - t1
            print(f"   [{k+1:2d}/{len(pairs)}] {a:8s} {m.a_wins:2d}-{m.b_wins:2d} {b:8s}  ({dt:.1f}s)", flush=True)

    out = {
        "params": vars(args),
        "agents": names,
        "matches": [],
        "duration_s": time.perf_counter() - t0,
    }
    for (a, b), s in pair_stats.items():
        p, lo, hi = wilson_ci(s["wins_a"], s["n"])
        out["matches"].append({
            "a": a, "b": b,
            "wins_a": s["wins_a"], "wins_b": s["wins_b"], "n": s["n"],
            "winrate": p, "ci_lo": lo, "ci_hi": hi,
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
