"""Expérience 7 : effet du niveau pour NMCS et NRPA.

NMCS L1, L2 et NRPA L1, L2 vs Flat. Permet d'évaluer le bénéfice
d'augmenter le niveau de récursion à effort relatif équivalent.
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
from experiments.agents import make_flat, make_nmcs, make_nrpa  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp7_levels.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)
    baseline = make_flat(args.budget)

    configs = [
        ("NMCS-L1", make_nmcs(level=1)),
        ("NMCS-L2", make_nmcs(level=2)),
        ("NRPA-L1", make_nrpa(budget=args.budget, level=1)),
        ("NRPA-L2", make_nrpa(budget=args.budget, level=2)),
    ]

    out = {"params": vars(args), "results": []}
    t0 = time.perf_counter()
    for name, agent in configs:
        t1 = time.perf_counter()
        print(f"{name} vs Flat ...", flush=True)
        m = play_match(name, agent, "Flat", baseline,
                       n_games=args.games, seed=args.seed)
        dt = time.perf_counter() - t1
        p, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"   -> {m.a_wins}-{m.b_wins}  wr={p:.2f} [{lo:.2f}, {hi:.2f}]  "
              f"t/coup={m.avg_time_a:.2f}s  ({dt:.1f}s)")
        out["results"].append({
            "name": name,
            "wins": m.a_wins, "losses": m.b_wins, "n": m.n_games,
            "winrate": p, "ci_lo": lo, "ci_hi": hi,
            "avg_time": m.avg_time_a,
            "duration_s": dt,
        })
    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
