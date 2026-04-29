"""Expérience 10 : PPATCS — Playout Policy Adaptation with Move Features.

PPATCS (Cazenave 2016) étend NRPA en apprenant des poids sur des
descripteurs de coup plutôt que sur (état, code de coup). On compare
PPATCS aux baselines naturelles : NRPA (même budget, même niveau) et
Flat. Le but est de mesurer si l'utilisation de descripteurs apporte un
gain dans le cadre 2-joueurs partagé que nous avons retenu.
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
    make_flat, make_nrpa, make_ppatcs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp10_ppatcs.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)

    duels = [
        ("PPATCS-L1", make_ppatcs(args.budget, level=1), "Flat", make_flat(args.budget)),
        ("PPATCS-L2", make_ppatcs(args.budget, level=2), "Flat", make_flat(args.budget)),
        ("PPATCS-L2", make_ppatcs(args.budget, level=2),
         "NRPA-L2", make_nrpa(args.budget, level=2)),
        ("PPATCS-L1", make_ppatcs(args.budget, level=1),
         "NRPA-L1", make_nrpa(args.budget, level=1)),
    ]

    out = {"params": vars(args), "results": []}
    t0 = time.perf_counter()
    for a_name, a, b_name, b in duels:
        t1 = time.perf_counter()
        print(f"{a_name} vs {b_name} ...", flush=True)
        m = play_match(a_name, a, b_name, b, n_games=args.games, seed=args.seed)
        dt = time.perf_counter() - t1
        p, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"   -> {m.a_wins}-{m.b_wins}  wr={p:.2f} [{lo:.2f}, {hi:.2f}]  ({dt:.1f}s)")
        out["results"].append({
            "a": a_name, "b": b_name,
            "wins": m.a_wins, "losses": m.b_wins, "n": m.n_games,
            "winrate": p, "ci_lo": lo, "ci_hi": hi,
            "avg_time_a": m.avg_time_a, "avg_time_b": m.avg_time_b,
            "duration_s": dt,
        })
    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
