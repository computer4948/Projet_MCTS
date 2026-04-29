"""Expérience 12 : ablation des features pour PPATCS sur Domineering.

**Contribution originale.** Le slide \"Playout Policy learning with Move
Features\" du cours suggère, pour Domineering, d'utiliser comme features
\"the cells next to the domino played\" (configuration binaire des cases
voisines du coup). Je compare trois jeux de features dans PPATCS :

  (i)   ``simple``    : 5 features denses hand-crafted (dont
        ``move_advantage``, distance au centre, indicateurs de
        coin/bord, ratio de voisins vides).
  (ii)  ``pattern``   : feature one-hot sur les $2^{10} = 1024$
        configurations binaires des cases adjacentes au domino,
        comme suggéré par le cours.
  (iii) ``combined``  : concaténation, $5 + 1024 = 1029$ features.

Chaque variante joue 16 parties contre Flat-200 et 16 contre NRPA-L1
(sans features), avec graine fixe.
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
from experiments.agents import make_flat, make_nrpa, make_ppatcs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp12_features.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)
    out = {"params": vars(args), "vs_flat": [], "vs_nrpa": []}
    t0 = time.perf_counter()

    print(f"--- PPATCS (level={args.level}, budget={args.budget}) "
          f"x 3 feature sets vs Flat ---")
    flat = make_flat(args.budget)
    for fs in ("simple", "pattern", "combined"):
        a = make_ppatcs(args.budget, level=args.level, feature_set=fs)
        t1 = time.perf_counter()
        m = play_match(f"PPATCS-{fs}", a, "Flat", flat,
                       n_games=args.games, seed=args.seed)
        dt = time.perf_counter() - t1
        p, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"  PPATCS({fs}): {m.a_wins}-{m.b_wins} wr={p:.2f} "
              f"[{lo:.2f}, {hi:.2f}] ({dt:.1f}s)")
        out["vs_flat"].append({
            "feature_set": fs, "wins": m.a_wins, "losses": m.b_wins,
            "n": m.n_games, "winrate": p, "ci_lo": lo, "ci_hi": hi,
            "duration_s": dt,
        })

    print(f"\n--- PPATCS (level={args.level}) x 3 feature sets vs NRPA-L1 ---")
    nrpa = make_nrpa(args.budget, level=args.level)
    for fs in ("simple", "pattern", "combined"):
        a = make_ppatcs(args.budget, level=args.level, feature_set=fs)
        t1 = time.perf_counter()
        m = play_match(f"PPATCS-{fs}", a, f"NRPA-L{args.level}", nrpa,
                       n_games=args.games, seed=args.seed)
        dt = time.perf_counter() - t1
        p, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"  PPATCS({fs}): {m.a_wins}-{m.b_wins} wr={p:.2f} "
              f"[{lo:.2f}, {hi:.2f}] ({dt:.1f}s)")
        out["vs_nrpa"].append({
            "feature_set": fs, "wins": m.a_wins, "losses": m.b_wins,
            "n": m.n_games, "winrate": p, "ci_lo": lo, "ci_hi": hi,
            "duration_s": dt,
        })

    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
