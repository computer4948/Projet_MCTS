"""Expérience 11 : agents étendus (MAST, SHOT, GNRPA) + mode misère.

Trois mini-études :
    (a) MAST, SHOT, GNRPA-L2 vs Flat à budget 200 sur 5x5 normal.
    (b) GNRPA vs NRPA à même niveau et budget (mesure le gain du biais).
    (c) UCT vs Flat en mode \\emph{misère} (le joueur sans coup gagne).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domineering.board import set_board_size, set_game_mode  # noqa: E402
from domineering.tournament import play_match  # noqa: E402
from domineering.stats import wilson_ci  # noqa: E402
from experiments.agents import (  # noqa: E402
    make_flat, make_uct, make_nrpa, make_nmcs,
    make_mast, make_shot, make_gnrpa,
)


def run_match(out, a_name, a, b_name, b, games, seed):
    print(f"  {a_name} vs {b_name} ...", flush=True)
    t1 = time.perf_counter()
    m = play_match(a_name, a, b_name, b, n_games=games, seed=seed)
    dt = time.perf_counter() - t1
    p, lo, hi = wilson_ci(m.a_wins, m.n_games)
    print(f"     -> {m.a_wins}-{m.b_wins}  wr={p:.2f} [{lo:.2f}, {hi:.2f}]  ({dt:.1f}s)")
    out.append({
        "a": a_name, "b": b_name,
        "wins": m.a_wins, "losses": m.b_wins, "n": m.n_games,
        "winrate": p, "ci_lo": lo, "ci_hi": hi,
        "duration_s": dt,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp11_extended.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)
    out = {"params": vars(args), "extended": [], "misere": [], "discounted_nmcs": []}
    t0 = time.perf_counter()

    # --- (a) Étendus vs Flat
    set_game_mode("normal")
    print("--- (a) Agents étendus vs Flat (mode normal) ---")
    flat = make_flat(args.budget)
    run_match(out["extended"], "MAST",      make_mast(args.budget),       "Flat", flat, args.games, args.seed)
    run_match(out["extended"], "SHOT",      make_shot(args.budget),       "Flat", flat, args.games, args.seed)
    run_match(out["extended"], "GNRPA-L2",  make_gnrpa(args.budget, level=2), "Flat", flat, args.games, args.seed)
    # GNRPA vs NRPA
    print("\n--- (b) GNRPA vs NRPA (mêmes paramètres) ---")
    run_match(out["extended"], "GNRPA-L1",  make_gnrpa(args.budget, level=1), "NRPA-L1", make_nrpa(args.budget, level=1), args.games, args.seed)
    run_match(out["extended"], "GNRPA-L2",  make_gnrpa(args.budget, level=2), "NRPA-L2", make_nrpa(args.budget, level=2), args.games, args.seed)

    # --- (c) Mode misère
    print("\n--- (c) Mode misère ---")
    set_game_mode("misere")
    run_match(out["misere"], "UCT-misere",  make_uct(args.budget),  "Flat-misere", make_flat(args.budget), args.games, args.seed)
    run_match(out["misere"], "NMCS-L1-misere", make_nmcs(level=1),  "Flat-misere", make_flat(args.budget), args.games, args.seed)
    set_game_mode("normal")

    # --- (d) NMCS discounted vs NMCS classique
    print("\n--- (d) NMCS-L1 discounted vs NMCS-L1 classique ---")
    run_match(out["discounted_nmcs"], "NMCS-L1-disc", make_nmcs(level=1, discounted=True),
              "NMCS-L1", make_nmcs(level=1), args.games, args.seed)

    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
