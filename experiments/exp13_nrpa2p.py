"""Expérience 13 : NRPA à deux politiques (co-évolution V/H).

**Contribution originale.** Le NRPA classique avec politique partagée
échoue en 2-joueurs (Exp.~6, 7) car les deux camps coévoluent vers le
même comportement. Je teste une variante personnelle avec deux
politiques séparées $\\theta_V$ et $\\theta_H$ co-évolutives :
chaque joueur apprend uniquement de ses propres coups dans les
sequences les plus favorables à son camp.

Comparaisons à budget 200, 16 parties par appariement :
    - NRPA-2P vs NRPA classique (mêmes niveau, budget) : mesure le
      gain net de la co-évolution
    - NRPA-2P vs Flat-200 : mesure absolue
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
from experiments.agents import make_flat, make_nrpa, make_nrpa2p  # noqa: E402


def run(out, label, a, b_label, b, games, seed):
    print(f"  {label} vs {b_label} ...", flush=True)
    t1 = time.perf_counter()
    m = play_match(label, a, b_label, b, n_games=games, seed=seed)
    dt = time.perf_counter() - t1
    p, lo, hi = wilson_ci(m.a_wins, m.n_games)
    print(f"     -> {m.a_wins}-{m.b_wins}  wr={p:.2f} [{lo:.2f},{hi:.2f}]  ({dt:.1f}s)")
    out.append({
        "a": label, "b": b_label,
        "wins": m.a_wins, "losses": m.b_wins, "n": m.n_games,
        "winrate": p, "ci_lo": lo, "ci_hi": hi,
        "duration_s": dt,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp13_nrpa2p.json")
    args = parser.parse_args()
    set_board_size(args.size, args.size)
    out = {"params": vars(args), "results": []}
    t0 = time.perf_counter()

    flat = make_flat(args.budget)
    print("--- vs Flat ---")
    run(out["results"], "NRPA-2P-L1", make_nrpa2p(args.budget, level=1),
        "Flat", flat, args.games, args.seed)
    run(out["results"], "NRPA-2P-L2", make_nrpa2p(args.budget, level=2),
        "Flat", flat, args.games, args.seed)

    print("\n--- vs NRPA classique (même niveau) ---")
    run(out["results"], "NRPA-2P-L1", make_nrpa2p(args.budget, level=1),
        "NRPA-L1", make_nrpa(args.budget, level=1), args.games, args.seed)
    run(out["results"], "NRPA-2P-L2", make_nrpa2p(args.budget, level=2),
        "NRPA-L2", make_nrpa(args.budget, level=2), args.games, args.seed)

    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
