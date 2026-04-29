"""Expérience 8 : ablation des hyperparamètres clés.

Trois balayages séparés (chacun à budget 200, contre Flat-200) :
    * c_puct ∈ {0.3, 0.7, 1.5, 3.0, 5.0}      (PUCT)
    * ref_min ∈ {10, 25, 50, 100, 200}        (GRAVE)
    * temperature ∈ {0.5, 1.0, 2.0, 4.0}      (UCT-biased / GRAVE-biased)
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
from domineering.algorithms import (  # noqa: E402
    puct_best_move, grave_best_move, uct_best_move,
)
from domineering.optimizations import biased_playout  # noqa: E402
from experiments.agents import make_flat  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp8_hyperparams.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)
    baseline = make_flat(args.budget)
    out = {"params": vars(args), "puct_c": [], "grave_refmin": [], "biased_temp": []}
    t0 = time.perf_counter()

    print("--- PUCT c_puct sweep ---")
    for c in [0.3, 0.7, 1.5, 3.0, 5.0]:
        agent = lambda b, c=c: puct_best_move(b, args.budget, c_puct=c)
        m = play_match(f"PUCT_c{c}", agent, "Flat", baseline,
                       n_games=args.games, seed=args.seed)
        p, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"  c_puct={c}: wr={p:.2f} [{lo:.2f},{hi:.2f}]")
        out["puct_c"].append({"c": c, "winrate": p, "ci_lo": lo, "ci_hi": hi,
                              "wins": m.a_wins, "n": m.n_games})

    print("--- GRAVE ref_min sweep ---")
    for rm in [10, 25, 50, 100, 200]:
        agent = lambda b, rm=rm: grave_best_move(b, args.budget, ref_min=rm)
        m = play_match(f"GRAVE_rm{rm}", agent, "Flat", baseline,
                       n_games=args.games, seed=args.seed)
        p, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"  ref_min={rm}: wr={p:.2f} [{lo:.2f},{hi:.2f}]")
        out["grave_refmin"].append({"ref_min": rm, "winrate": p,
                                    "ci_lo": lo, "ci_hi": hi,
                                    "wins": m.a_wins, "n": m.n_games})

    print("--- UCT-biased temperature sweep ---")
    for tau in [0.5, 1.0, 2.0, 4.0]:
        from functools import partial
        playout = partial(biased_playout, temperature=tau)
        agent = lambda b, p=playout: uct_best_move(b, args.budget, playout_fn=p)
        m = play_match(f"UCTbiased_t{tau}", agent, "Flat", baseline,
                       n_games=args.games, seed=args.seed)
        p_, lo, hi = wilson_ci(m.a_wins, m.n_games)
        print(f"  T={tau}: wr={p_:.2f} [{lo:.2f},{hi:.2f}]")
        out["biased_temp"].append({"tau": tau, "winrate": p_,
                                   "ci_lo": lo, "ci_hi": hi,
                                   "wins": m.a_wins, "n": m.n_games})

    out["duration_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée : {out['duration_s']:.1f}s, sortie : {args.out}")


if __name__ == "__main__":
    main()
