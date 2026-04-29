"""Expérience 1 : tournoi round-robin à budget fixé.

Tous les algorithmes principaux s'affrontent (sauf NMCS qui est piloté par
``level``). Sortie : matrice de taux de victoire + temps moyen par coup.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Permet d'exécuter ce script depuis experiments/ comme depuis la racine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domineering.board import set_board_size  # noqa: E402
from domineering.tournament import round_robin  # noqa: E402
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
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=str,
                        default="experiments/results/exp1_round_robin.json")
    args = parser.parse_args()

    set_board_size(args.size, args.size)

    agents = {
        "Flat": make_flat(args.budget),
        "UCB": make_ucb(args.budget),
        "UCT": make_uct(args.budget),
        "RAVE": make_rave(args.budget),
        "GRAVE": make_grave(args.budget),
        "PUCT": make_puct(args.budget),
        "NMCS-L1": make_nmcs(level=1),
        "NRPA-L1": make_nrpa(budget=args.budget, level=1),
    }

    t0 = time.perf_counter()
    results = round_robin(agents, n_games=args.games, seed=args.seed, verbose=True)
    duration = time.perf_counter() - t0

    out = {
        "params": vars(args),
        "duration_s": duration,
        "matches": [r.as_dict() for r in results],
        "agents": list(agents.keys()),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDurée totale : {duration:.1f}s")
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
