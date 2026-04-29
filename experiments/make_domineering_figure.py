"""Génère une illustration pédagogique du jeu Domineering.

Trois sous-figures :
  (a) Plateau initial 6x6 vide, avec annotations des conventions V/H.
  (b) Position de mi-partie avec dominos V (verticaux) et H (horizontaux).
  (c) Affichage des coups légaux pour Vertical depuis cette position.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domineering.board import Board, Move, set_board_size, VERTICAL, HORIZONTAL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures", "fig_domineering.png")


def draw_grid(ax, dx, dy):
    for x in range(dx + 1):
        ax.axhline(x, color="black", linewidth=0.6)
    for y in range(dy + 1):
        ax.axvline(y, color="black", linewidth=0.6)
    ax.set_xlim(0, dy)
    ax.set_ylim(dx, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


V_COLOR = "#1f77b4"
H_COLOR = "#d62728"


def draw_domino(ax, color, r, c):
    """Dessine un domino dont la case haut-gauche est (r, c)."""
    if color == VERTICAL:
        rect = patches.Rectangle((c + 0.06, r + 0.06), 0.88, 1.88,
                                 facecolor=V_COLOR, alpha=0.85,
                                 edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(c + 0.5, r + 1.0, "V", color="white",
                ha="center", va="center", fontsize=14, fontweight="bold")
    else:
        rect = patches.Rectangle((c + 0.06, r + 0.06), 1.88, 0.88,
                                 facecolor=H_COLOR, alpha=0.85,
                                 edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(c + 1.0, r + 0.5, "H", color="white",
                ha="center", va="center", fontsize=14, fontweight="bold")


def main():
    set_board_size(6, 6)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    # (a) Conventions
    ax = axes[0]
    draw_grid(ax, 6, 6)
    # Une seule pièce de chaque type pour montrer les conventions
    draw_domino(ax, VERTICAL, 1, 1)
    draw_domino(ax, HORIZONTAL, 4, 2)
    ax.set_title("(a) Conventions :\n"
                 "V joue 2$\\times$1 vertical,\n"
                 "H joue 1$\\times$2 horizontal")

    # (b) Position de mi-partie : on simule quelques coups depuis l'état initial
    b = Board()
    moves_played = [
        Move(VERTICAL, 0, 0),
        Move(HORIZONTAL, 5, 1),
        Move(VERTICAL, 2, 5),
        Move(HORIZONTAL, 0, 2),
        Move(VERTICAL, 3, 0),
        Move(HORIZONTAL, 2, 1),
        Move(VERTICAL, 0, 4),
        Move(HORIZONTAL, 4, 4),
    ]
    for m in moves_played:
        if any(m.r == p.r and m.c == p.c and m.color == p.color for p in b.legal_moves()):
            b.play(m)
    ax = axes[1]
    draw_grid(ax, 6, 6)
    seen = set()
    for r in range(6):
        for c in range(6):
            if (r, c) in seen:
                continue
            v = b.board[r, c]
            if v == VERTICAL and r + 1 < 6 and b.board[r + 1, c] == VERTICAL and (r + 1, c) not in seen:
                draw_domino(ax, VERTICAL, r, c)
                seen.add((r, c)); seen.add((r + 1, c))
            elif v == HORIZONTAL and c + 1 < 6 and b.board[r, c + 1] == HORIZONTAL and (r, c + 1) not in seen:
                draw_domino(ax, HORIZONTAL, r, c)
                seen.add((r, c)); seen.add((r, c + 1))
    ax.set_title("(b) Position de mi-partie\n(au trait : V)")

    # (c) Coups légaux pour V depuis (b)
    ax = axes[2]
    draw_grid(ax, 6, 6)
    for r in range(6):
        for c in range(6):
            if (r, c) in seen:
                continue
            v = b.board[r, c]
            if v == VERTICAL and r + 1 < 6 and b.board[r + 1, c] == VERTICAL and (r + 1, c) not in seen:
                draw_domino(ax, VERTICAL, r, c)
                seen.add((r, c)); seen.add((r + 1, c))
            elif v == HORIZONTAL and c + 1 < 6 and b.board[r, c + 1] == HORIZONTAL and (r, c + 1) not in seen:
                draw_domino(ax, HORIZONTAL, r, c)
                seen.add((r, c)); seen.add((r, c + 1))
    seen = set()
    for r in range(6):
        for c in range(6):
            if (r, c) in seen:
                continue
            v = b.board[r, c]
            if v == VERTICAL and r + 1 < 6 and b.board[r + 1, c] == VERTICAL and (r + 1, c) not in seen:
                seen.add((r, c)); seen.add((r + 1, c))
            elif v == HORIZONTAL and c + 1 < 6 and b.board[r, c + 1] == HORIZONTAL and (r, c + 1) not in seen:
                seen.add((r, c)); seen.add((r, c + 1))
    for m in b.legal_moves():
        rect = patches.Rectangle((m.c + 0.18, m.r + 0.18),
                                 0.64, 1.64,
                                 facecolor="none",
                                 edgecolor="green",
                                 linewidth=2.0,
                                 linestyle="--")
        ax.add_patch(rect)
    ax.set_title(f"(c) Coups légaux pour V\n({len(b.legal_moves())} possibilités)")

    fig.suptitle("Domineering (Crosscram) sur 6$\\times$6", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, bbox_inches="tight", dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
