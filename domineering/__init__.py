"""Domineering - Monte Carlo Tree Search project (M2 IASD).

Modules
-------
board : moteur du jeu Domineering avec hashing Zobrist.
algorithms : Flat MC, UCB, UCT, RAVE, GRAVE, NMCS, NRPA, PUCT.
optimizations : heavy playouts, decisive moves, etc.
tournament : utilitaires d'auto-jeu pour comparer deux agents.
"""

from .board import Board, Move, VERTICAL, HORIZONTAL, EMPTY

__all__ = ["Board", "Move", "VERTICAL", "HORIZONTAL", "EMPTY"]
