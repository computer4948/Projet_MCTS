"""Algorithmes Monte Carlo pour Domineering.

Tous les agents exposent une fonction ``best_move(board, budget, **kwargs)``
qui renvoie le meilleur coup à jouer pour ``board.turn`` étant donné un
budget de ``budget`` simulations.
"""
from .flat import flat_best_move
from .ucb import ucb_best_move
from .uct import uct_best_move
from .rave import rave_best_move
from .grave import grave_best_move
from .nmcs import nmcs_best_move
from .nrpa import nrpa_best_move
from .puct import puct_best_move
from .sequential_halving import sh_best_move
from .ppatcs import ppatcs_best_move
from .mast import mast_best_move
from .shot import shot_best_move
from .gnrpa import gnrpa_best_move
from .nrpa_2p import nrpa2p_best_move

__all__ = [
    "flat_best_move",
    "ucb_best_move",
    "uct_best_move",
    "rave_best_move",
    "grave_best_move",
    "nmcs_best_move",
    "nrpa_best_move",
    "puct_best_move",
    "sh_best_move",
    "ppatcs_best_move",
    "mast_best_move",
    "shot_best_move",
    "gnrpa_best_move",
    "nrpa2p_best_move",
]
