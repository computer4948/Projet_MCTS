"""Domineering board with Zobrist hashing.

Conventions
-----------
* Le plateau est de taille ``Dx`` (lignes) x ``Dy`` (colonnes).
* Deux joueurs : ``VERTICAL`` (joue le premier) place des dominos 2x1 (verticaux),
  ``HORIZONTAL`` place des dominos 1x2 (horizontaux).
* Convention de fin (normal play) : le joueur qui ne peut plus jouer perd.
* La fonction ``score`` renvoie 1.0 si VERTICAL gagne, 0.0 si HORIZONTAL gagne,
  0.5 sinon (partie en cours), à l'image de la convention White/Black du cours.

Le codage des coups (utilisé par AMAF / RAVE / GRAVE) :
``code = color * Dx * Dy + r * Dy + c`` où ``(r, c)`` est la case haut-gauche
du domino. ``MaxCodeMoves = 3 * Dx * Dy`` couvre largement les codes possibles.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

EMPTY = 0
VERTICAL = 1
HORIZONTAL = 2

# Taille par défaut. Peut être surchargée à la création d'un Board, mais le
# hashTable est calé sur (Dx, Dy) au chargement du module pour la cohérence
# entre toutes les instances d'une même partie.
Dx = 8
Dy = 8

# Mode du jeu : "normal" (le joueur qui ne peut plus jouer perd) ou
# "misere" (le joueur qui ne peut plus jouer gagne). Cf. cours
# Cazenave (slide *Misère Domineering*).
GAME_MODE = "normal"


def set_game_mode(mode: str) -> None:
    global GAME_MODE
    if mode not in ("normal", "misere"):
        raise ValueError(f"Mode inconnu : {mode}")
    GAME_MODE = mode


def set_board_size(dx: int, dy: int, seed: Optional[int] = 42) -> None:
    """Reconfigure la taille du plateau et régénère la Zobrist table.

    Doit être appelé une seule fois en début de programme (avant toute
    utilisation des algorithmes basés sur la table de transposition).
    """
    global Dx, Dy, _hashTable, _hashTurn, MaxLegalMoves, MaxCodeMoves
    Dx, Dy = dx, dy
    rng = random.Random(seed)
    _hashTable = [
        [[rng.randint(0, 2 ** 64 - 1) for _ in range(Dy)] for _ in range(Dx)]
        for _ in range(3)
    ]
    _hashTurn = rng.randint(0, 2 ** 64 - 1)
    MaxLegalMoves = max(Dx * (Dy - 1), (Dx - 1) * Dy)
    MaxCodeMoves = 3 * Dx * Dy


# Initialisation par défaut
set_board_size(Dx, Dy)


class Move:
    """Un coup : un joueur place un domino dont la case haut-gauche est (r, c).

    Pour ``VERTICAL`` le domino occupe (r, c) et (r+1, c).
    Pour ``HORIZONTAL`` le domino occupe (r, c) et (r, c+1).
    """

    __slots__ = ("color", "r", "c")

    def __init__(self, color: int, r: int, c: int) -> None:
        self.color = color
        self.r = r
        self.c = c

    def cells(self):
        if self.color == VERTICAL:
            return (self.r, self.c), (self.r + 1, self.c)
        return (self.r, self.c), (self.r, self.c + 1)

    def code(self) -> int:
        return self.color * Dx * Dy + self.r * Dy + self.c

    def __repr__(self) -> str:
        sym = "V" if self.color == VERTICAL else "H"
        return f"{sym}({self.r},{self.c})"


class Board:
    """État d'une partie de Domineering."""

    __slots__ = ("board", "turn", "h", "_legal_cache", "_legal_cache_turn")

    def __init__(self) -> None:
        self.board = np.zeros((Dx, Dy), dtype=np.int8)
        self.turn = VERTICAL
        self.h = 0
        self._legal_cache: Optional[List[Move]] = None
        self._legal_cache_turn = -1

    # --- Copie -----------------------------------------------------------
    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.board = self.board.copy()
        b.turn = self.turn
        b.h = self.h
        b._legal_cache = None
        b._legal_cache_turn = -1
        return b

    # --- Génération des coups -------------------------------------------
    def legal_moves(self) -> List[Move]:
        if self._legal_cache is not None and self._legal_cache_turn == self.turn:
            return self._legal_cache
        moves: List[Move] = []
        bd = self.board
        if self.turn == VERTICAL:
            for r in range(Dx - 1):
                row = bd[r]
                row1 = bd[r + 1]
                for c in range(Dy):
                    if row[c] == EMPTY and row1[c] == EMPTY:
                        moves.append(Move(VERTICAL, r, c))
        else:
            for r in range(Dx):
                row = bd[r]
                for c in range(Dy - 1):
                    if row[c] == EMPTY and row[c + 1] == EMPTY:
                        moves.append(Move(HORIZONTAL, r, c))
        self._legal_cache = moves
        self._legal_cache_turn = self.turn
        return moves

    # --- Mécanique du jeu -----------------------------------------------
    def play(self, m: Move) -> None:
        c = m.color
        (r1, c1), (r2, c2) = m.cells()
        self.board[r1, c1] = c
        self.board[r2, c2] = c
        self.h ^= _hashTable[c][r1][c1]
        self.h ^= _hashTable[c][r2][c2]
        self.h ^= _hashTurn
        self.turn = HORIZONTAL if c == VERTICAL else VERTICAL
        self._legal_cache = None

    def terminal(self) -> bool:
        return len(self.legal_moves()) == 0

    def score(self) -> float:
        """1.0 si VERTICAL gagne, 0.0 si HORIZONTAL gagne, 0.5 sinon.
        En mode misère, la convention est inversée : le joueur qui ne
        peut plus jouer gagne."""
        if not self.terminal():
            return 0.5
        if GAME_MODE == "normal":
            return 0.0 if self.turn == VERTICAL else 1.0
        # misere : le joueur sans coup gagne
        return 1.0 if self.turn == VERTICAL else 0.0

    def winner(self) -> Optional[int]:
        if not self.terminal():
            return None
        return HORIZONTAL if self.turn == VERTICAL else VERTICAL

    # --- Playout aléatoire pur ------------------------------------------
    def playout(self) -> float:
        while True:
            moves = self.legal_moves()
            if len(moves) == 0:
                if GAME_MODE == "normal":
                    return 0.0 if self.turn == VERTICAL else 1.0
                return 1.0 if self.turn == VERTICAL else 0.0
            self.play(moves[random.randrange(len(moves))])

    # --- Affichage ------------------------------------------------------
    def __str__(self) -> str:
        # ASCII : 'V' pour les deux moitiés d'un domino vertical, 'H' pour
        # un domino horizontal, '.' pour case vide.
        symbols = np.full((Dx, Dy), ".", dtype=object)
        for r in range(Dx):
            for c in range(Dy):
                col = self.board[r, c]
                if col == VERTICAL:
                    symbols[r, c] = "V"
                elif col == HORIZONTAL:
                    symbols[r, c] = "H"
        rows = []
        rows.append("    " + " ".join(f"{c:2d}" for c in range(Dy)))
        rows.append("   +" + "---" * Dy + "+")
        for r in range(Dx):
            rows.append(f"{r:2d} |" + "".join(f" {symbols[r, c]} " for c in range(Dy)) + "|")
        rows.append("   +" + "---" * Dy + "+")
        rows.append(f"  Trait : {'V' if self.turn == VERTICAL else 'H'}")
        return "\n".join(rows)

    def print(self) -> None:
        print(self.__str__())

    # --- Accès table de hash ---------------------------------------------
    @property
    def hash(self) -> int:
        return self.h


# Tables Zobrist (mises à jour par set_board_size)
_hashTable: list  # type: ignore[assignment]
_hashTurn: int
MaxLegalMoves: int
MaxCodeMoves: int
