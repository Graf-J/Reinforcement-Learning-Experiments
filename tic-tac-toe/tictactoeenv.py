"""
tictactoe_env.py

A minimal, Gym-style Tic Tac Toe environment for reinforcement learning.

This module contains ONLY environment dynamics (state transitions, legal
actions, terminal detection, rewards). No agent, no Q-table, no learning
logic lives here.

State representation
---------------------
A state is a length-9 tuple of ints, one per board cell (row-major,
index 0 = top-left, index 8 = bottom-right):
    0  -> empty
    1  -> X
    -1 -> O

Tuples are hashable, so a state can be used directly as a dict key in a
Q-table: Q[(state, action)] = value.

Turn tracking
-------------
The environment tracks whose turn it is internally (self.current_player).
`step(action)` places CURRENT player's mark, then flips whose turn it is
(unless the game just ended).

Reward convention
------------------
`step()` returns the reward from the perspective of the player who just
moved (the one who took `action`), NOT the player about to move next.
    +1  -> the mover just won
    -1  -> the mover just made a move that lets the opponent win next
           (not used in standard TTT since a move can't cause an
           immediate loss for the mover -- included for completeness /
           custom reward shaping)
     0  -> draw, or game continues

If you are training a single agent against a fixed opponent, wrap this
environment and re-derive the reward relative to your agent's identity.
If you are doing self-play (training X and O simultaneously, as you
mentioned wanting to explore later), you can use the per-move reward
directly: whichever player just moved receives it.

Example
-------
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    while not done:
        action = your_policy(state, env.legal_actions())
        state, reward, done, info = env.step(action)
    print(info["winner"])   # 1, -1, or 0 (draw)
"""

from itertools import product
from typing import Any

State = tuple[int, ...]

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]


class TicTacToeEnv:
    """A two-player Tic Tac Toe environment with a Gym-like reset/step API."""

    def __init__(self):
        self.board: list[int] = [EMPTY] * 9
        self.current_player: int = PLAYER_X
        self.done: bool = False
        self.winner: int | None = None  # 1, -1, or 0 for draw, None if ongoing

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> State:
        """Reset the board to empty, X to move. Returns the initial state."""
        self.board = [EMPTY] * 9
        self.current_player = PLAYER_X
        self.done = False
        self.winner = None
        return self._get_state()

    def legal_actions(self, state: State | None = None) -> list[int]:
        """Return the list of empty cell indices (0-8) available to play."""
        board = state if state is not None else self.board
        return [i for i, v in enumerate(board) if v == EMPTY]

    def step(self, action: int) -> tuple[State, float, bool, dict[str, Any]]:
        """
        Apply `action` (an int 0-8) for the current player.

        Returns:
            next_state: the resulting board state (tuple)
            reward: float, from the perspective of the player who just moved
            done: bool, whether the episode has ended
            info: dict with extra diagnostics, including:
                  - "winner": 1, -1, 0 (draw), or None (not done)
                  - "player_who_moved": which player just took this action

        Raises:
            ValueError if the game is already over or the action is illegal.
        """
        if self.done:
            raise ValueError("Cannot call step() on a finished episode. Call reset().")
        if action not in self.legal_actions():
            raise ValueError(
                f"Illegal action {action}. Cell is occupied or out of range."
            )

        mover = self.current_player
        self.board[action] = mover

        winner = self._check_winner()
        reward = 0.0

        if winner is not None:
            self.done = True
            self.winner = winner
            if winner == mover:
                reward = 1.0
            elif winner == 0:
                reward = 0.0  # draw
            # note: winner can't equal the *other* player immediately after
            # this move in standard TTT, but is handled for completeness
            else:
                reward = -1.0
        else:
            # game continues -- hand turn to the other player
            self.current_player = -mover

        info = {"winner": self.winner, "player_who_moved": mover}
        return self._get_state(), reward, self.done, info

    def render(self) -> None:
        """Print a human-readable board to stdout."""
        symbols = {EMPTY: ".", PLAYER_X: "X", PLAYER_O: "O"}
        rows = []
        for r in range(3):
            row = self.board[r * 3 : (r + 1) * 3]
            rows.append(" ".join(symbols[v] for v in row))
        print("\n".join(rows))
        print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> State:
        return tuple(self.board)

    def _check_winner(self) -> int | None:
        """
        Returns:
            1 or -1 if that player has three in a row,
            0 if the board is full with no winner (draw),
            None if the game is still ongoing.
        """
        for a, b, c in WIN_LINES:
            if (
                self.board[a] != EMPTY
                and self.board[a] == self.board[b] == self.board[c]
            ):
                return self.board[a]
        if EMPTY not in self.board:
            return 0  # draw
        return None

    @staticmethod
    def all_states() -> list[State]:
        """
        Enumerate every reachable board configuration (including illegal /
        unreachable ones from {-1,0,1}^9). Useful if you want to pre-size a
        Q-table or sanity-check coverage. Not filtered for reachability --
        for a small game like this it's cheap enough not to bother.
        """
        return list(product([EMPTY, PLAYER_X, PLAYER_O], repeat=9))


# ----------------------------------------------------------------------
# Quick manual smoke test / demo -- not part of the API, just for sanity
# checking the environment by hand from the command line.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = TicTacToeEnv()
    state = env.reset()
    env.render()

    done = False
    while not done:
        legal = env.legal_actions()
        print(f"Player {'X' if env.current_player == PLAYER_X else 'O'} to move.")
        print(f"Legal actions: {legal}")
        action = int(input("Enter cell index (0-8): "))
        state, reward, done, info = env.step(action)
        env.render()
        print(f"reward={reward}, done={done}, info={info}\n")

    if info["winner"] == 0:
        print("Draw!")
    else:
        print(f"Player {'X' if info['winner'] == PLAYER_X else 'O'} wins!")
