import pyspiel
import numpy as np


_SIZE = 6
_NEEDED = 4

_MARK_EMPTY = 0
_MARK_X = 1
_MARK_O = 2


_GAME_TYPE = pyspiel.GameType(
    short_name="ttt",
    long_name=f"Tic-Tac-Toe {_SIZE}x{_SIZE}",

    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,

    max_num_players=2,
    min_num_players=2,

    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={}
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_SIZE * _SIZE,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_SIZE * _SIZE
)

class TTTGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return TTTState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        _iig = iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False)
        return TTTObserver(_iig, params)


class TTTState(pyspiel.State):
    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._game_over = False
        self._next_player = 0
        self._scores = [0, 0]
        self._moves_played = 0
        self.board = np.full((_SIZE, _SIZE), _MARK_EMPTY)

    def _to_action(self, coord):
        row, col = coord
        return _SIZE * row + col

    def _coord(self, action):
        return divmod(action, _SIZE)

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._next_player

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0
        return [i for i in range(_SIZE * _SIZE) if self.board[self._coord(i)] == _MARK_EMPTY]

    def _check_line(self, x, y, dx, dy):
        mark = self.board[y, x]
        _found = 1
        for m in [-1, 1]:
            _x = x
            _y = y
            while True:
                _x += m * dx
                _y += m * dy
                if not (0 <= _x < _SIZE) or not (0 <= _y < _SIZE):
                    break
                if self.board[_y, _x] == mark:
                    _found += 1
                else:
                    break
        return _found >= _NEEDED

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        y, x = self._coord(action)
        mark = _MARK_X if self._next_player == 0 else _MARK_O
        self.board[y, x] = mark
        self._moves_played += 1

        # check all directions for a winning condition
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self._check_line(x, y, dx, dy):
                self._scores[self._next_player] += 1

        if self._moves_played >= _SIZE * _SIZE:
            self._game_over = True
        self._next_player = 1 - self._next_player


    def _action_to_string(self, player, action):
        """Action -> string."""
        y, x = self._coord(action)
        p = "X" if player == 0 else "O"
        return f"{p}(row={y}, col={x})"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if not self._game_over or self._scores[0] == self._scores[1]:
            return [0, 0]
        p1 = 1 if self._scores[0] > self._scores[1] else -1
        return [p1, -p1]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        SYMBOLS = [" ", "X", "O"]
        res = ""
        for row in range(_SIZE):
            for col in range(_SIZE):
                res += SYMBOLS[self.board[row, col]]
            res += "\n"
        return res


class TTTObserver:
    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = (1 + 2, _SIZE, _SIZE)  # (player, row, col)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = { "observation": np.reshape(self.tensor, shape) }

    def one_hot(self, x):
        return np.identity(14)[x].flatten()

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        del player # unused
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs.fill(0)
        for row in range(_SIZE):
            for col in range(_SIZE):
                cell_state = state.board[row, col]
                obs[cell_state, row, col] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        return str(state)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, TTTGame)