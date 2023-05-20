import pygame
from game import _SIZE
import pyspiel
import time
import numpy as np
import azero


CELL_SIZE = 100
SCREEN_WIDTH = _SIZE * CELL_SIZE + 2 * CELL_SIZE
SCREEN_HEIGHT = _SIZE * CELL_SIZE
FONT_SIZE = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SYMBOLS = [" ", "X", "O"]
COLORS = [WHITE, RED, BLUE]


class _InteractivePlay:
    def __init__(self, game: pyspiel.Game, player1 = None, player2 = None, delay = 0):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, FONT_SIZE)
        self._delay = delay
        self._players = [player1, player2]
        self.state = game.new_initial_state()

    def is_terminal(self):
        return self.state.is_terminal()

    def _render_text(self, text, cx, cy, color):
        number_surface = self.font.render(str(text), True, color)
        number_rect = number_surface.get_rect(center=(cx, cy))
        self.screen.blit(number_surface, number_rect)

    def _update_display(self):
        self.screen.fill(WHITE)

        # draw the board
        for row in range(_SIZE):
            for col in range(_SIZE):
                pygame.draw.rect(self.screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), width=1)
                cx = col * CELL_SIZE + CELL_SIZE // 2
                cy = row * CELL_SIZE + CELL_SIZE // 2
                _occupancy = self.state.board[row, col]
                self._render_text(SYMBOLS[_occupancy], cx, cy, COLORS[_occupancy])

        # draw the score
        cx = SCREEN_WIDTH - CELL_SIZE // 2
        cy = SCREEN_HEIGHT // 2
        pygame.draw.rect(self.screen, BLACK, (SCREEN_WIDTH - CELL_SIZE, 0, CELL_SIZE, SCREEN_HEIGHT))
        _score = f"{self.state._scores[0]}-{self.state._scores[1]}"
        self._render_text(_score, cx, cy, WHITE)

        # Update the display
        pygame.display.update()

    def next_move(self) -> None:
        self._update_display()
        player_to_play = self.state._next_player
        assert player_to_play in (0, 1)
        play_fn = self._players[player_to_play]

        if play_fn is not None:
            action = play_fn(self.state)
        else:
            action = self._get_human_action()

        self.state.apply_action(action)
        self._update_display()
        if self._delay > 0:
            time.sleep(self._delay)

    def _get_human_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise RuntimeError("Closing the window during the game")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos

                    # Check if the click was within the grid
                    if x < CELL_SIZE * _SIZE and y < CELL_SIZE * _SIZE:
                        row = y // CELL_SIZE
                        col = x // CELL_SIZE

                        if self.state.board[row, col] == 0:
                            action = self.state._to_action((row, col))
                            return action


if __name__ == "__main__":
    config = dict(
        game="ttt",
        path="./logs",
        learning_rate=0.001,
        weight_decay=1e-4,
        train_batch_size=256,
        replay_buffer_size=2**14,
        replay_buffer_reuse=4,
        max_steps=300,
        checkpoint_freq=3,

        actors=4,
        evaluators=4,
        uct_c=0.2,
        max_simulations=20,
        policy_alpha=0.25,
        policy_epsilon=1,
        temperature=1,
        temperature_drop=4,
        evaluation_window=50,
        eval_levels=7,

        nn_model="mlp",
        nn_width=64,
        nn_depth=8,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )

    mcts_bot = azero.load_mcts_bot(config)
    trained, _ = azero.load_trained_bot(config, "/tmp/az-2023-05-20-21-28-ttt-e_6rz_ww", -1)

    def _random_play(state):
        return np.random.choice(state.legal_actions())

    game = pyspiel.load_game("ttt")
    visual = _InteractivePlay(game, trained.step, None, delay=0)
    while not visual.is_terminal():
        visual.next_move()
    while True: pass
