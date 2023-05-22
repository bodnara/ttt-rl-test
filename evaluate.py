"""Evaluation of trained bot in play against: random, mcts and self-better."""
import itertools
import logging
import os
import random
import shutil
from dataclasses import dataclass

import pandas as pd
import pyspiel
import tqdm
from azero import load_mcts_bot, load_trained_bot

import game  # register game
import wandb


_eval_config = dict(
    uct_c=0.2,
    max_simulations=20,
    nn_model="mlp",
    nn_width=64,
    nn_depth=8,

    # fixed
    game="ttt",

    # these are not used for evaluation
    path=None,
    learning_rate=None,
    weight_decay=None,
    train_batch_size=None,
    replay_buffer_size=None,
    replay_buffer_reuse=None,
    max_steps=None,
    checkpoint_freq=None,
    actors=None,
    evaluators=None,
    policy_alpha=None,
    policy_epsilon=None,
    temperature=None,
    temperature_drop=None,
    evaluation_window=None,
    eval_levels=None,
    observation_shape=None,
    output_size=None,
    quiet=True,
)


def _load_bot(config, checkpoint_dir_path: str, checkpoint: int):
    _cfg = _eval_config | config  # works on python>=3.9.0
    return load_trained_bot(_cfg, checkpoint_dir_path, checkpoint, is_eval=True)


def _restore_checkpoint_files(path: str, chkt: int, run_path: str, move_path: str):
    path = os.path.join(path, f"checkpoint-{chkt}")

    if os.path.exists(move_path):
        logging.warn(f"Destination folder {move_path=} exists, skipping...")
        return

    for suffix in [".index", ".meta", ".data-00000-of-00001"]:
        wandb.restore(path + suffix, run_path=run_path)

    dir_path = os.path.dirname(path)
    logging.info(f"Moving loaded files from {dir_path=} to {move_path=}")
    shutil.move(dir_path, move_path)


RUNS = {
    ## run_path, MCTS simuls, arch, depth, width, uct_
    "random": None,
    "warm-flower-18": ("logs/", "miba/ttt6x6/c13ng2xs", 50, "conv2d", 8, 256, 1.4),
    "hopeful-spaceship-20": ("logs/", "miba/ttt6x6/qolemwgq", 200, "resnet", 4, 256, 1.4)
}

MCTS_SIMULS = [0, 5, 10, 15, 20, 50, 120, 250]
MCTS_RATE = 1.4
GAMES = 20


@dataclass
class Result:
    player: str
    mcts_simuls: int
    mcts_rate: float
    player_first: bool
    result_from_player: int
    score_diff_from_player: int
    moves: list


def _eval():
    random.seed(0)
    results = []
    game = pyspiel.load_game(_eval_config["game"])

    for player_str in tqdm.tqdm(RUNS.keys(), desc="players"):
        if player_str == "random":
            play_fn = lambda state: random.choice(state.legal_actions())
        else:
            path, run_path, sims, arch, depth, width, uct = RUNS[player_str]
            _restore_checkpoint_files(path, -1, run_path, player_str)
            cfg = dict(uct_c=uct, max_simulations=sims, nn_model=arch, nn_width=width, nn_depth=depth)
            bot, _ = _load_bot(cfg, player_str, -1)
            play_fn = bot.step

        for mcts_simuls in tqdm.tqdm(MCTS_SIMULS, desc="mcts", leave=None):
            if mcts_simuls == 0:
                mcts_fn = lambda state: random.choice(state.legal_actions())
            else:
                mcts_cfg = dict(uct_c=MCTS_RATE, max_simulations=mcts_simuls)
                mcts_bot = load_mcts_bot(_eval_config | mcts_cfg, is_eval=True)
                mcts_fn = mcts_bot.step

            for i in tqdm.trange(GAMES, leave=None, desc=f"{player_str} vs. mcts({mcts_simuls})"):
                state = game.new_initial_state()
                players = [play_fn, mcts_fn] if i % 2 == 0 else [mcts_fn, play_fn]
                actions = []

                for p in itertools.cycle(players):
                    if state.is_terminal():
                        break
                    action = p(state)
                    state.apply_action(action)
                    actions.append(action)

                player_res = state.returns()[i % 2]
                p1, p2 = state._scores
                score_diff = p1 - p2 if i % 2 == 0 else p2 - p1
                res = Result(player_str, mcts_simuls, MCTS_RATE, i % 2 == 0, player_res, score_diff, actions)
                results.append(res)

    df = pd.DataFrame([vars(x) for x in results])
    print(df)
    df.to_csv("evaluation.csv")




if __name__ == "__main__":
    import wandb
    logging.basicConfig(level=logging.NOTSET, format='[%(asctime)s] %(levelname)s: %(message)s')
    _eval()
    # # restored = wandb.restore('logs/checkpoint--1.index', run_path="miba/ttt6x6/qolemwgq")
    # _restore_checkpoint_files('logs/checkpoint--1', run_path="miba/ttt6x6/qolemwgq", move_path="test_run")
    # # print(restored)
