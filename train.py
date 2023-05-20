import azero
import wandb
import game  # for registering the game
import os


os.environ["WANDB_IGNORE_GLOBS"] = "venv/"


if __name__ == "__main__":
    checkpoint = None
    checkpoint_dir = "logs/"

    config = dict(
        game="ttt",
        path=checkpoint_dir,
        learning_rate=0.001,
        weight_decay=1e-4,
        train_batch_size=16,
        replay_buffer_size=2**4,
        replay_buffer_reuse=4,
        max_steps=1,
        checkpoint_freq=3,

        actors=4,
        evaluators=4,
        uct_c=1.4,
        max_simulations=20,
        policy_alpha=0.25,
        policy_epsilon=1,
        temperature=1,
        temperature_drop=4,
        evaluation_window=50,
        eval_levels=7,

        nn_model="mlp",
        nn_width=256,
        nn_depth=8,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )

    wandb.init(config=config, project="ttt6x6")
    wandb.run.log_code(".")
    wandb.save(checkpoint_dir)
    with azero.spawn.main_handler():
        azero.alpha_zero(azero.Config(**config), is_win_loose=True, checkpoint=checkpoint, start_step=1)
    wandb.finish()
