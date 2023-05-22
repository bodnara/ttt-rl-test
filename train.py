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
        learning_rate=0.0001,
        weight_decay=1e-4,
        train_batch_size=16,
        replay_buffer_size=2**8,
        replay_buffer_reuse=4,
        max_steps=1000,
        checkpoint_freq=50,

        actors=2,
        evaluators=2,
        uct_c=1.4,
        max_simulations=20,
        policy_alpha=0.25,
        policy_epsilon=1,
        temperature=1,
        temperature_drop=4,
        evaluation_window=50,
        eval_levels=7,

        nn_model="conv2d",
        nn_width=512,
        nn_depth=8,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )

    wandb.init(config=config, project="ttt6x6")
    # wandb.run.log_code("*.py")
    with azero.spawn.main_handler():
        azero.alpha_zero(azero.Config(**config), is_win_loose=True, checkpoint=checkpoint, start_step=1)
    for _ in range(2): # first one symlinks to W&B directory, second saves now
        wandb.save(checkpoint_dir + "*")
    wandb.finish()
