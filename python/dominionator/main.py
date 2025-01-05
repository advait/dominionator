import logging
import warnings

import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings
import torch
import typer

from dominionator.nn import ModelConfig
from dominionator.training import train_single_gen

logging.basicConfig(format="%(name)s [%(levelname)s] %(message)s", level=logging.DEBUG)
maturin_import_hook.reset_logger()
maturin_import_hook.install(
    settings=MaturinSettings(
        uv=True,
    ),
    enable_automatic_installation=True,
)

import dominionator.dominionator_rust as rust  # noqa: E402 F403  # type: ignore

app = typer.Typer()


@app.command()
def train(
    base_dir: str = "training",
    device: str = "cuda",
    dim_embedding: int = 128,
    n_transformer_heads: int = 8,
    n_transformer_layers: int = 8,
    lr: float = 1e-3,
    l2_reg: float = 1e-5,
    n_self_play_games: int = 100,
    n_mcts_iterations: int = 100,
    c_exploration: float = 1.0,
    self_play_batch_size: int = 100,
    training_batch_size: int = 100,
):
    config = ModelConfig(
        n_embeddings=rust.N_EMBEDDINGS,
        dim_embedding=dim_embedding,
        n_transformer_heads=n_transformer_heads,
        n_transformer_layers=n_transformer_layers,
        dim_policy=rust.N_ACTIONS,
        lr=lr,
        l2_reg=l2_reg,
    )
    train_single_gen(
        base_dir,
        torch.device(device),
        config,
        n_self_play_games,
        n_mcts_iterations,
        c_exploration,
        self_play_batch_size,
        training_batch_size,
    )


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    app()
