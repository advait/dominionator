import warnings

import torch
import typer

from dominionator.nn import ModelConfig
from dominionator.training import TrainingGen, train_single_gen

import dominionator.dominionator_rust as rust  # noqa: E402 F403  # type: ignore

app = typer.Typer()


@app.command()
def train(
    base_dir: str = "training",
    device: str = "cuda",
    dim_embedding: int = 32,
    n_transformer_heads: int = 4,
    n_transformer_layers: int = 3,
    lr: float = 1e-3,
    l2_reg: float = 1e-5,
    n_self_play_games: int = 100,
    n_mcts_iterations: int = 10,
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
    parent_gen = TrainingGen.load_latest_with_default(
        base_dir,
        n_mcts_iterations,
        c_exploration,
        n_self_play_games,
        self_play_batch_size,
        training_batch_size,
        config,
    )
    train_single_gen(
        base_dir,
        torch.device(device),
        parent_gen,
    )


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    app()
