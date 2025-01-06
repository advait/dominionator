"""
Generation-based training loop that alternates between self-play and training.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
import os
import pickle
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin, config
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader

from dominionator.nn import Batch, DominionatorModel, ModelConfig

import dominionator.dominionator_rust as rust


@dataclass
class TrainingGen(DataClassJsonMixin):
    """
    Represents a single generation of training.
    """

    created_at: datetime = field(
        metadata=config(
            encoder=datetime.isoformat,
            decoder=datetime.fromisoformat,
        )
    )
    gen_n: int
    n_mcts_iterations: int
    c_exploration: float
    n_self_play_games: int
    self_play_batch_size: int
    training_batch_size: int
    parent: Optional[datetime] = None
    val_loss: Optional[float] = None

    @staticmethod
    def _gen_folder(created_at: datetime, base_dir: str) -> str:
        return os.path.join(base_dir, created_at.isoformat())

    def gen_folder(self, base_dir: str) -> str:
        return TrainingGen._gen_folder(self.created_at, base_dir)

    def save_all(
        self,
        base_dir: str,
        games: Optional[rust.PlayGamesResult],
        model: DominionatorModel,
    ):
        self.save_metadata(base_dir)
        gen_dir = self.gen_folder(base_dir)

        play_result_path = os.path.join(gen_dir, "games.pkl")
        with open(play_result_path, "wb") as f:
            pickle.dump(games, f)

        model_path = os.path.join(gen_dir, "model.pt")
        with open(model_path, "wb") as f:
            torch.save(model, f)

    def save_metadata(self, base_dir: str):
        gen_dir = self.gen_folder(base_dir)
        os.makedirs(gen_dir, exist_ok=True)

        metadata_path = os.path.join(gen_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            f.write(self.to_json(indent=2))

    @staticmethod
    def load(base_dir: str, created_at: datetime) -> "TrainingGen":
        gen_folder = TrainingGen._gen_folder(created_at, base_dir)
        with open(os.path.join(gen_folder, "metadata.json"), "r") as f:
            return TrainingGen.from_json(f.read())

    @staticmethod
    def load_all(base_dir: str) -> List["TrainingGen"]:
        timestamps = sorted(
            [
                datetime.fromisoformat(f)
                for f in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, f))
            ],
            reverse=True,
        )
        return [TrainingGen.load(base_dir, t) for t in timestamps]

    @staticmethod
    def load_latest(base_dir: str) -> "TrainingGen":
        return TrainingGen.load_all(base_dir)[0]

    @staticmethod
    def load_latest_with_default(
        base_dir: str,
        n_mcts_iterations: int,
        c_exploration: float,
        n_self_play_games: int,
        self_play_batch_size: int,
        training_batch_size: int,
        model_config: ModelConfig,
    ):
        try:
            return TrainingGen.load_latest(base_dir)
        except (FileNotFoundError, IndexError):
            logger.info("No existing generations found, initializing root")
            gen = TrainingGen(
                created_at=datetime.now(),
                gen_n=0,
                n_mcts_iterations=n_mcts_iterations,
                c_exploration=c_exploration,
                n_self_play_games=n_self_play_games,
                self_play_batch_size=self_play_batch_size,
                training_batch_size=training_batch_size,
            )
            model = DominionatorModel(model_config)
            gen.save_all(base_dir, None, model)
            return gen

    def get_games(self, base_dir: str) -> Optional[rust.PlayGamesResult]:
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "games.pkl"), "rb") as f:
            return pickle.load(f)

    def get_model(self, base_dir: str) -> DominionatorModel:
        """Gets the model for this generation."""
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "model.pt"), "rb") as f:
            return torch.load(f, weights_only=False)


def train_single_gen(
    base_dir: str,
    device: torch.device,
    parent: TrainingGen,
) -> TrainingGen:
    """
    Trains a new generation from the given parent.
    First generate games using dominionator._rust.play_games.
    Then train a new model based on the parent model using the generated samples.
    Finally, save the resulting games and model in the training directory.
    """
    gen_n = parent.gen_n + 1
    logger.info(f"Beginning new generation {gen_n} from {parent.gen_n}")

    # TODO: log experiment metadata in MLFlow

    # Self play
    model = parent.get_model(base_dir)
    model.to(device)
    reqs = [rust.GameMetadata(id, 0, 0) for id in range(parent.n_self_play_games)]  # type: ignore
    games = rust.play_games(  # type: ignore
        reqs,
        parent.self_play_batch_size,
        parent.n_mcts_iterations,
        parent.c_exploration,
        lambda batch: model.forward_numpy(batch),  # type: ignore
    )

    # Training
    logger.info("Beginning training")
    model = copy.deepcopy(model)
    data_module = SampleDataModule(games, parent.training_batch_size)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    model.train()  # Switch batch normalization to train mode for training bn params
    trainer.fit(model, data_module)
    logger.info("Finished training")

    gen = TrainingGen(
        created_at=datetime.now(),
        gen_n=gen_n,
        n_mcts_iterations=parent.n_mcts_iterations,
        c_exploration=parent.c_exploration,
        n_self_play_games=parent.n_self_play_games,
        self_play_batch_size=parent.self_play_batch_size,
        training_batch_size=parent.training_batch_size,
        parent=parent.created_at,
        val_loss=trainer.callback_metrics["val_loss"].item(),
    )
    gen.save_all(base_dir, games, model)
    return gen


class SampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        result: rust.PlayGamesResult,
        batch_size: int,
    ):
        super().__init__()
        train, test = result.split_train_test(0.8, 1337)  # type: ignore
        self.batch_size = batch_size
        self.training_data = train
        self.validation_data = test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=SampleDataModule.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            collate_fn=SampleDataModule.collate_fn,
        )

    @staticmethod
    def collate_fn(samples: List[rust.Sample]) -> "Batch":
        state_raw_embeddings, policy_target_logprobs, ply1_log_neg_target = zip(
            *(
                (
                    sample.state_to_token_indices(),
                    sample.policy_logprobs,
                    (sample.ply1_log_neg,),
                )
                for sample in samples
            )
        )
        return Batch(
            state_raw_embeddings=torch.tensor(state_raw_embeddings),
            policy_target_logprobs=torch.tensor(policy_target_logprobs),
            ply1_log_neg_target=torch.tensor(ply1_log_neg_target),
        )
