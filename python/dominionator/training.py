"""
Generation-based training loop that alternates between self-play and training.
"""

from copy import copy
from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from typing import List, Optional

from loguru import logger
import torch

from dominionator.nn import DominionatorModel, ModelConfig

import dominionator.dominionator_rust as rust


# @dataclass
# class TrainingGen:
#     """
#     Represents a single generation of training.
#     """

#     created_at: datetime
#     gen_n: int
#     n_mcts_iterations: int
#     c_exploration: float
#     c_ply_penalty: float
#     self_play_batch_size: int
#     training_batch_size: int
#     parent: Optional[datetime] = None
#     val_loss: Optional[float] = None

#     @staticmethod
#     def _gen_folder(created_at: datetime, base_dir: str) -> str:
#         return os.path.join(base_dir, created_at.isoformat())

#     def gen_folder(self, base_dir: str) -> str:
#         return TrainingGen._gen_folder(self.created_at, base_dir)

#     def save_all(
#         self,
#         base_dir: str,
#         games: Optional[PlayGamesResult],
#         model: DominionatorModel,
#     ):
#         gen_dir = self.gen_folder(base_dir)
#         os.makedirs(gen_dir, exist_ok=True)

#         metadata_path = os.path.join(gen_dir, "metadata.json")
#         with open(metadata_path, "w") as f:
#             f.write(self.model_dump_json(indent=2))

#         play_result_path = os.path.join(gen_dir, "games.pkl")
#         with open(play_result_path, "wb") as f:
#             pickle.dump(games, f)

#         model_path = os.path.join(gen_dir, "model.pkl")
#         with open(model_path, "wb") as f:
#             pickle.dump(model, f)

#     def save_metadata(self, base_dir: str):
#         gen_dir = self.gen_folder(base_dir)
#         os.makedirs(gen_dir, exist_ok=True)

#         metadata_path = os.path.join(gen_dir, "metadata.json")
#         with open(metadata_path, "w") as f:
#             f.write(self.model_dump_json(indent=2))

#     @staticmethod
#     def load(base_dir: str, created_at: datetime) -> "TrainingGen":
#         gen_folder = TrainingGen._gen_folder(created_at, base_dir)
#         with open(os.path.join(gen_folder, "metadata.json"), "r") as f:
#             return TrainingGen.model_validate_json(f.read())

#     @staticmethod
#     def load_all(base_dir: str) -> List["TrainingGen"]:
#         timestamps = sorted(
#             [
#                 datetime.fromisoformat(f)
#                 for f in os.listdir(base_dir)
#                 if os.path.isdir(os.path.join(base_dir, f))
#             ],
#             reverse=True,
#         )
#         return [TrainingGen.load(base_dir, t) for t in timestamps]

#     @staticmethod
#     def load_latest(base_dir: str) -> "TrainingGen":
#         timestamps = sorted(
#             [
#                 datetime.fromisoformat(f)
#                 for f in os.listdir(base_dir)
#                 if os.path.isdir(os.path.join(base_dir, f))
#             ],
#             reverse=True,
#         )
#         if not timestamps:
#             raise FileNotFoundError("No existing generations")
#         return TrainingGen.load(base_dir, timestamps[0])

#     @staticmethod
#     def load_latest_with_default(
#         base_dir: str,
#         n_mcts_iterations: int,
#         c_exploration: float,
#         c_ply_penalty: float,
#         self_play_batch_size: int,
#         training_batch_size: int,
#         model_config: ModelConfig,
#     ):
#         try:
#             return TrainingGen.load_latest(base_dir)
#         except FileNotFoundError:
#             logger.info("No existing generations found, initializing root")
#             gen = TrainingGen(
#                 created_at=datetime.now(),
#                 gen_n=0,
#                 n_mcts_iterations=n_mcts_iterations,
#                 c_exploration=c_exploration,
#                 c_ply_penalty=c_ply_penalty,
#                 self_play_batch_size=self_play_batch_size,
#                 training_batch_size=training_batch_size,
#             )
#             model = DominionatorModel(model_config)
#             gen.save_all(base_dir, None, model)
#             return gen

#     def get_games(self, base_dir: str) -> Optional[PlayGamesResult]:
#         gen_folder = self.gen_folder(base_dir)
#         with open(os.path.join(gen_folder, "games.pkl"), "rb") as f:
#             return pickle.load(f)

#     def get_model(self, base_dir: str) -> DominionatorModel:
#         """Gets the model for this generation."""
#         gen_folder = self.gen_folder(base_dir)
#         with open(os.path.join(gen_folder, "model.pkl"), "rb") as f:
#             model = pickle.load(f)
#             return model


def train_single_gen(
    base_dir: str,
    device: torch.device,
    base_config: ModelConfig,
    n_self_play_games: int,
    n_mcts_iterations: int,
    c_exploration: float,
    self_play_batch_size: int,
    training_batch_size: int,
) -> None:
    """
    Trains a new generation from the given parent.
    First generate games using dominionator._rust.play_games.
    Then train a new model based on the parent model using the generated samples.
    Finally, save the resulting games and model in the training directory.
    """
    # gen_n = parent.gen_n + 1
    # logger.info(f"Beginning new generation {gen_n} from {parent.gen_n}")

    # TODO: log experiment metadata in MLFlow

    # Self play
    model = DominionatorModel(base_config)
    model.to(device)
    reqs = [rust.GameMetadata(id, 0, 0) for id in range(n_self_play_games)]  # type: ignore
    games = rust.play_games(  # type: ignore
        reqs,
        self_play_batch_size,
        n_mcts_iterations,
        c_exploration,
        lambda batch: model.forward_numpy(batch),  # type: ignore
    )

    print(len(games))
    exit(0)

    # Training
    logger.info("Beginning training")
    model = copy.deepcopy(model)
    train, test = games.split_train_test(0.8, 1337)  # type: ignore
    data_module = SampleDataModule(train, test, training_batch_size)
    best_model_cb = BestModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[
            best_model_cb,
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
    )
    trainer.gen_n = gen_n  # type: ignore
    model.train()  # Switch batch normalization to train mode for training bn params
    trainer.fit(model, data_module)
    logger.info("Finished training")

    gen = TrainingGen(
        created_at=datetime.now(),
        gen_n=parent.gen_n + 1,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        parent=parent.created_at,
        val_loss=trainer.callback_metrics["val_loss"].item(),
        solver_score=solver_score,
    )
    gen.save_all(base_dir, games, best_model_cb.get_best_model())
    return gen
