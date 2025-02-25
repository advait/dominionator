from typing import List
import einops
import numpy as np
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_embeddings: int
    """Number of unique embeddings."""

    dim_embedding: int
    """Dimension of the embedding space. Must be a multiple of n_transformer_heads."""

    n_transformer_heads: int
    """Number of transformer heads."""

    n_transformer_layers: int
    """Number of transformer layers."""

    dim_policy: int
    """Dimension of the policy output."""

    lr: float
    """Learning rate."""

    l2_reg: float
    """L2 regularization."""

    embedding_padding_idx: int = 0
    """Padding index for the embedding bag."""

    dim_feedforward: int = 0
    """Dimension of the feedforward network."""

    def __post_init__(self):
        if self.dim_feedforward == 0:
            self.dim_feedforward = 4 * self.dim_embedding


@dataclass
class Forward:
    ply1_log_neg: torch.Tensor  # (batch, 1)
    policy_logprobs: torch.Tensor  # (batch, dim_policy)


@dataclass
class ForwardItem:
    ply1_log_neg: float  # (batch, 1)
    policy_logprobs: List[float]


class DominionatorModel(pl.LightningModule):
    EPS = 1e-8

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding_builder = EmbeddingBuilder(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_embedding,
            dim_feedforward=config.dim_feedforward,
            nhead=config.n_transformer_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_transformer_layers,
        )

        self.ply_head = nn.Sequential(
            nn.Linear(config.dim_embedding, 1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.dim_embedding, config.dim_policy),
            nn.LogSoftmax(dim=-1),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg
        )

    def training_step(self, batch, _batch_idx):
        return self.step(batch, log_prefix="train")

    def validation_step(self, batch, _batch_idx):
        return self.step(batch, log_prefix="val")

    def forward(self, x: torch.Tensor) -> Forward:
        # Transform the state into a sequence of embeddings
        x = self.embedding_builder(x)
        assert x.ndim == 3  # (batch_size, seq_len, embedding_dim)
        (batch_size, seq_len, embedding_dim) = x.shape
        assert embedding_dim == self.config.dim_embedding

        # Run the state token set through the transformer
        x = self.transformer(x)
        assert x.shape == (batch_size, seq_len, embedding_dim)

        # Use the cls token as the state representation
        x = x[:, 0, :]

        # Run the final state through the ply and policy heads
        ply1_log_neg = -self.ply_head(x)
        policy_logprobs = self.policy_head(x)

        return Forward(ply1_log_neg=ply1_log_neg, policy_logprobs=policy_logprobs)

    def forward_numpy(self, x: np.ndarray | List[List[float]]) -> List[ForwardItem]:
        """Forward pass for numpy input. Model is run in inference mode. Used for self play."""
        self.eval()
        tensor = torch.tensor(x).to(self.device)
        with torch.no_grad():
            forward = self.forward(tensor)

        ret = [
            ForwardItem(
                ply1_log_neg=ply1_log_neg.cpu().item(),
                policy_logprobs=policy_logprobs.cpu().tolist(),
            )
            for (ply1_log_neg, policy_logprobs) in zip(
                forward.ply1_log_neg, forward.policy_logprobs
            )
        ]
        return ret

    def step(self, batch: "Batch", log_prefix: str):
        forward = self.forward(batch.state_raw_embeddings)

        ply_loss = F.mse_loss(forward.ply1_log_neg, batch.ply1_log_neg_target)
        policy_loss = F.kl_div(
            forward.policy_logprobs,
            # Convert logprobs to probs so that -Inf converts to prob of 0 (to avoid NaNs)
            batch.policy_target_logprobs.exp(),
            log_target=False,
            reduction="batchmean",
        )
        total_loss = ply_loss + policy_loss

        self.log(f"{log_prefix}_ply_loss", ply_loss, prog_bar=log_prefix == "val")
        self.log(f"{log_prefix}_policy_loss", policy_loss, prog_bar=log_prefix == "val")
        self.log(f"{log_prefix}_loss", total_loss, prog_bar=log_prefix == "val")
        return total_loss


class EmbeddingBuilder(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=config.n_embeddings,
            embedding_dim=config.dim_embedding,
            padding_idx=config.embedding_padding_idx,
        )
        self.cls_token = nn.Parameter(torch.randn(config.dim_embedding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3  # (batch, seq, n_embeddings)
        (batch_size, seq_len, _n_embeddings) = x.shape
        x = einops.rearrange(x, "b s n -> (b s) n")
        bags = self.embedding_bag(x)
        bags = einops.rearrange(bags, "(b s) n -> b s n", b=batch_size, s=seq_len)

        # Prepend the cls_token to the seq
        cls_token = einops.repeat(self.cls_token, "n -> b 1 n", b=batch_size)
        bags = torch.cat([cls_token, bags], dim=1)  # (batch, seq + 1, dim_embedding)

        return bags


@dataclass
class Batch:
    state_raw_embeddings: torch.Tensor  # (batch, seq, max_embeddings_per_token)
    ply1_log_neg_target: torch.Tensor  # (batch, 1)
    policy_target_logprobs: torch.Tensor  # (batch, dim_policy)

    @property
    def batch_size(self) -> int:
        return self.state_raw_embeddings.shape[0]

    @property
    def seq_len(self) -> int:
        return self.state_raw_embeddings.shape[1]

    def __post_init__(self):
        assert self.state_raw_embeddings.ndim == 3
        assert self.ply1_log_neg_target.ndim == 2
        assert self.policy_target_logprobs.ndim == 2
        assert (
            self.state_raw_embeddings.shape[0]
            == self.ply1_log_neg_target.shape[0]
            == self.policy_target_logprobs.shape[0]
        ), "Batch size mismatch"
