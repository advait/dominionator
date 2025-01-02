import pytest
import torch

from dominionator.nn import ModelConfig, DominionatorModel, Batch


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_dominionator_model():
    config = ModelConfig(
        n_embeddings=100,
        dim_embedding=64,
        n_transformer_heads=8,
        n_transformer_layers=2,
        dim_policy=10,
        lr=1e-4,
        l2_reg=1e-5,
    )
    model = DominionatorModel(config)

    # Create dummy batch
    batch_size = 4
    seq_len = 3
    batch = Batch(
        state_raw_embeddings=torch.randint(
            0, config.n_embeddings, (batch_size, seq_len, 10)
        ),
        q_target=torch.randn(batch_size, 1),
        policy_target_probs=torch.softmax(
            torch.randn(batch_size, config.dim_policy), dim=-1
        ),
    )

    # Test forward pass
    forward = model(batch.state_raw_embeddings)
    assert forward.q.shape == (batch_size, 1)
    assert forward.policy_logits.shape == (batch_size, config.dim_policy)

    # Test training step
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
