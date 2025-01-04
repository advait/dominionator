import pytest
import torch

from dominionator.nn import ModelConfig, DominionatorModel, Batch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


def new_dummy_batch(batch_size: int = 4, seq_len: int = 3) -> Batch:
    """Creates a dummy batch for testing."""
    state_raw_embeddings = torch.randint(
        0, config.n_embeddings, (batch_size, seq_len, 10)
    )
    return Batch(
        state_raw_embeddings=state_raw_embeddings,
        ply1_log_neg_target=torch.rand(batch_size, 1) * 100,
        policy_target_probs=torch.softmax(
            torch.randn(batch_size, config.dim_policy), dim=-1
        ),
    )


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_sanity():
    batch = new_dummy_batch()

    # Test forward pass
    forward = model.forward(batch.state_raw_embeddings)
    assert forward.ply1_log_neg.shape == (batch.batch_size, 1)
    assert forward.policy_logits.shape == (batch.batch_size, config.dim_policy)

    # Test training step
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 1e-5


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_zero_loss_with_own_outputs():
    batch = new_dummy_batch()

    # Get model outputs
    model.eval()
    with torch.no_grad():
        forward = model.forward(batch.state_raw_embeddings)
        policy_probs = torch.softmax(forward.policy_logits, dim=-1)

    # Create batch using model's own outputs as targets
    batch = Batch(
        state_raw_embeddings=batch.state_raw_embeddings,
        ply1_log_neg_target=forward.ply1_log_neg,
        policy_target_probs=policy_probs.detach().clone(),
    )

    loss = model.training_step(batch, 0).item()
    assert loss < 1e-5
