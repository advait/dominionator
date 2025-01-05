pub mod actions;
pub mod cards;
pub mod embeddings;
pub mod mcts;
pub mod pile;
pub mod policy;
pub mod pybridge;
pub mod self_play;
pub mod state;
pub mod types;

use pyo3::prelude::*;

use embeddings::Embedding;
use types::{GameMetadata, GameResult, Sample};

#[pymodule(name = "_rust")]
fn dominionator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("N_EMBEDDINGS", Embedding::N_EMBEDDINGS)?;

    m.add_class::<GameResult>()?;
    m.add_class::<GameMetadata>()?;
    m.add_class::<Sample>()?;

    Ok(())
}
