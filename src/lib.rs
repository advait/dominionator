pub mod actions;
pub mod cards;
pub mod embeddings;
pub mod mcts;
pub mod policy;
pub mod self_play;
pub mod state;
pub mod types;
pub mod utils;

use embeddings::Embedding;
use pyo3::prelude::*;

#[pymodule(name = "_rust")]
fn dominionator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("N_EMBEDDINGS", Embedding::N_EMBEDDINGS)?;
    Ok(())
}
