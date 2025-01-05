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

use actions::Action;
use env_logger::Env;
use pybridge::PlayGamesResult;
use pyo3::prelude::*;

use embeddings::Embedding;
use types::{GameMetadata, GameResult, Sample};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, or Python will not be able to
/// import the module.
#[pymodule(name = "dominionator_rust")]
fn dominionator_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    m.add("N_ACTIONS", Action::N_ACTIONS)?;
    m.add("N_EMBEDDINGS", Embedding::N_EMBEDDINGS)?;

    m.add_class::<GameResult>()?;
    m.add_class::<GameMetadata>()?;
    m.add_class::<Sample>()?;
    m.add_class::<PlayGamesResult>()?;

    m.add_function(wrap_pyfunction!(pybridge::play_games, m)?)?;

    Ok(())
}
