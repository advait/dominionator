pub mod actions;
pub mod cards;
pub mod embeddings;
pub mod mcts;
pub mod policy;
pub mod self_play;
pub mod state;
pub mod types;
pub mod utils;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn dominionator_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    println!("Inside dominionator_rust module creation");
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
