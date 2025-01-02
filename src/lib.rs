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

#[pyfunction]
fn double(x: usize) -> usize {
    x * 2
}

#[pyfunction]
fn panic() {
    panic!("This is a panic message");
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "_rust")]
fn dominionator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(panic, m)?)?;
    Ok(())
}
