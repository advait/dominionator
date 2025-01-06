use pyo3::{
    prelude::*,
    types::{PyBytes, PyList},
};
use serde::{Deserialize, Serialize};

use crate::{
    self_play::self_play,
    types::{GameMetadata, GameResult, NNEst, NNEstParams, NNEstT, Sample},
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

/// Play games via MCTS. This is a python wrapper around [self_play].
/// `reqs` is a list of [GameMetadata] that describes the games to play.
/// The `py_eval_pos_cb` callback is expected to be a pytorch model that runs on the GPU.
#[pyfunction]
pub fn play_games<'py>(
    py: Python<'py>,
    reqs: &Bound<'py, PyList>,
    max_nn_batch_size: usize,
    n_mcts_iterations: usize,
    c_exploration: f32,
    py_eval_pos_cb: &Bound<'py, PyAny>,
) -> PyResult<PlayGamesResult> {
    let reqs: Vec<GameMetadata> = reqs.extract().expect("error extracting reqs");

    let eval_pos = PyEvalPos {
        nn_forward_cb: py_eval_pos_cb.clone().unbind(),
    };

    let results = {
        // Start background processing threads while releasing the GIL with allow_threads.
        // This allows other python threads (e.g. pytorch) to continue while we generate training
        // samples. When we need to call the py_eval_pos callback, we will re-acquire the GIL.
        py.allow_threads(move || {
            self_play(
                eval_pos,
                reqs,
                max_nn_batch_size,
                n_mcts_iterations,
                c_exploration,
            )
        })
    };

    Ok(PlayGamesResult { results })
}

/// The result of [play_games].
/// Note we explicitly spcify pyclass(module="dominionator_rust") as the module name is required in
/// order for pickling to work.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "dominionator_rust")]
pub struct PlayGamesResult {
    #[pyo3(get)]
    pub results: Vec<GameResult>,
}

#[pymethods]
impl PlayGamesResult {
    /// Empty constructor is required for unpickling.
    #[new]
    fn new() -> Self {
        PlayGamesResult { results: vec![] }
    }

    fn to_cbor(&self, py: Python) -> PyResult<PyObject> {
        let cbor = serde_cbor::to_vec(&self).expect("Failed to serialize PlayGamesResult");
        Ok(PyBytes::new(py, &cbor).into())
    }

    #[staticmethod]
    fn from_cbor(_py: Python, cbor: &[u8]) -> PyResult<Self> {
        Ok(serde_cbor::from_slice(cbor).expect("Failed to deserialize PlayGamesResult"))
    }

    /// Used for pickling serialization.
    fn __getstate__(&self, _py: Python) -> PyResult<PyObject> {
        self.to_cbor(_py)
    }

    /// Used for pickling deserialization.
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let cbor: &[u8] = state.extract(py)?;
        *self = Self::from_cbor(py, cbor)?;
        Ok(())
    }

    /// Combine two PlayGamesResult objects.
    fn __add__<'py>(&mut self, py: Python<'py>, other: PyObject) -> PyResult<Self> {
        let other = other.extract::<PlayGamesResult>(py)?;
        Ok(PlayGamesResult {
            results: self
                .results
                .iter()
                .chain(other.results.iter())
                .cloned()
                .collect(),
        })
    }

    fn __len__(&self) -> usize {
        self.results.len()
    }

    /// Splits the results into training and test datasets.
    /// Ensures that whole games end up in either the training set or test set.
    /// Expects `train_frac` to be in [0, 1].
    fn split_train_test(
        &mut self,
        train_frac: f32,
        seed: u64,
    ) -> PyResult<(Vec<Sample>, Vec<Sample>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.results.shuffle(&mut rng);
        let n_train = (self.results.len() as f32 * train_frac).round() as usize;
        let (train, test) = self.results.split_at(n_train);
        Ok((
            train.into_iter().flat_map(|r| r.samples.clone()).collect(),
            test.into_iter().flat_map(|r| r.samples.clone()).collect(),
        ))
    }

    /// Returns the number of unique states in the results.
    fn unique_states(&self) -> usize {
        self.results
            .iter()
            .flat_map(|r| r.samples.iter())
            .map(|s| &s.state)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
}

/// [EvalPosT] implementation that calls the `py_eval_pos_cb` python callback.
struct PyEvalPos {
    nn_forward_cb: Py<PyAny>,
}

impl NNEstT for PyEvalPos {
    /// Evaluates a batch of positions by calling the [Self::nn_forward_cb] callback.
    /// This is intended to be a pytorch model that runs on the GPU. Because this is a python
    /// call we need to first re-acquire the GIL to call this function from a background thread
    /// before performing the callback.
    fn est_states(&self, params: NNEstParams) -> Vec<NNEst> {
        Python::with_gil(|py| {
            let token_batch = params
                .states
                .iter()
                .map(|s| s.to_tokens_indices())
                .collect::<Vec<_>>();

            let items: Vec<Py<PyAny>> = (&self
                .nn_forward_cb
                .call(py, (token_batch,), None)
                .expect("Failed to call nn_forward_cb"))
                .extract(py)
                .expect("Failed to extract result");

            items
                .iter()
                .map(|item| NNEst {
                    ply1_log_neg: item
                        .getattr(py, "ply1_log_neg")
                        .expect("Failed to get ply1_log_neg")
                        .extract(py)
                        .expect("Failed to extract ply1_log_neg"),
                    policy_logprobs: item
                        .getattr(py, "policy_logprobs")
                        .expect("Failed to get policy_logprobs")
                        .extract(py)
                        .expect("Failed to extract policy_logprobs"),
                })
                .collect()
        })
    }
}
