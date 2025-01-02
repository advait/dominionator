use crate::{policy::Policy, state::State};

/// The number of turns that have been played in the game.
pub type Ply = u8;

/// The lucrativeness value of a given position. This is the objective we are trying to maximize.
pub type QValue = f32;

/// ID of the Model's NN.
pub type ModelID = u64;

#[derive(Debug, Clone)]
/// The estimated value and policy of a given position.
pub struct NNEst {
    /// The estimated value of the position.
    pub q: QValue,

    /// The estimated policy of the position.
    pub policy_logprobs: Policy,

    /// The maximum ply reached in the MCTS algorithm.
    pub max_ply: u8,

    /// The average ply reached in the MCTS algorithm.
    pub avg_ply: u8,
}

/// Estimate the value and policy of a batch of states with an NN forward pass.
/// The ordering of the results corresponds to the ordering of the input states.
pub trait NNEstT {
    fn est_states(&self, model_id: ModelID, states: Vec<State>) -> Vec<NNEst>;
}

/// Metadata about a game.
#[derive(Debug, Clone, Default)]
pub struct GameMetadata {
    pub game_id: u64,
    pub player0_id: ModelID,
    pub player1_id: ModelID,
}

/// The finished result of a game.
#[derive(Debug, Clone)]
pub struct GameResult {
    pub metadata: GameMetadata,
    pub samples: Vec<Sample>,
}

/// A training sample generated via self-play.
#[derive(Debug, Clone)]
pub struct Sample {
    pub state: State,
    pub policy: Policy,
    pub q: QValue,
}
