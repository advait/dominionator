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
    /// -log(ply_est + 1) where ply_est is the estimated remaining number of turns until game
    /// completion.
    pub ply1_log_neg: f32,

    /// The estimated policy of the position.
    pub policy_logprobs: Policy,
}

impl NNEst {
    pub fn new_from_ply(ply: u8, policy_logprobs: Policy) -> Self {
        Self {
            ply1_log_neg: Self::ply1_log_neg_from_ply(ply),
            policy_logprobs,
        }
    }

    pub fn ply(&self) -> f32 {
        -(self.ply1_log_neg.exp() - 1.0)
    }

    pub fn ply1_log_neg_from_ply(ply: u8) -> f32 {
        -((ply + 1) as f32).ln()
    }
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
    pub policy_logprobs: Policy,
    pub ply1_log_neg: f32,
}

impl Sample {
    pub fn new_from_terminal_ply(state: State, policy_logprobs: Policy, terminal_ply: u8) -> Self {
        let ply1_log_neg = NNEst::ply1_log_neg_from_ply(terminal_ply - state.ply);
        Self {
            state,
            policy_logprobs,
            ply1_log_neg,
        }
    }
}
