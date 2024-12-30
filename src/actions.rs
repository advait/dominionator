use std::fmt::Display;

use crate::cards::{Card, CARD_SET, N_CARDS};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Action {
    EndTurn,
    Buy(&'static Card),
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::EndTurn => write!(f, "EndTurn"),
            Action::Buy(card) => write!(f, "Buy({})", card.short_name),
        }
    }
}

pub const N_ACTIONS: usize = 1 + N_CARDS;

// The ordering of ACTION_SET must be globally consistent across all games as we use the index of
// actions within this array to indicate whether a given action is valid.
pub static ACTION_SET: [Action; N_ACTIONS] = {
    let mut actions = [Action::EndTurn; N_ACTIONS];
    let mut i = 0;
    while i < N_CARDS {
        actions[1 + i] = Action::Buy(CARD_SET[i]);
        i += 1;
    }
    actions
};

/// Returns the index of the given action in the ACTION_SET array.
pub fn action_to_idx(action: &Action) -> usize {
    ACTION_SET
        .iter()
        .position(|a| a == action)
        .expect("Action not found in ACTION_SET")
}
