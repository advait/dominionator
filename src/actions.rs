use crate::cards::{Card, CARD_SET, N_CARDS};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Action {
    EndTurn,
    Buy(&'static Card),
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
