use std::{array, fmt::Display};

use crate::{cards::Card, pile::Pile};

/// An action that can be played in the game of Dominion.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Action {
    /// Ends the current phase.
    EndPhase,
    /// Selects a card from the kingdom. The operation performed depends on the phase of the game.
    SelectCard(Card),
    /// Padding action (never playable) exist in case the number of kingdom cards is lower than 17
    Padding,
}

impl Action {
    pub const N_ACTIONS: usize = 1 + Pile::MAX_UNIQUE_CARDS;

    /// Returns the index of the action in the policy vector.
    pub fn to_idx(&self, kingdom: &Pile) -> usize {
        match self {
            Action::EndPhase => 0,
            Action::SelectCard(card) => kingdom.index_of(*card) + 1,
            Action::Padding => panic!("Padding action should not be indexed"),
        }
    }

    /// Returns the action corresponding to the given index.
    pub fn from_idx(idx: usize, kingdom: &Pile) -> Self {
        if idx == 0 {
            Action::EndPhase
        } else {
            Action::SelectCard(kingdom.card_at_index(idx - 1))
        }
    }

    /// Returns an array of all possible actions, ordered by the index of the action in the
    /// policy vector.
    pub fn all<'a>(kingdom: &'a Pile) -> [Self; Action::N_ACTIONS] {
        let n_kingdom_cards = kingdom.unique_card_count();
        array::from_fn(|i| {
            if i == 0 {
                Action::EndPhase
            } else if (i - 1) < n_kingdom_cards {
                Action::SelectCard(kingdom.card_at_index(i - 1))
            } else {
                Action::Padding
            }
        })
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::EndPhase => write!(f, "EndPhase"),
            Action::SelectCard(card) => write!(f, "SelectCard({})", card),
            Action::Padding => write!(f, "Padding"),
        }
    }
}
