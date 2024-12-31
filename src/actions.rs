use crate::cards::Card;
use std::fmt::Display;

macro_rules! define_actions {
    (
        simple_actions: [ $($simple:ident),* $(,)? ],
        card_actions: [ $($action:ident),* $(,)? ],
    ) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub enum Action {
            $($simple,)*
            $($action(Card),)*
        }

        impl Action {
            pub const fn name(&self) -> &'static str {
                match self {
                    $(Self::$simple => stringify!($simple),)*
                    $(Self::$action(_) => stringify!($action),)*
                }
            }

            pub const N_ACTIONS: usize = {
                let n_simple = [$($simple),*].len();
                let n_card_specific = [$($action),*].len() * Card::N_CARDS;
                n_simple + n_card_specific
            };

            pub const ALL: [Action; Self::N_ACTIONS] = {
                let mut actions = [Action::EndTurn; Self::N_ACTIONS];
                let mut idx = 0;

                // Add simple actions
                $(
                    actions[idx] = Action::$simple;
                    idx += 1;
                )*

                // Add card-specific actions
                let mut card_idx = 0;
                while card_idx < Card::N_CARDS {
                    $(
                        actions[idx] = Action::$action(Card::ALL[card_idx]);
                        idx += 1;
                    )*
                    card_idx += 1;
                }

                actions
            };

            pub const fn to_idx(&self) -> usize {
                match *self {
                    $(Self::$simple => 0,)*
                    Self::Play(card) => 1 + card.to_idx() * 3 + 0,
                    Self::Trash(card) => 1 + card.to_idx() * 3 + 1,
                    Self::Buy(card) => 1 + card.to_idx() * 3 + 2,
                }
            }

            pub const fn from_idx(idx: usize) -> Option<Self> {
                if idx < Self::N_ACTIONS {
                    Some(Self::ALL[idx])
                } else {
                    None
                }
            }
        }

        impl Display for Action {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$simple => write!(f, "{}", stringify!($simple)),)*
                    $(Self::$action(card) => write!(f, "{}({})", stringify!($action), card),)*
                }
            }
        }
    };
}

use Action::*;

define_actions! {
    simple_actions: [EndTurn],
    card_actions: [
        Play,
        Trash,
        Buy,
    ],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{Card::*, *};

    #[test]
    fn test_action_indices() {
        // Test that indices are sequential and reversible
        for (i, action) in Action::ALL.iter().enumerate() {
            assert_eq!(action.to_idx(), i);
            assert_eq!(Action::from_idx(i), Some(*action));
        }

        // Test out of bounds
        assert_eq!(Action::from_idx(Action::N_ACTIONS), None);
    }

    #[test]
    fn test_action_display() {
        assert_eq!(Action::EndTurn.to_string(), "EndTurn");
        assert_eq!(Action::Play(Silver).to_string(), "Play(S)");
        assert_eq!(Action::Buy(Copper).to_string(), "Buy(C)");
        assert_eq!(Action::Trash(Gold).to_string(), "Trash(G)");
    }

    #[test]
    fn test_card_specific_actions() {
        // Verify that we have the right number of actions
        let expected_actions = 1 + // EndTurn
            (3 * Card::N_CARDS); // Buy/Play/Trash for each card
        assert_eq!(Action::N_ACTIONS, expected_actions);

        // Verify that each card-specific action exists exactly once
        let mut count_buy = 0;
        let mut count_play = 0;
        let mut count_trash = 0;

        for action in Action::ALL {
            match action {
                Action::Buy(_) => count_buy += 1,
                Action::Play(_) => count_play += 1,
                Action::Trash(_) => count_trash += 1,
                Action::EndTurn => (),
            }
        }

        assert_eq!(count_buy, Card::N_CARDS);
        assert_eq!(count_play, Card::N_CARDS);
        assert_eq!(count_trash, Card::N_CARDS);
    }
}
