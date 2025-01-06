use serde::{Deserialize, Serialize};

/// Macro to define a Card enum that encapsulates all the properties of a card.
macro_rules! define_cards {
    (
        $(
            $name:ident: {
                short_name: $short_name:literal,
                cost: $cost:literal,
                treasure: $treasure:literal,
                victory: $victory:literal,
            }
        ),* $(,)?
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub enum Card {
            $(
                $name,
            )*
        }

        impl Card {
            pub const fn name(&self) -> &'static str {
                match self {
                    $(
                        Self::$name => stringify!($name),
                    )*
                }
            }

            pub const fn short_name(&self) -> &'static str {
                match self {
                    $(
                        Self::$name => $short_name,
                    )*
                }
            }

            pub const fn cost(&self) -> u8 {
                match self {
                    $(
                        Self::$name => $cost,
                    )*
                }
            }

            pub const fn treasure(&self) -> u8 {
                match self {
                    $(
                        Self::$name => $treasure,
                    )*
                }
            }

            pub const fn victory(&self) -> u8 {
                match self {
                    $(
                        Self::$name => $victory,
                    )*
                }
            }

            pub const fn to_idx(&self) -> usize {
                *self as usize
            }

            pub const fn from_idx(idx: usize) -> Option<Self> {
                if idx < Self::N_CARDS {
                    // Safety: we just checked that idx is in bounds
                    Some(unsafe { std::mem::transmute(idx as u8) })
                } else {
                    None
                }
            }

            pub const N_CARDS: usize = [$(Self::$name),*].len();
            pub const ALL: [Self; Self::N_CARDS] = [$(Self::$name),*];
        }

        impl std::fmt::Display for Card {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.short_name())
            }
        }

    };
}

define_cards! {
    Copper: {
        short_name: "C",
        cost: 0,
        treasure: 1,
        victory: 0,
    },
    Silver: {
        short_name: "S",
        cost: 3,
        treasure: 2,
        victory: 0,
    },
    Gold: {
        short_name: "G",
        cost: 6,
        treasure: 3,
        victory: 0,
    },
    Estate: {
        short_name: "E",
        cost: 2,
        treasure: 0,
        victory: 1,
    },
    Duchy: {
        short_name: "D",
        cost: 5,
        treasure: 0,
        victory: 3,
    },
    Province: {
        short_name: "P",
        cost: 8,
        treasure: 0,
        victory: 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Card::*;

    #[test]
    fn test_card_properties() {
        assert_eq!(Copper.name(), "Copper");
        assert_eq!(Copper.short_name(), "C");
        assert_eq!(Copper.cost(), 0);
        assert_eq!(Copper.treasure(), 1);
        assert_eq!(Copper.victory(), 0);
    }

    #[test]
    fn test_card_indices() {
        // Test that indices are sequential
        for (i, card) in Card::ALL.iter().enumerate() {
            assert_eq!(card.to_idx(), i);
            assert_eq!(Card::from_idx(i), Some(*card));
        }

        // Test out of bounds
        assert_eq!(Card::from_idx(Card::N_CARDS), None);
    }

    #[test]
    fn test_card_ordering() {
        // Test that cards can be used in sorted collections
        use std::collections::BTreeSet;
        let mut set = BTreeSet::new();
        set.insert(Copper);
        set.insert(Silver);
        assert!(set.contains(&Copper));
    }
}
