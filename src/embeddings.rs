use crate::cards::Card;

macro_rules! define_enum_with_variant_count {
    (
        $vis:vis enum $name:ident {
            $( $variant:ident ),* $(,)?
        }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $variant ),*
        }

        impl $name {
            pub const N_VARIANTS: usize = {
                let mut count = 0;
                $(
                    let _ = stringify!($variant);
                    count += 1;
                )*
                count
            };

            pub const ALL: [Self; Self::N_VARIANTS] = [
                $(
                    Self::$variant,
                )*
            ];
        }
    };
}

define_enum_with_variant_count! {
    pub enum PileType {
        Hand,
        Draw,
        Discard,
        Kingdom,
    }
}

define_enum_with_variant_count! {
    pub enum DiscreteCount {
        Zero,
        One,
        Two,
        Three,
        Four,
        Five,
        Six,
        Seven,
        Eight,
        Nine,
        Ten,
        Eleven,
        Twelve,
        ThirteenPlus,
    }
}

impl DiscreteCount {
    pub fn from_count(count: u8) -> Self {
        match count {
            0 => Self::Zero,
            1 => Self::One,
            2 => Self::Two,
            3 => Self::Three,
            4 => Self::Four,
            5 => Self::Five,
            6 => Self::Six,
            7 => Self::Seven,
            8 => Self::Eight,
            9 => Self::Nine,
            10 => Self::Ten,
            11 => Self::Eleven,
            12 => Self::Twelve,
            _ => Self::ThirteenPlus,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Embedding {
    Padding,
    Card(Card),
    PileType(PileType),
    DiscreteCount(DiscreteCount),
}

/// The number of embeddings per token (card, pile type, discrete count)
pub const N_EMBEDDINGS_PER_TOKEN: usize = 7;

pub type Token = [Embedding; N_EMBEDDINGS_PER_TOKEN];

pub trait TokenExt: Sized {
    fn get_pile_type(&self) -> Option<PileType>;
    fn get_card(&self) -> Option<Card>;
    fn get_discrete_count(&self) -> Option<DiscreteCount>;
    fn to_token_indices(tokens: &[Self]) -> Vec<[usize; N_EMBEDDINGS_PER_TOKEN]>;
    fn from_token_indices(indices: &[[usize; N_EMBEDDINGS_PER_TOKEN]]) -> Vec<Self>;
}

impl TokenExt for Token {
    fn get_pile_type(&self) -> Option<PileType> {
        self.iter().find_map(|&e| match e {
            Embedding::PileType(pt) => Some(pt),
            _ => None,
        })
    }

    fn get_card(&self) -> Option<Card> {
        self.iter().find_map(|&e| match e {
            Embedding::Card(c) => Some(c),
            _ => None,
        })
    }

    fn get_discrete_count(&self) -> Option<DiscreteCount> {
        self.iter().find_map(|&e| match e {
            Embedding::DiscreteCount(c) => Some(c),
            _ => None,
        })
    }

    fn to_token_indices(tokens: &[Self]) -> Vec<[usize; N_EMBEDDINGS_PER_TOKEN]> {
        tokens
            .iter()
            .map(|token| token.map(|embedding| embedding.to_idx()))
            .collect()
    }

    fn from_token_indices(indices: &[[usize; N_EMBEDDINGS_PER_TOKEN]]) -> Vec<Token> {
        indices
            .iter()
            .map(|indices| indices.map(|i| Embedding::from_idx(i).unwrap()))
            .collect()
    }
}

impl FromIterator<Embedding> for Token {
    fn from_iter<I: IntoIterator<Item = Embedding>>(iter: I) -> Self {
        let mut embeddings = [Embedding::Padding; N_EMBEDDINGS_PER_TOKEN];
        for (i, embedding) in iter.into_iter().enumerate() {
            if i >= N_EMBEDDINGS_PER_TOKEN {
                panic!("Too many embeddings in token");
            }
            embeddings[i] = embedding;
        }
        embeddings
    }
}

impl Embedding {
    pub const N_EMBEDDINGS: usize = 1 // Padding
        + Card::N_CARDS
        + PileType::N_VARIANTS
        + DiscreteCount::N_VARIANTS;

    pub const ALL: [Self; Self::N_EMBEDDINGS] = {
        let mut embeddings = [Self::Padding; Self::N_EMBEDDINGS];

        // idx=0 is reserved for padding
        embeddings[0] = Self::Padding;
        let mut idx = 1;

        // Add Card embeddings
        let mut i = 0;
        while i < Card::N_CARDS {
            embeddings[idx] = Self::Card(Card::ALL[i]);
            idx += 1;
            i += 1;
        }

        // Add PileType embeddings
        i = 0;
        while i < PileType::N_VARIANTS {
            embeddings[idx] = Self::PileType(PileType::ALL[i]);
            idx += 1;
            i += 1;
        }

        // Add DiscreteCount embeddings
        i = 0;
        while i < DiscreteCount::N_VARIANTS {
            embeddings[idx] = Self::DiscreteCount(DiscreteCount::ALL[i]);
            idx += 1;
            i += 1;
        }

        assert!(
            idx == Self::N_EMBEDDINGS,
            "Not all embeddings were initialized"
        );

        embeddings
    };

    pub fn to_idx(&self) -> usize {
        Self::ALL
            .iter()
            .position(|e| e == self)
            .expect("Embedding not found in ALL")
    }

    pub const fn from_idx(idx: usize) -> Option<Self> {
        if idx >= Self::N_EMBEDDINGS {
            return None;
        }
        Some(Self::ALL[idx])
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::cards::Card::*;
    use crate::state::StateBuilder;

    use super::*;

    fn get_tokens_for_pile_type(embeddings: &[Token], pile_type: PileType) -> Vec<&Token> {
        embeddings
            .iter()
            .filter(|token| token.get_pile_type() == Some(pile_type))
            .collect()
    }

    fn assert_has_token_for_card(tokens: &[&Token], card: Card, expected_count: u8) {
        let token = tokens
            .iter()
            .find(|t| t.get_card() == Some(card))
            .unwrap_or_else(|| panic!("No token found for card {:?}", card));
        assert_eq!(
            token.get_discrete_count(),
            Some(DiscreteCount::from_count(expected_count)),
            "Wrong count for card {:?}",
            card
        );
    }

    #[test]
    fn test_embedding_indices() {
        // Test that indices are sequential
        for (i, embedding) in Embedding::ALL.iter().enumerate() {
            assert_eq!(
                embedding.to_idx(),
                i,
                "Embedding index mismatch for {:?}",
                embedding
            );
            assert_eq!(
                Embedding::from_idx(i),
                Some(*embedding),
                "Embedding index mismatch for {:?}",
                embedding
            );
        }

        // Test out of bounds
        assert_eq!(Embedding::from_idx(Embedding::N_EMBEDDINGS), None);
    }

    #[test]
    fn test_variant_counts() {
        assert_eq!(PileType::N_VARIANTS, 4);
        assert_eq!(DiscreteCount::N_VARIANTS, 14);
    }

    #[test]
    fn test_empty_state_embeddings() {
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_hand(&[])
            .with_draw(&[])
            .with_kingdom(&[(Copper, 0), (Estate, 0)]) // Empty but defined kingdom
            .build(&mut rng);

        let embeddings = Token::from_token_indices(&state.to_tokens_indices());

        // Each pile type should have tokens for all kingdom cards
        for pile_type in PileType::ALL {
            let pile_tokens = get_tokens_for_pile_type(&embeddings, pile_type);
            assert_eq!(
                pile_tokens.len(),
                state.kingdom.unique_card_count(),
                "{:?} should have tokens for all kingdom cards",
                pile_type
            );

            for &card in state.kingdom.iter_cards() {
                assert_has_token_for_card(&pile_tokens, card, 0);
            }
        }
    }

    #[test]
    fn test_simple_hand_embeddings() {
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_hand(&[(Copper, 2), (Estate, 1)])
            .with_draw(&[])
            .with_kingdom(&[(Copper, 1), (Estate, 1), (Silver, 0)]) // Include Silver with count 0
            .build(&mut rng);

        let embeddings = Token::from_token_indices(&state.to_tokens_indices());
        let hand_tokens = get_tokens_for_pile_type(&embeddings, PileType::Hand);

        // Should have tokens for all kingdom cards
        assert_eq!(
            hand_tokens.len(),
            state.kingdom.unique_card_count(),
            "Should have one token per kingdom card"
        );

        // Verify counts for cards in hand
        for &card in state.kingdom.iter_cards() {
            let count = state.hand[card];
            assert_has_token_for_card(&hand_tokens, card, count);
        }
    }

    #[test]
    fn test_kingdom_embeddings() {
        let mut rng = SmallRng::seed_from_u64(1);
        let kingdom_cards = [(Copper, 10), (Silver, 5), (Gold, 0)]; // Include Gold with count 0
        let state = StateBuilder::new()
            .with_hand(&[])
            .with_draw(&[])
            .with_kingdom(&kingdom_cards)
            .build(&mut rng);

        let embeddings = Token::from_token_indices(&state.to_tokens_indices());
        let kingdom_tokens = get_tokens_for_pile_type(&embeddings, PileType::Kingdom);

        // Should have one token per kingdom card
        assert_eq!(
            kingdom_tokens.len(),
            state.kingdom.unique_card_count(),
            "Should have one token per kingdom card"
        );

        // Verify counts for all cards in kingdom
        for (card, count) in kingdom_cards {
            assert_has_token_for_card(&kingdom_tokens, card, count);
        }
    }

    #[test]
    fn test_all_pile_types_present() {
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_hand(&[])
            .with_draw(&[])
            .with_kingdom(&[(Silver, 5), (Gold, 3), (Copper, 0)]) // Include Copper with count 0
            .build(&mut rng);

        let embeddings = Token::from_token_indices(&state.to_tokens_indices());

        // Each pile type should have tokens for all kingdom cards
        for pile_type in PileType::ALL {
            let pile_tokens = get_tokens_for_pile_type(&embeddings, pile_type);
            assert_eq!(
                pile_tokens.len(),
                state.kingdom.unique_card_count(),
                "{:?} should have tokens for all kingdom cards",
                pile_type
            );
        }

        // Total number of tokens should be number of pile types * number of kingdom cards
        assert_eq!(
            embeddings.len(),
            PileType::N_VARIANTS * state.kingdom.unique_card_count(),
            "Total number of tokens should be pile_types * kingdom_cards"
        );
    }
}
