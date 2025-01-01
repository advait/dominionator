macro_rules! define_enum_with_variant_count {
    (
        $vis:vis enum $name:ident {
            $( $variant:ident ),* $(,)?
        }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
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

    (
        $vis:vis enum $name:ident {
            $( $variant:ident(f32) ),* $(,)?
        }
    ) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        $vis enum $name {
            $( $variant ( f32 ) ),*
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
                    Self::$variant(0.0),
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

define_enum_with_variant_count! {
    pub enum ContinuousCount {
        N(f32),
        LogN(f32),
        P(f32),
        LogP(f32),
    }
}

use crate::cards::Card;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Embedding {
    Card(Card),
    PileType(PileType),
    DiscreteCount(DiscreteCount),
    ContinuousCount(ContinuousCount),
}

impl Embedding {
    pub const N_EMBEDDINGS: usize = Card::N_CARDS
        + PileType::N_VARIANTS
        + DiscreteCount::N_VARIANTS
        + ContinuousCount::N_VARIANTS;

    pub const ALL: [Self; Self::N_EMBEDDINGS] = {
        let mut embeddings = [Self::Card(Card::ALL[0]); Self::N_EMBEDDINGS];
        let mut idx = 0;

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

        // Add ContinuousCount embeddings
        i = 0;
        while i < ContinuousCount::N_VARIANTS {
            embeddings[idx] = Self::ContinuousCount(ContinuousCount::ALL[i]);
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

// Verify that our const array has the correct length
static_assertions::const_assert_eq!(Embedding::N_EMBEDDINGS, Embedding::ALL.len());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_indices() {
        // Test that indices are sequential
        for (i, embedding) in Embedding::ALL.iter().enumerate() {
            assert_eq!(embedding.to_idx(), i);
            assert_eq!(Embedding::from_idx(i), Some(*embedding));
        }

        // Test out of bounds
        assert_eq!(Embedding::from_idx(Embedding::N_EMBEDDINGS), None);
    }

    #[test]
    fn test_variant_counts() {
        assert_eq!(PileType::N_VARIANTS, 4);
        assert_eq!(DiscreteCount::N_VARIANTS, 14);
        assert_eq!(ContinuousCount::N_VARIANTS, 4);
    }
}
