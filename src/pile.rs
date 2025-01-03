use std::{
    collections::{btree_map::Iter, BTreeMap},
    fmt::{Display, Formatter},
    ops::Index,
};

use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::SmallRng};

use crate::{
    cards::Card,
    embeddings::{ContinuousCount, DiscreteCount, Embedding, PileType, Token},
};

/// A pile is a set of cards with a count.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Pile {
    counts: BTreeMap<Card, u8>,
    preserve_zero: bool,
}

impl Pile {
    pub fn new(preserve_zero: bool) -> Self {
        Self {
            counts: BTreeMap::new(),
            preserve_zero,
        }
    }

    /// Setting the preserve_zero flag preserves entries with a count of 0.
    /// This is useful for kingdom piles, where we want to keep track of empty slots.
    /// If preserve_zero is false, then all entries with a count of 0 are removed.
    pub fn with_preserve_zero(mut self, preserve_zero: bool) -> Self {
        self.preserve_zero = preserve_zero;
        self
    }

    /// Draws a card from the pile, returning the card. Uses the rng to sample from the pile
    /// according to the counts of the cards.
    /// Panics if the pile is empty.
    pub fn draw(&mut self, rng: &mut SmallRng) -> Card {
        let dist = WeightedIndex::new(self.counts.values()).expect("Pile is empty");
        let index = dist.sample(rng);
        let card = *self.counts.keys().nth(index).unwrap();
        self.take(card);
        card
    }

    /// Returns an iterator over the cards in the pile.
    pub fn keys(&self) -> impl Iterator<Item = &Card> {
        self.counts.keys()
    }

    /// Returns true if the pile contains at least one of the given card.
    pub fn contains(&self, card: Card) -> bool {
        self[card] > 0
    }

    /// Returns the total non-unique count of cards in the pile.
    pub fn len(&self) -> u8 {
        self.counts.values().sum()
    }

    /// Returns true if the pile contains no cards.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Takes a card from the pile, returning the remining count of the card in the pile.
    pub fn take(&mut self, card: Card) -> u8 {
        match self.counts.get(&card) {
            None => panic!("Cannot take card from empty pile"),
            Some(1) => {
                if self.preserve_zero {
                    self.counts.insert(card, 0);
                } else {
                    self.counts.remove(&card);
                }
                0
            }
            Some(&count) => {
                self.counts.insert(card, count - 1);
                count - 1
            }
        }
    }

    /// Adds a card to the pile, returning the new count of the card in the pile.
    pub fn push(&mut self, card: Card) -> u8 {
        let count = self.counts.entry(card).or_insert(0);
        *count += 1;
        *count
    }

    /// Drains the cards from `consumed` into `self`.
    pub fn drain_from(&mut self, consumed: &mut Self) {
        for (card, count) in consumed.iter() {
            for _ in 0..count {
                self.push(card);
            }
        }
        if self.preserve_zero {
            consumed.counts = consumed
                .iter()
                .map(|(card, _)| (card, 0))
                .collect::<BTreeMap<_, _>>();
        } else {
            consumed.counts.clear();
        }
    }

    pub fn iter(&self) -> PileIter {
        PileIter {
            iter: self.counts.iter(),
        }
    }

    /// Returns the number of unique types of cards in the pile.
    /// If preserve_zero is true then this will include cards with a count of 0.
    pub fn unique_card_count(&self) -> usize {
        self.counts.len()
    }

    /// Returns the total victory points in the pile.
    pub fn victory_points(&self) -> u8 {
        self.iter()
            .map(|(card, count)| card.victory() * count)
            .sum()
    }

    /// Converts the pile into a list of tokens that summarize the pile.
    pub fn to_tokens(&self, pile_type: PileType) -> Vec<Token> {
        let total = self.len();
        self.iter()
            .map(|(card, count)| {
                [
                    Embedding::PileType(pile_type),
                    Embedding::Card(card),
                    Embedding::DiscreteCount(DiscreteCount::from_count(count)),
                ]
                .into_iter()
                .chain(ContinuousCount::from_counts(count, total))
                .collect::<Token>()
            })
            .collect()
    }
}

impl Index<Card> for Pile {
    type Output = u8;

    fn index(&self, index: Card) -> &Self::Output {
        self.counts.get(&index).unwrap_or(&0)
    }
}

impl Display for Pile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.counts
                .iter()
                .map(|(card, count)| format!("{}({})", card.short_name(), count))
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}

impl FromIterator<(Card, u8)> for Pile {
    fn from_iter<T: IntoIterator<Item = (Card, u8)>>(iter: T) -> Self {
        Self {
            counts: BTreeMap::from_iter(iter),
            preserve_zero: false,
        }
    }
}

impl FromIterator<Card> for Pile {
    fn from_iter<T: IntoIterator<Item = Card>>(iter: T) -> Self {
        Self {
            counts: BTreeMap::from_iter(iter.into_iter().map(|card| (card, 1))),
            preserve_zero: false,
        }
    }
}

impl IntoIterator for Pile {
    type Item = (Card, u8);
    type IntoIter = <BTreeMap<Card, u8> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.counts.into_iter()
    }
}

pub struct PileIter<'a> {
    iter: Iter<'a, Card, u8>,
}

impl<'a> Iterator for PileIter<'a> {
    type Item = (Card, u8);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&card, &count)| (card, count))
    }
}
