use std::{
    collections::{btree_map::Iter, BTreeMap},
    fmt::{Display, Formatter},
    ops::Index,
};

use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    embeddings::{ContinuousCount, DiscreteCount, Embedding, PileType, Token},
};

/// A pile is a set of cards with a count.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pile {
    counts: BTreeMap<Card, u8>,
}

impl Pile {
    /// Creates a new Pile with all the cards from the kingdom but with a count of 0.
    pub fn from_kingdom(kingdom: &Pile) -> Self {
        kingdom.iter().map(|(card, _count)| (card, 0)).collect()
    }

    /// Adds the given cards to the pile with the given counts.
    pub fn with_counts(mut self, counts: &[(Card, u8)]) -> Self {
        for &(card, count) in counts {
            for _ in 0..count {
                self.push(card);
            }
        }
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
            None => panic!("Card not present in pile"),
            Some(0) => panic!("Cannot take card from empty pile"),
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
        consumed.counts = consumed
            .iter()
            .map(|(card, _)| (card, 0))
            .collect::<BTreeMap<_, _>>();
    }

    /// Iterates over the cards in the pile including those with a count of 0.
    pub fn iter(&self) -> PileIter {
        PileIter {
            iter: self.counts.iter(),
        }
    }

    /// Returns an iterator over the cards in the pile including those with a count of 0.
    pub fn iter_cards(&self) -> impl Iterator<Item = &Card> {
        self.counts.keys()
    }

    /// Returns the number of unique types of cards in the pile including those with a count of 0.
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
        }
    }
}

impl FromIterator<Card> for Pile {
    fn from_iter<T: IntoIterator<Item = Card>>(iter: T) -> Self {
        Self {
            counts: BTreeMap::from_iter(iter.into_iter().map(|card| (card, 1))),
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
