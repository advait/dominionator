use crate::{
    actions::{Action, ACTION_SET, N_ACTIONS},
    cards::{Card, DEFAULT_DECK, DEFAULT_KINGDOM},
    types::{Ply, Policy},
};
use rand::{rngs::SmallRng, seq::SliceRandom};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct State {
    hand: Vec<&'static Card>,
    draw: Vec<&'static Card>,
    discard: Vec<&'static Card>,
    kingdom: BTreeMap<&'static Card, u8>,
    unspent_gold: u8,
    unspent_buys: u8,
    pub ply: Ply,
    win_conditions: Vec<WinCondition>,
}

#[derive(Debug, Copy, Clone)]
pub enum WinCondition {
    Victory(u8),
}

impl State {
    /// The number of cards we draw at the start of each turn.
    const HAND_SIZE: usize = 5;

    /// The maximum number of plys. Games are terminal if they reach this number.
    pub const MAX_PLY: Ply = 100;

    pub fn new(
        discard: &[&'static Card],
        kingdom: &[(&'static Card, u8)],
        win_conditions: &[WinCondition],
        rng: &mut SmallRng,
    ) -> Self {
        let mut ret = Self {
            hand: Vec::default(),
            draw: Vec::default(),
            discard: discard.to_vec(),
            kingdom: BTreeMap::from_iter(kingdom.into_iter().cloned()),
            unspent_gold: 0,
            unspent_buys: 1,
            ply: 0,
            win_conditions: win_conditions.to_vec(),
        };
        ret.draw(Self::HAND_SIZE, rng);

        ret
    }

    pub fn new_default(rng: &mut SmallRng) -> Self {
        Self::new(
            &DEFAULT_DECK,
            &DEFAULT_KINGDOM,
            &[WinCondition::Victory(10)],
            rng,
        )
    }

    /// Increases the ply, discards the hand, draws a new hand, and resets gold/buys.
    fn new_turn(&mut self, rng: &mut SmallRng) -> &mut Self {
        self.ply += 1;

        // Discard hand
        self.discard.extend(self.hand.iter());
        self.hand.clear();

        // Draw new hand
        self.draw(Self::HAND_SIZE, rng);

        // Reset unspent gold and buys
        self.unspent_gold = 0;
        self.unspent_buys = 1;

        self
    }

    /// Reshuffles the discard pile into the draw pile.
    fn reshuffle_discard(&mut self, rng: &mut SmallRng) -> &mut Self {
        let mut cards = self.discard.drain(..).collect::<Vec<_>>();
        cards.extend(self.draw.drain(..));
        cards.shuffle(&mut *rng);
        self.draw = cards;
        self
    }

    /// Draws `n` cards from the draw pile into the hand.
    /// If we run out of cards in the draw pile, we reshuffle the discard pile.
    /// If we run out of cards in the discard pile, we return having drawn fewer than n.
    fn draw(&mut self, n: usize, rng: &mut SmallRng) -> &mut Self {
        for _ in 0..n {
            if self.draw.is_empty() {
                if self.discard.is_empty() {
                    return self;
                }
                self.reshuffle_discard(rng);
            }

            let card = self.draw.pop().unwrap();
            self.unspent_gold += card.treasure;
            self.hand.push(card);
        }

        self
    }

    /// For each possible action, returns a tuple of the action and whether it is valid or not.
    pub fn valid_actions(&self) -> [(Action, bool); N_ACTIONS] {
        ACTION_SET.map(|action| match action {
            Action::EndTurn => (action, true),
            Action::Buy(card) => (
                action,
                self.unspent_gold >= card.cost && self.unspent_buys > 0,
            ),
        })
    }

    /// Mask the policy logprobs by setting illegal moves to [f32::NEG_INFINITY].
    pub fn mask_policy(&self, policy_logprobs: &mut Policy) {
        let valid_actions = self.valid_actions();

        for i in 0..N_ACTIONS {
            let (_, can_play) = valid_actions[i];
            if !can_play {
                policy_logprobs[i] = f32::NEG_INFINITY;
            }
        }
    }

    /// Returns the ply at which the game is terminal, or None if it is not terminal.
    pub fn is_terminal(&self) -> Option<Ply> {
        if self.ply >= Self::MAX_PLY {
            return Some(self.ply);
        }

        for &win_condition in self.win_conditions.iter() {
            match win_condition {
                WinCondition::Victory(target_victory) => {
                    if self.ply >= target_victory {
                        return Some(self.ply);
                    }
                }
            }
        }
        None
    }

    /// Applies an action to the state returning the resulting state.
    pub fn apply_action(&self, action: Action, rng: &mut SmallRng) -> Self {
        let mut next = self.clone();
        match action {
            Action::EndTurn => {
                next.new_turn(rng);
            }
            Action::Buy(card) => {
                next.kingdom.entry(card).and_modify(|count| *count -= 1);
                next.discard.push(card);
                next.unspent_gold -= card.cost;
                next.unspent_buys -= 1;
            }
        }
        next
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{COPPER, ESTATE};
    use proptest::prelude::*;
    use rand::SeedableRng;

    fn assert_can_play_action(state: &State, action: Action, can_play: bool) {
        assert_eq!(can_play_action(state, action), can_play);
    }

    fn can_play_action(state: &State, action: Action) -> bool {
        let actions = state.valid_actions();
        actions.contains(&(action, true))
    }

    proptest! {
        #[test]
        fn test_sanity(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let state = State::new_default(&mut rng);
            assert_eq!(state.ply, 0);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 5);
            assert_eq!(state.discard.len(), 0);
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply_action(Action::EndTurn, &mut rng);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 5);
        }

        #[test]
        fn test_buy_copper(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let state = State::new_default(&mut rng);
            let unspent_gold = state.unspent_gold;

            // Buy a copper
            assert_can_play_action(&state, Action::Buy(&COPPER), true);
            let state = state.apply_action(Action::Buy(&COPPER), &mut rng);
            assert_eq!(state.unspent_gold, unspent_gold - COPPER.cost);
            assert!(state.discard.contains(&&COPPER));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(&COPPER), false);
            assert_can_play_action(&state, Action::Buy(&ESTATE), false);

            // End turn
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply_action(Action::EndTurn, &mut rng);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 6); // Includes new copper
            assert_eq!(state.kingdom.len(), 2);
            assert_eq!(state.kingdom[&COPPER], 59);
            assert_eq!(state.kingdom[&ESTATE], 12);
        }

        #[test]
        fn test_buy_estate(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let state = State::new_default(&mut rng);
            let unspent_gold = state.unspent_gold;

            // Buy an estate
            assert_can_play_action(&state, Action::Buy(&ESTATE), true);
            let state = state.apply_action(Action::Buy(&ESTATE), &mut rng);
            assert_eq!(state.unspent_gold, unspent_gold - ESTATE.cost);
            assert!(state.discard.contains(&&ESTATE));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(&COPPER), false);
            assert_can_play_action(&state, Action::Buy(&ESTATE), false);

            // End turn
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply_action(Action::EndTurn, &mut rng);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 6); // Includes new estate
            assert_eq!(state.kingdom.len(), 2);
            assert_eq!(state.kingdom[&COPPER], 60);
            assert_eq!(state.kingdom[&ESTATE], 11);
        }

        #[test]
        fn test_victory(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = State::new_default(&mut rng);
            assert_eq!(state.is_terminal(), None);

            for _ in 0..100 {
                if state.is_terminal().is_some() {
                    return Ok(());
                }

                if can_play_action(&state, Action::Buy(&ESTATE)) {
                    state = state.apply_action(Action::Buy(&ESTATE), &mut rng);
                } else {
                    assert_can_play_action(&state, Action::Buy(&COPPER), true);
                    state = state.apply_action(Action::Buy(&COPPER), &mut rng);
                }
                state = state.apply_action(Action::EndTurn, &mut rng);
            }

            prop_assert!(false, "Failed to reach victory in {} turns", state.ply);
        }
    }
}
