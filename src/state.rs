use crate::cards::{Card, CARD_SET, DEFAULT_DECK, DEFAULT_KINGDOM, N_CARDS};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Action {
    EndTurn,
    Buy(&'static Card),
}

// The ordering of ACTION_SET must be globally consistent across all games as we use the index of
// actions within this array to indicate whether a given action is valid.
static ACTION_SET: [Action; 1 + N_CARDS] = {
    let mut actions = [Action::EndTurn; 1 + N_CARDS];
    let mut i = 0;
    while i < N_CARDS {
        actions[1 + i] = Action::Buy(CARD_SET[i]);
        i += 1;
    }
    actions
};

type Ply = u8;

#[derive(Debug, Clone)]
pub struct State {
    hand: VecDeque<&'static Card>,
    draw: VecDeque<&'static Card>,
    discard: VecDeque<&'static Card>,
    kingdom: BTreeMap<&'static Card, u8>,
    unspent_gold: u8,
    unspent_buys: u8,
    ply: Ply,
    win_conditions: Vec<WinCondition>,
    rng: SmallRng,
}

#[derive(Debug, Copy, Clone)]
pub enum WinCondition {
    Victory(u8),
}

impl State {
    const HAND_SIZE: usize = 5;

    pub fn new(
        seed: u64,
        discard: &[&'static Card],
        kingdom: &[(&'static Card, u8)],
        win_conditions: &[WinCondition],
    ) -> Self {
        let mut ret = Self {
            hand: VecDeque::default(),
            draw: VecDeque::default(),
            discard: VecDeque::from_iter(discard.into_iter().cloned()),
            kingdom: BTreeMap::from_iter(kingdom.into_iter().cloned()),
            unspent_gold: 0,
            unspent_buys: 1,
            ply: 0,
            win_conditions: win_conditions.to_vec(),
            rng: SmallRng::seed_from_u64(seed),
        };
        ret.draw(Self::HAND_SIZE);

        ret
    }

    pub fn new_default(seed: u64) -> Self {
        Self::new(
            seed,
            &DEFAULT_DECK,
            &DEFAULT_KINGDOM,
            &[WinCondition::Victory(10)],
        )
    }

    fn new_turn(&mut self) -> &mut Self {
        self.ply += 1;

        // Discard hand
        self.discard.extend(self.hand.iter());
        self.hand.clear();

        // Draw new hand
        self.draw(Self::HAND_SIZE);

        // Reset unspent gold and buys
        self.unspent_gold = 0;
        self.unspent_buys = 1;

        self
    }

    fn reshuffle_discard(&mut self) -> &mut Self {
        let mut cards = self.discard.drain(..).collect::<Vec<_>>();
        cards.extend(self.draw.drain(..));
        cards.shuffle(&mut self.rng);
        self.draw = VecDeque::from(cards);
        self
    }

    fn draw(&mut self, n: usize) -> &mut Self {
        for _ in 0..n {
            if self.draw.is_empty() {
                if self.discard.is_empty() {
                    return self;
                }
                self.reshuffle_discard();
            }

            let card = self.draw.pop_front().unwrap();
            self.unspent_gold += card.treasure;
            self.hand.push_back(card);
        }

        self
    }

    pub fn valid_actions(&self) -> Vec<(Action, bool)> {
        ACTION_SET
            .iter()
            .map(|&action| match action {
                Action::EndTurn => (action, true),
                Action::Buy(card) => (
                    action,
                    self.unspent_gold >= card.cost && self.unspent_buys > 0,
                ),
            })
            .collect()
    }

    pub fn is_terminal(&self) -> Option<Ply> {
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

    pub fn apply(&self, action: Action) -> Self {
        let mut next = self.clone();
        match action {
            Action::EndTurn => {
                next.new_turn();
            }
            Action::Buy(card) => {
                next.kingdom.entry(card).and_modify(|count| *count -= 1);
                next.discard.push_back(card);
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

    const N_TEST_SEEDS: u64 = 100;

    fn assert_can_play_action(state: &State, action: Action, can_play: bool) {
        assert_eq!(can_play_action(state, action), can_play);
    }

    fn can_play_action(state: &State, action: Action) -> bool {
        let actions = state.valid_actions();
        actions.contains(&(action, true))
    }

    #[test]
    fn test_sanity() {
        let state = State::new_default(0);
        assert_eq!(state.ply, 0);
        assert_eq!(state.hand.len(), 5);
        assert_eq!(state.draw.len(), 5);
        assert_eq!(state.discard.len(), 0);
        assert_can_play_action(&state, Action::EndTurn, true);
        let state = state.apply(Action::EndTurn);
        assert_eq!(state.ply, 1);
        assert_eq!(state.hand.len(), 5);
        assert_eq!(state.draw.len(), 0);
        assert_eq!(state.discard.len(), 5);
    }

    #[test]
    fn test_buy_copper() {
        for seed in 0..N_TEST_SEEDS {
            let state = State::new_default(seed);
            let unspent_gold = state.unspent_gold;

            // Buy a copper
            assert_can_play_action(&state, Action::Buy(&COPPER), true);
            let state = state.apply(Action::Buy(&COPPER));
            assert_eq!(state.unspent_gold, unspent_gold - COPPER.cost);
            assert!(state.discard.contains(&&COPPER));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(&COPPER), false);
            assert_can_play_action(&state, Action::Buy(&ESTATE), false);

            // End turn
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply(Action::EndTurn);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 6); // Includes new copper
            assert_eq!(state.kingdom.len(), 2);
            assert_eq!(state.kingdom[&COPPER], 59);
            assert_eq!(state.kingdom[&ESTATE], 12);
        }
    }

    #[test]
    fn test_buy_estate() {
        for seed in 0..N_TEST_SEEDS {
            let state = State::new_default(seed);
            let unspent_gold = state.unspent_gold;

            // Buy an estate
            assert_can_play_action(&state, Action::Buy(&ESTATE), true);
            let state = state.apply(Action::Buy(&ESTATE));
            assert_eq!(state.unspent_gold, unspent_gold - ESTATE.cost);
            assert!(state.discard.contains(&&ESTATE));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(&COPPER), false);
            assert_can_play_action(&state, Action::Buy(&ESTATE), false);

            // End turn
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply(Action::EndTurn);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 6); // Includes new estate
            assert_eq!(state.kingdom.len(), 2);
            assert_eq!(state.kingdom[&COPPER], 60);
            assert_eq!(state.kingdom[&ESTATE], 11);
        }
    }

    #[test]
    fn test_victory() {
        for seed in 0..N_TEST_SEEDS {
            let mut state = State::new_default(seed);
            assert_eq!(state.is_terminal(), None);

            for _ in 0..100 {
                if state.is_terminal().is_some() {
                    return;
                }

                if can_play_action(&state, Action::Buy(&ESTATE)) {
                    state = state.apply(Action::Buy(&ESTATE));
                } else {
                    assert_can_play_action(&state, Action::Buy(&COPPER), true);
                    state = state.apply(Action::Buy(&COPPER));
                }
                state = state.apply(Action::EndTurn);
            }

            panic!("Failed to reach victory in {} turns", state.ply);
        }
    }
}
