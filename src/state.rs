use crate::{
    actions::Action,
    cards::Card::{self, *},
    embeddings::{PileType, Token},
    pile::Pile,
    policy::Policy,
    types::Ply,
};
use rand::rngs::SmallRng;
use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct State {
    pub hand: Pile,
    pub draw: Pile,
    pub discard: Pile,
    pub kingdom: Pile,
    pub unspent_gold: u8,
    pub unspent_buys: u8,
    pub ply: Ply,
    pub win_conditions: Vec<WinCondition>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum WinCondition {
    VictoryPoints(u8),
}

pub struct StateBuilder<'a> {
    discard: &'a [(Card, u8)],
    kingdom: &'a [(Card, u8)],
    win_conditions: &'a [WinCondition],
}

impl<'a> Default for StateBuilder<'a> {
    fn default() -> Self {
        Self {
            discard: &Self::DEFAULT_DECK[..],
            kingdom: &Self::DEFAULT_KINGDOM[..],
            win_conditions: &Self::DEFAULT_WIN_CONDITIONS[..],
        }
    }
}

impl<'a> StateBuilder<'a> {
    const DEFAULT_DECK: [(Card, u8); 2] = [(Copper, 7), (Estate, 3)];

    const DEFAULT_KINGDOM: [(Card, u8); 6] = [
        (Copper, 60),
        (Silver, 40),
        (Gold, 30),
        (Estate, 8),
        (Duchy, 10),
        (Province, 12),
    ];

    const DEFAULT_WIN_CONDITIONS: [WinCondition; 1] = [WinCondition::VictoryPoints(10)];

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_discard(mut self, discard: &'a [(Card, u8)]) -> Self {
        self.discard = discard;
        self
    }

    pub fn with_kingdom(mut self, kingdom: &'a [(Card, u8)]) -> Self {
        self.kingdom = kingdom;
        self
    }

    pub fn with_win_conditions(mut self, win_conditions: &'a [WinCondition]) -> Self {
        self.win_conditions = win_conditions;
        self
    }

    pub fn build(self, rng: &mut SmallRng) -> State {
        State::new(self.discard, self.kingdom, self.win_conditions, rng)
    }
}

impl Debug for State {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self)
    }
}

impl Display for State {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "hand: {}\ndraw: {}\ndiscard: {}\nkingdom: {}\ngold {}, buys {}\nply {}\nterminal: {}",
            &self.hand,
            &self.draw,
            &self.discard,
            &self.kingdom,
            self.unspent_gold,
            self.unspent_buys,
            self.ply,
            self.is_terminal().is_some()
        )
    }
}

impl State {
    /// The number of cards we draw at the start of each turn.
    const HAND_SIZE: usize = 5;

    /// The maximum number of plys. Games are terminal if they reach this number.
    pub const MAX_PLY: Ply = 100;

    pub fn new(
        discard: &[(Card, u8)],
        kingdom: &[(Card, u8)],
        win_conditions: &[WinCondition],
        rng: &mut SmallRng,
    ) -> Self {
        let mut ret = Self {
            hand: Pile::default(),
            draw: Pile::default(),
            discard: Pile::from_iter(discard.into_iter().cloned()),
            kingdom: Pile::from_iter(kingdom.into_iter().cloned()).with_preserve_zero(true),
            unspent_gold: 0,
            unspent_buys: 1,
            ply: 0,
            win_conditions: win_conditions.to_vec(),
        };
        ret.draw(Self::HAND_SIZE, rng);

        ret
    }

    /// Increases the ply, discards the hand, draws a new hand, and resets gold/buys.
    fn new_turn(&mut self, rng: &mut SmallRng) -> &mut Self {
        self.ply += 1;

        // Discard hand
        self.discard.drain_from(&mut self.hand);

        // Reset unspent gold and buys
        self.unspent_gold = 0;
        self.unspent_buys = 1;

        // Draw new hand
        self.draw(Self::HAND_SIZE, rng);

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
                self.draw.drain_from(&mut self.discard);
            }

            let card = self.draw.draw(rng);
            self.unspent_gold += card.treasure();
            self.hand.push(card);
        }

        self
    }

    /// For each possible action, returns a tuple of the action and whether it is valid or not.
    pub fn valid_actions(&self) -> [(Action, bool); Action::N_ACTIONS] {
        Action::ALL.map(|action| match action {
            Action::EndTurn => (action, true),
            Action::Play(_) => (action, false),
            Action::Trash(_) => (action, false),
            Action::Buy(card) => (
                action,
                self.unspent_gold >= card.cost()
                    && self.unspent_buys > 0
                    && self.kingdom.contains(card),
            ),
        })
    }

    pub fn can_play_action(&self, action: Action) -> bool {
        self.valid_actions().contains(&(action, true))
    }

    /// Mask the policy logprobs by setting illegal moves to [f32::NEG_INFINITY].
    pub fn mask_policy(&self, policy_logprobs: &mut Policy) {
        let valid_actions = self.valid_actions();

        for i in 0..Action::N_ACTIONS {
            let (_, can_play) = valid_actions[i];
            if !can_play {
                policy_logprobs[i] = f32::NEG_INFINITY;
            }
        }
    }

    pub fn victory_points(&self) -> u8 {
        self.hand.victory_points() + self.draw.victory_points() + self.discard.victory_points()
    }

    /// Returns the ply at which the game is terminal, or None if it is not terminal.
    pub fn is_terminal(&self) -> Option<Ply> {
        if self.ply >= Self::MAX_PLY {
            return Some(self.ply);
        }

        for &win_condition in self.win_conditions.iter() {
            match win_condition {
                WinCondition::VictoryPoints(target_victory) => {
                    if self.victory_points() >= target_victory {
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
            Action::Play(_) => unimplemented!(),
            Action::Trash(_) => unimplemented!(),
            Action::Buy(card) => {
                if !next.kingdom.contains(card) {
                    panic!("Cannot buy unavailable card {}", card.name());
                }
                next.kingdom.take(card);
                assert!(next.unspent_gold >= card.cost(), "Cannot afford card");
                next.unspent_gold -= card.cost();
                assert!(next.unspent_buys > 0, "Tried to buy card with no buys left");
                next.unspent_buys -= 1;
                next.discard.push(card);
            }
        }
        next
    }

    pub fn to_tokens(&self) -> Vec<Token> {
        let mut tokens = Vec::new();
        tokens.extend(self.hand.to_tokens(PileType::Hand));
        tokens.extend(self.draw.to_tokens(PileType::Draw));
        tokens.extend(self.discard.to_tokens(PileType::Discard));
        tokens.extend(self.kingdom.to_tokens(PileType::Kingdom));

        // TODO: Add pile summary tokens

        tokens
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    pub fn assert_can_play_action(state: &State, action: Action, can_play: bool) {
        assert_eq!(
            state.can_play_action(action),
            can_play,
            "Attempting to play invalid action {}",
            action
        );
    }

    proptest! {
        #[test]
        fn test_sanity(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let state = StateBuilder::new().build(&mut rng);
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
            let state = StateBuilder::new()
                .with_kingdom(&[(Copper, 10), (Estate, 10)])
                .build(&mut rng);
            let unspent_gold = state.unspent_gold;

            // Buy a copper
            assert_can_play_action(&state, Action::Buy(Copper), true);
            let state = state.apply_action(Action::Buy(Copper), &mut rng);
            assert_eq!(state.unspent_gold, unspent_gold - Copper.cost());
            assert!(state.discard.contains(Copper));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(Copper), false);
            assert_can_play_action(&state, Action::Buy(Estate), false);

            // End turn
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply_action(Action::EndTurn, &mut rng);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 6); // Includes new copper
            assert_eq!(state.kingdom[Copper], 9);
            assert_eq!(state.kingdom[Estate], 10);
        }

        #[test]
        fn test_buy_estate(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let state = StateBuilder::new()
                .with_kingdom(&[(Copper, 10), (Estate, 10)])
                .build(&mut rng);
            let unspent_gold = state.unspent_gold;

            // Buy an estate
            assert_can_play_action(&state, Action::Buy(Estate), true);
            let state = state.apply_action(Action::Buy(Estate), &mut rng);
            assert_eq!(state.unspent_gold, unspent_gold - Estate.cost());
            assert!(state.discard.contains(Estate));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(Copper), false);
            assert_can_play_action(&state, Action::Buy(Estate), false);

            // End turn
            assert_can_play_action(&state, Action::EndTurn, true);
            let state = state.apply_action(Action::EndTurn, &mut rng);
            assert_eq!(state.ply, 1);
            assert_eq!(state.hand.len(), 5);
            assert_eq!(state.draw.len(), 0);
            assert_eq!(state.discard.len(), 6); // Includes new estate
            assert_eq!(state.kingdom.unique_card_count(), 2);
            assert_eq!(state.kingdom[Copper], 10);
            assert_eq!(state.kingdom[Estate], 9);
        }

        #[test]
        fn test_victory(seed in 0..u64::MAX) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = StateBuilder::new()
                .with_discard(&[(Copper, 5), (Estate, 5)])
                .with_kingdom(&[(Copper, 10), (Estate, 10)])
                .with_win_conditions(&[WinCondition::VictoryPoints(10)])
                .build(&mut rng);
            assert_eq!(state.is_terminal(), None);

            while state.ply < State::MAX_PLY {
                if state.is_terminal().is_some() {
                    return Ok(());
                }
                if state.can_play_action(Action::Buy(Estate)) {
                    state = state.apply_action(Action::Buy(Estate), &mut rng);
                } else if state.can_play_action(Action::Buy(Copper)) {
                    state = state.apply_action(Action::Buy(Copper), &mut rng);
                }
                assert_can_play_action(&state, Action::EndTurn, true);
                state = state.apply_action(Action::EndTurn, &mut rng);
            }

            prop_assert!(false, "Failed to reach victory in {} turns", state.ply);
        }
    }
}
