use crate::{
    actions::Action,
    cards::Card::{self, *},
    embeddings::{PileType, TokenExt, N_EMBEDDINGS_PER_TOKEN},
    pile::Pile,
    policy::Policy,
    types::Ply,
};
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct State {
    pub hand: Pile,
    pub draw: Pile,
    pub discard: Pile,
    pub kingdom: Pile,
    pub unspent_gold: u8,
    pub unspent_buys: u8,
    pub unspent_actions: u8,
    pub turn_phase: TurnPhase,
    pub ply: Ply,
    pub win_conditions: Vec<WinCondition>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum TurnPhase {
    ActionPhase,
    BuyPhase,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum WinCondition {
    VictoryPoints(u8),
}

pub struct StateBuilder<'a> {
    hand: HandBuilder<'a>,
    draw: &'a [(Card, u8)],
    discard: &'a [(Card, u8)],
    kingdom: &'a [(Card, u8)],
    win_conditions: &'a [WinCondition],
}

impl<'a> Default for StateBuilder<'a> {
    fn default() -> Self {
        Self {
            hand: HandBuilder::RandomHand,
            draw: &Self::DEFAULT_DECK,
            discard: &[],
            kingdom: &Self::DEFAULT_KINGDOM,
            win_conditions: &Self::DEFAULT_WIN_CONDITIONS,
        }
    }
}

enum HandBuilder<'a> {
    RandomHand,
    SpecificHand(&'a [(Card, u8)]),
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

    pub fn with_hand(mut self, hand: &'a [(Card, u8)]) -> Self {
        self.hand = HandBuilder::SpecificHand(hand);
        self
    }

    pub fn with_discard(mut self, discard: &'a [(Card, u8)]) -> Self {
        self.discard = discard;
        self
    }

    pub fn with_draw(mut self, draw: &'a [(Card, u8)]) -> Self {
        self.draw = draw;
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
        let hand = match self.hand {
            HandBuilder::RandomHand => &[],
            HandBuilder::SpecificHand(hand) => hand,
        };

        let mut state = State::new(
            hand,
            self.draw,
            self.discard,
            self.kingdom,
            self.win_conditions,
        );

        if matches!(self.hand, HandBuilder::RandomHand) {
            state.draw(State::HAND_SIZE, rng);
        }
        state
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
        hand: &[(Card, u8)],
        draw: &[(Card, u8)],
        discard: &[(Card, u8)],
        kingdom: &[(Card, u8)],
        win_conditions: &[WinCondition],
    ) -> Self {
        let kingdom = kingdom.into_iter().cloned().collect::<Pile>();
        let hand = Pile::from_kingdom(&kingdom).with_counts(hand);
        let draw = Pile::from_kingdom(&kingdom).with_counts(draw);
        let discard = Pile::from_kingdom(&kingdom).with_counts(discard);

        Self {
            unspent_gold: hand.treasure(),
            unspent_buys: 1,
            unspent_actions: 1,
            hand,
            draw,
            discard,
            kingdom,
            turn_phase: TurnPhase::ActionPhase,
            ply: 0,
            win_conditions: win_conditions.to_vec(),
        }
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
        match self.turn_phase {
            TurnPhase::ActionPhase => Action::ALL.map(|action| match action {
                Action::EndPhase => (action, true),
                Action::Play(card) => (
                    action,
                    card.is_action() && self.hand[card] > 0 && self.unspent_actions > 0,
                ),
                Action::Trash(_) => (action, false),
                Action::Buy(_) => (action, false),
            }),
            TurnPhase::BuyPhase => Action::ALL.map(|action| match action {
                Action::EndPhase => (action, true),
                Action::Play(_) => (action, false),
                Action::Trash(_) => (action, false),
                Action::Buy(card) => (
                    action,
                    self.unspent_gold >= card.cost()
                        && self.unspent_buys > 0
                        && self.kingdom[card] > 0,
                ),
            }),
        }
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

        if self.win_conditions.iter().any(|&cond| match cond {
            WinCondition::VictoryPoints(target) => self.victory_points() >= target,
        }) {
            Some(self.ply)
        } else {
            None
        }
    }

    /// Applies an action to the state returning the resulting state.
    pub fn apply_action(&self, action: Action, rng: &mut SmallRng) -> Self {
        let mut next = self.clone();
        match action {
            Action::EndPhase => match next.turn_phase {
                TurnPhase::ActionPhase => {
                    next.turn_phase = TurnPhase::BuyPhase;
                }
                TurnPhase::BuyPhase => {
                    // When buy phaase is over, perform cleanup
                    next.turn_phase = TurnPhase::ActionPhase;
                    next.ply += 1;

                    // Discard hand
                    next.discard.drain_from(&mut next.hand);

                    // Reset unspent gold and buys
                    next.unspent_gold = 0;
                    next.unspent_buys = 1;
                    next.unspent_actions = 1;

                    // Draw new hand
                    next.draw(Self::HAND_SIZE, rng);
                }
            },
            Action::Play(card) => {
                next.unspent_actions -= 1;
                next.hand.take(card);
                next.discard.push(card);
                match card {
                    Smithy => {
                        next.draw(3, rng);
                    }
                    Village => {
                        next.unspent_actions += 2;
                        next.draw(1, rng);
                    }
                    _ => panic!("Cannot play card {}", card.name()),
                }
            }
            Action::Trash(_) => unimplemented!(),
            Action::Buy(card) => {
                next.kingdom.take(card);
                next.unspent_gold -= card.cost();
                next.unspent_buys -= 1;
                next.discard.push(card);
            }
        }
        next
    }

    pub fn to_tokens_indices(&self) -> Vec<[usize; N_EMBEDDINGS_PER_TOKEN]> {
        let mut tokens = Vec::new();
        tokens.extend(self.hand.to_tokens(PileType::Hand));
        tokens.extend(self.draw.to_tokens(PileType::Draw));
        tokens.extend(self.discard.to_tokens(PileType::Discard));
        tokens.extend(self.kingdom.to_tokens(PileType::Kingdom));

        // TODO: Add pile summary tokens

        TokenExt::to_token_indices(&tokens)
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

            assert_eq!(state.turn_phase, TurnPhase::ActionPhase);
            assert_can_play_action(&state, Action::EndPhase, true);
            let state = state.apply_action(Action::EndPhase, &mut rng);

            assert_eq!(state.turn_phase, TurnPhase::BuyPhase);
            assert_can_play_action(&state, Action::EndPhase, true);
            let state = state.apply_action(Action::EndPhase, &mut rng);

            assert_eq!(state.turn_phase, TurnPhase::ActionPhase);
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

            assert_eq!(state.turn_phase, TurnPhase::ActionPhase);
            assert_can_play_action(&state, Action::EndPhase, true);
            let state = state.apply_action(Action::EndPhase, &mut rng);
            assert_eq!(state.turn_phase, TurnPhase::BuyPhase);

            // Buy a copper
            assert_can_play_action(&state, Action::Buy(Copper), true);
            let state = state.apply_action(Action::Buy(Copper), &mut rng);
            assert_eq!(state.unspent_gold, unspent_gold - Copper.cost());
            assert!(state.discard.contains(Copper));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(Copper), false);
            assert_can_play_action(&state, Action::Buy(Estate), false);

            // End turn
            assert_can_play_action(&state, Action::EndPhase, true);
            let state = state.apply_action(Action::EndPhase, &mut rng);
            assert_eq!(state.turn_phase, TurnPhase::ActionPhase);
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

            assert_eq!(state.turn_phase, TurnPhase::ActionPhase);
            assert_can_play_action(&state, Action::EndPhase, true);
            let state = state.apply_action(Action::EndPhase, &mut rng);
            assert_eq!(state.turn_phase, TurnPhase::BuyPhase);

            // Buy an estate
            assert_can_play_action(&state, Action::Buy(Estate), true);
            let state = state.apply_action(Action::Buy(Estate), &mut rng);
            assert_eq!(state.unspent_gold, unspent_gold - Estate.cost());
            assert!(state.discard.contains(Estate));

            // Can't buy additional cards
            assert_can_play_action(&state, Action::Buy(Copper), false);
            assert_can_play_action(&state, Action::Buy(Estate), false);

            // End turn
            assert_can_play_action(&state, Action::EndPhase, true);
            let state = state.apply_action(Action::EndPhase, &mut rng);
            assert_eq!(state.turn_phase, TurnPhase::ActionPhase);
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
                assert_can_play_action(&state, Action::EndPhase, true);
                state = state.apply_action(Action::EndPhase, &mut rng);
            }

            prop_assert!(false, "Failed to reach victory in {} turns", state.ply);
        }
    }
}
