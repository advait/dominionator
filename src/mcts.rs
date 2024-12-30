use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::SmallRng};

use crate::{
    actions::{action_to_idx, Action, ACTION_SET, N_ACTIONS},
    policy::{apply_temperature, policy_from_iter, softmax, Policy},
    state::State,
    types::QValue,
    utils::OrdF32,
};

#[derive(Debug)]
pub struct MCTS {
    root: Rc<RefCell<Node>>,
    leaf: Rc<RefCell<Node>>,
    actions: Vec<Action>,
    rng: SmallRng,
}

pub struct NNEst {
    q: QValue,
    policy_logprobs: Policy,
    c_exploration: f32,
    max_ply: u8,
    avg_ply: u8,
}

impl MCTS {
    const UNIFORM_POLICY: Policy = [1.0 / N_ACTIONS as f32; N_ACTIONS];

    pub fn new(state: State, rng: SmallRng) -> Self {
        let root = Rc::new(RefCell::new(Node::new(Weak::new(), state, 0.0)));
        Self {
            root: root.clone(),
            leaf: root,
            actions: Vec::new(),
            rng,
        }
    }

    pub fn root_visit_count(&self) -> usize {
        self.root.borrow().visit_count
    }

    pub fn on_received_nn_est(&mut self, mut est: NNEst) {
        let mut leaf = self.leaf.borrow_mut();
        if let Some(q_actual) = leaf.get_terminal_q_value(est.max_ply, est.avg_ply) {
            // If we've reached a terminal state, backpropagate the actual q value and attempt
            // to select a new leaf node.
            leaf.backpropagate(q_actual);
            drop(leaf); // Drop leaf borrow so we can reassign self.leaf

            self.select_new_leaf(est.c_exploration);
        } else {
            // Non-terminal state found, proceed with normal expansion
            leaf.state.mask_policy(&mut est.policy_logprobs);
            let policy_est = softmax(est.policy_logprobs);
            leaf.expand(self.leaf.clone(), policy_est, &mut self.rng);
            leaf.backpropagate(est.q);

            drop(leaf); // Drop leaf borrow so we can reassign self.leaf
            self.select_new_leaf(est.c_exploration);
        }
    }

    /// Select the next leaf node by traversing from the root node, repeatedly selecting the child
    /// with the highest [Node::uct_value] until we reach a node with no expanded children (leaf
    /// node).
    pub fn select_new_leaf(&mut self, c_exploration: f32) {
        let mut node_ref = Rc::clone(&self.root);
        loop {
            let node = node_ref.borrow();
            let best_child = node.best_child(c_exploration);
            if let Some(best_child) = best_child {
                drop(node); // Drop node borrow so we can reassign node_ref
                node_ref = Rc::clone(&best_child);
                // TODO: If this child is a terminal state, we can preemptively backpropagate
                // the q value and select a new leaf node instead of waiting for the NN to
                // evaluate a terminal position.
            } else {
                break;
            }
        }

        self.leaf = node_ref;
    }

    /// Makes a move, updating the root node to be the child node corresponding to the action.
    /// Stores the previous position and policy in the [Self::moves] vector.
    pub fn make_move(&mut self, action: Action, c_exploration: f32) {
        let root = self.root.borrow_mut();
        let child_idx = action_to_idx(&action);

        let child = root
            .children
            .as_ref()
            .expect("apply_action called on leaf with no children")[child_idx]
            .as_ref()
            .expect("illegal action");
        let child = Rc::clone(child);
        drop(root); // Drop root borrow so we can reassign self.root
        self.root = child;

        self.select_new_leaf(c_exploration);
        self.actions.push(action);
    }

    /// Makes a move probabalistically based on the root node's policy.
    ///
    /// The temperature parameter scales the policy probabilities, with values > 1.0 making the
    /// sampled distribution more uniform and values < 1.0 making the sampled distribution favor
    /// the most lucrative moves.
    pub fn make_random_move(&mut self, temperature: f32, c_exploration: f32) {
        let policy = self.root.borrow().policy();
        let policy = apply_temperature(&policy, temperature);
        let dist = WeightedIndex::new(policy).unwrap();
        let action = ACTION_SET[dist.sample(&mut self.rng)];
        self.make_move(action, c_exploration);
    }
}

/// A node within an MCTS tree.
/// [Self::parent] is a weak reference to the parent node to avoid reference cycles.
/// [Self::children] is an array of optional child nodes. If a child is None, it means that the
/// move is illegal. Otherwise the child is a [Rc<RefCell<Node>>] reference to the child node.
/// We maintain two separate Q values: one with ply penalties applied ([Self::q_sum_penalty]) and
/// one without ([Self::q_sum_no_penalty]). These are normalized with [Self::visit_count] to get the
/// average [QValue]s in [Self::q_with_penalty()] and [Self::q_no_penalty()].
#[derive(Debug, Clone)]
struct Node {
    state: State,
    parent: Weak<RefCell<Node>>,
    visit_count: usize,
    q_sum: QValue,
    initial_policy_value: QValue,
    children: Option<[Option<Rc<RefCell<Node>>>; N_ACTIONS]>,
}

impl Node {
    const EPS: f32 = 1e-8;

    pub fn new(parent: Weak<RefCell<Node>>, state: State, initial_policy_value: f32) -> Self {
        Self {
            state,
            parent,
            visit_count: 0,
            q_sum: 0.0,
            initial_policy_value,
            children: None,
        }
    }

    /// The exploitation component of the UCT value.
    pub fn q_value(&self) -> f32 {
        self.q_sum / (self.visit_count as f32 + 1.0)
    }

    /// The exploration component of the UCT value. Higher visit counts result in lower values.
    /// We also weight the exploration value by the initial policy value to allow the network
    /// to guide the search.
    fn exploration_value(&self) -> QValue {
        let parent_visit_count = self
            .parent
            .upgrade()
            .map_or(self.visit_count as f32, |parent| {
                parent.borrow().visit_count as f32
            }) as f32;
        let exploration_value = (parent_visit_count.ln() / (self.visit_count as f32 + 1.)).sqrt();
        exploration_value * (self.initial_policy_value + Self::EPS)
    }

    /// The UCT value of this node. Represents the lucrativeness of this node according to MCTS.
    fn uct_value(&self, c_exploration: f32) -> QValue {
        self.q_value() + c_exploration * self.exploration_value()
    }

    /// Returns the Q value of the terminal state between [-1, 1].
    /// Returns None if the state is not terminal.
    fn get_terminal_q_value(&self, max_ply: u8, avg_ply: u8) -> Option<QValue> {
        if self.state.is_terminal().is_none() {
            return None;
        }
        let ply = self.state.ply as f32;
        let avg_ply = avg_ply as f32;
        let max_ply = max_ply as f32;

        // For wins/losses, map ply from [0, max_ply] to [-1, 1]
        // with avg_ply mapping to 0
        let q = if ply <= avg_ply {
            // Early segment: map [0, avg_ply] to [1, 0]
            1.0 - (ply / avg_ply)
        } else {
            // Late segment: map [avg_ply, max_ply] to [0, -1]
            -(ply - avg_ply) / (max_ply - avg_ply)
        };
        Some(q)
    }

    /// Returns the child with the highest UCT value.
    fn best_child(&self, c_exploration: f32) -> Option<Rc<RefCell<Node>>> {
        self.children
            .as_ref()?
            .iter()
            .flatten()
            .max_by_key(|&child| {
                let score = child.borrow().uct_value(c_exploration);
                OrdF32(score)
            })
            .cloned()
    }

    /// Uses the child counts as weights to determine the implied policy from this position.
    fn policy(&self) -> Policy {
        if let Some(children) = &self.children {
            let child_counts = policy_from_iter(children.iter().map(|maybe_child| {
                maybe_child
                    .as_ref()
                    .map_or(0., |child_ref| child_ref.borrow().visit_count as f32)
            }));
            let child_counts_sum = child_counts.iter().sum::<f32>();
            if child_counts_sum == 0.0 {
                MCTS::UNIFORM_POLICY
            } else {
                softmax(child_counts)
            }
        } else {
            MCTS::UNIFORM_POLICY
        }
    }

    fn backpropagate(&mut self, q_value: QValue) {
        self.q_sum += q_value;
        self.visit_count += 1;

        if let Some(parent) = self.parent.upgrade() {
            parent.borrow_mut().backpropagate(q_value);
        }
    }

    fn expand(&mut self, parent_ref: Rc<RefCell<Node>>, policy_probs: Policy, rng: &mut SmallRng) {
        if self.children.is_some() {
            panic!("expand called on node with children");
        }

        let legal_moves = self.state.valid_actions();
        let children: [Option<Rc<RefCell<Node>>>; N_ACTIONS] = std::array::from_fn(|i| {
            let (action, can_play) = legal_moves[i];
            if can_play {
                let child_state = self.state.apply_action(action, rng);
                let child = Node::new(Rc::downgrade(&parent_ref), child_state, policy_probs[i]);
                Some(Rc::new(RefCell::new(child)))
            } else {
                None
            }
        });
        self.children = Some(children);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cards::*,
        policy::policy_value_for_action,
        state::{tests::assert_can_play_action, StateBuilder, WinCondition},
    };

    use super::*;
    use more_asserts::assert_gt;
    use rand::SeedableRng;

    /// If we can buy an estate to win, we should do so.
    #[test]
    fn test_obvious_win_buy_estate() {
        let c_exploration = 2.0;
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_discard(&[&COPPER, &COPPER, &COPPER, &COPPER, &COPPER])
            .with_kingdom(&[(&COPPER, 1), (&ESTATE, 1)])
            .with_win_conditions(&[WinCondition::VictoryPoints(1)])
            .build(&mut rng);
        assert_eq!(state.is_terminal(), None);
        let mut mcts = MCTS::new(state, rng);

        while mcts.root_visit_count() < 10 {
            mcts.on_received_nn_est(NNEst {
                q: 0.0,
                policy_logprobs: MCTS::UNIFORM_POLICY,
                c_exploration,
                max_ply: 5,
                avg_ply: 3,
            });
        }
        let policy = mcts.root.borrow().policy();
        assert_gt!(
            policy_value_for_action(&policy, &Action::Buy(&ESTATE)),
            0.99
        );

        // Buy the estate
        mcts.make_random_move(0.0, c_exploration);
        assert!(mcts.root.borrow().state.is_terminal().is_some());
    }

    /// If we can buy a Gold, Silver, and Copper, we should buy Gold.
    #[test]
    fn test_obvious_win_buy_gold() {
        let c_exploration = 2.0;
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_discard(&[&COPPER, &COPPER, &COPPER, &COPPER, &SILVER])
            .with_kingdom(&[(&COPPER, 5), (&SILVER, 5), (&GOLD, 5), (&PROVINCE, 1)])
            .with_win_conditions(&[WinCondition::VictoryPoints(6)])
            .build(&mut rng);
        assert_eq!(state.is_terminal(), None);
        let mut mcts = MCTS::new(state, rng);

        while mcts.root_visit_count() < 100 {
            mcts.on_received_nn_est(NNEst {
                q: 0.0,
                policy_logprobs: MCTS::UNIFORM_POLICY,
                c_exploration,
                max_ply: 7,
                avg_ply: 3,
            });
        }
        let policy = mcts.root.borrow().policy();
        assert_gt!(policy_value_for_action(&policy, &Action::Buy(&GOLD)), 0.99);
        assert_can_play_action(&mcts.root.borrow().state, Action::Buy(&PROVINCE), false);
    }

    /// If we can buy a province and gold and the win target is 7vp, we should buy the province.
    #[test]
    fn test_buy_province_over_gold() {
        let c_exploration = 2.0;
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_discard(&[&GOLD, &GOLD, &SILVER])
            .with_kingdom(&[
                (&COPPER, 5),
                (&SILVER, 5),
                (&GOLD, 5),
                (&ESTATE, 5),
                (&DUCHY, 5),
                (&PROVINCE, 5),
            ])
            .with_win_conditions(&[WinCondition::VictoryPoints(7)])
            .build(&mut rng);
        assert_eq!(state.is_terminal(), None);
        let mut mcts = MCTS::new(state, rng);

        while mcts.root_visit_count() < 100 {
            mcts.on_received_nn_est(NNEst {
                q: 0.0,
                policy_logprobs: MCTS::UNIFORM_POLICY,
                c_exploration,
                max_ply: 7,
                avg_ply: 3,
            });
        }
        let policy = mcts.root.borrow().policy();
        assert_gt!(
            policy_value_for_action(&policy, &Action::Buy(&PROVINCE)),
            0.99
        );
    }

    #[test]
    fn test_terminal_q_value() {
        let mut rng = SmallRng::seed_from_u64(1);
        let max_ply = 100;
        let avg_ply = 30;

        // Non-terminal state should return None
        let non_terminal_state = State::new(
            &[&COPPER, &COPPER, &COPPER, &COPPER, &COPPER],
            &[(&COPPER, 1), (&ESTATE, 1)],
            &[WinCondition::VictoryPoints(100)],
            &mut rng,
        );
        let node = Node::new(Weak::new(), non_terminal_state, 0.0);
        assert_eq!(node.get_terminal_q_value(max_ply, avg_ply), None);

        let terminal_state = State::new(
            &[&COPPER, &COPPER, &COPPER, &COPPER, &COPPER],
            &[(&COPPER, 1), (&ESTATE, 1)],
            &[WinCondition::VictoryPoints(0)],
            &mut rng,
        );
        let test_cases = [
            (0, Some(1.0)),           // Terminal state at ply 0 should return 1.0
            (avg_ply, Some(0.0)),     // Terminal state at avg_ply should return 0.0
            (max_ply, Some(-1.0)),    // Terminal state at max_ply should return -1.0
            (avg_ply / 2, Some(0.5)), // Terminal state at ply 15 (halfway between 0 and avg_ply) should return 0.5
            (avg_ply + (max_ply - avg_ply) / 2, Some(-0.5)), // Terminal state at ply 65 (halfway between avg_ply and max_ply) should return -0.5
        ];
        for &(ply, expected) in &test_cases {
            let mut node = Node::new(Weak::new(), terminal_state.clone(), 0.0);
            node.state.ply = ply;
            assert_eq!(node.get_terminal_q_value(max_ply, avg_ply), expected);
        }
    }
}
