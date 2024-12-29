use std::{
    array,
    cell::RefCell,
    rc::{Rc, Weak},
};

use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::SmallRng, SeedableRng};

use crate::{
    actions::{Action, ACTION_SET, N_ACTIONS},
    state::State,
    types::{policy_from_iter, Policy, QValue},
    utils::OrdF32,
};

#[derive(Debug)]
pub struct MCTS {
    root: Rc<RefCell<Node>>,
    leaf: Rc<RefCell<Node>>,
    actions: Vec<Action>,
    rng: SmallRng,
}

impl MCTS {
    const UNIFORM_POLICY: Policy = [1.0 / N_ACTIONS as f32; N_ACTIONS];
    const MAX_PLY: usize = 100;
    const AVG_PLY: usize = 30;
    const C_EXPLORATION: f32 = 8.0;

    pub fn new(state: State, seed: u64) -> Self {
        let root = Rc::new(RefCell::new(Node::new(state, 0.0)));
        Self {
            root: root.clone(),
            leaf: root,
            actions: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn on_received_nn_est(&mut self, q_est: QValue, mut policy_logprobs_est: Policy) {
        let leaf = self.leaf.borrow();

        if let Some(q_actual) = leaf.get_terminal_q_value(MCTS::MAX_PLY, MCTS::AVG_PLY) {
            drop(leaf); // Drop leaf borrow so we can reassign self.leaf

            // If this is a terminal state, the received policy is irrelevant.
            // We backpropagate the objective terminal value and select a new leaf.
            self.backpropagate(q_actual);
            self.select_new_leaf();
        } else {
            // Mask the NN's policy to only include legal moves.
            leaf.state.mask_policy(&mut policy_logprobs_est);
            drop(leaf); // Drop leaf borrow so we can reassign self.leaf

            // If this is a non-terminal state, we expand the leaf and backpropagate the NN
            // estimate.
            let policy_est = softmax(policy_logprobs_est);
            self.expand_leaf(policy_est);
            self.backpropagate(q_est);
            self.select_new_leaf();
        }
    }

    /// Expands the the leaf by adding child nodes to it which then be eligible for exploration via
    /// subsequent MCTS iterations. Each child node's [Node::initial_policy_value] is determined by
    /// the provided policy.
    /// Noop for terminal nodes.
    fn expand_leaf(&mut self, policy_probs: Policy) {
        if self.leaf.borrow().state.is_terminal().is_some() {
            return;
        }
        let legal_moves = self.leaf.borrow().state.valid_actions();

        let children: [Option<Rc<RefCell<Node>>>; N_ACTIONS] = std::array::from_fn(|i| {
            let (action, can_play) = legal_moves[i];
            if can_play {
                let child_state = self.leaf.borrow().state.apply_action(action, &mut self.rng);
                let child = Node::new(child_state, policy_probs[i]);
                Some(Rc::new(RefCell::new(child)))
            } else {
                None
            }
        });
        let mut leaf = self.leaf.borrow_mut();
        leaf.children = Some(children);
    }

    /// Backpropagates the Q value up the tree, incrementing visit counts.
    pub fn backpropagate(&mut self, q_value: QValue) {
        let mut node_ref = Rc::clone(&self.leaf);
        loop {
            let mut node = node_ref.borrow_mut();
            node.q_sum += q_value;
            node.visit_count += 1;

            if let Some(parent) = node.parent.upgrade() {
                drop(node); // Drop node borrow so we can reassign node_ref
                node_ref = parent;
            } else {
                break;
            }
        }
    }

    /// Select the next leaf node by traversing from the root node, repeatedly selecting the child
    /// with the highest [Node::uct_value] until we reach a node with no expanded children (leaf
    /// node).
    pub fn select_new_leaf(&mut self) {
        let mut node_ref = Rc::clone(&self.root);
        loop {
            let node = node_ref.borrow();
            let children = node.children.as_ref().cloned();

            if let Some(children) = children {
                let max_child = children.iter().flatten().max_by_key(|&child| {
                    let score = child.borrow().uct_value(MCTS::C_EXPLORATION);
                    OrdF32(score)
                });

                if let Some(next) = max_child {
                    drop(node); // Drop node borrow so we can reassign node_ref
                    node_ref = Rc::clone(&next);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.leaf = node_ref;
    }

    /// Makes a move, updating the root node to be the child node corresponding to the action.
    /// Stores the previous position and policy in the [Self::moves] vector.
    pub fn make_move(&mut self, action: Action) {
        let leaf = self.leaf.borrow_mut();
        let child_idx = ACTION_SET
            .iter()
            .position(|&a| a == action)
            .expect(format!("Action {:?} not in ACTION_SET", action).as_str());

        let child = leaf
            .children
            .as_ref()
            .expect("apply_action called on leaf with no children")[child_idx]
            .as_ref()
            .expect("illegal action");
        self.root = Rc::clone(child);

        drop(leaf); // Drop leaf borrow so we can reassign self.leaf
        self.select_new_leaf();
        self.actions.push(action);
    }

    /// Makes a move probabalistically based on the root node's policy.
    ///
    /// The temperature parameter scales the policy probabilities, with values > 1.0 making the
    /// sampled distribution more uniform and values < 1.0 making the sampled distribution favor
    /// the most lucrative moves.
    pub fn make_random_move(&mut self, temperature: f32) {
        let policy = self.root.borrow().policy();
        let policy = apply_temperature(&policy, temperature);
        let dist = WeightedIndex::new(policy).unwrap();
        let action = ACTION_SET[dist.sample(&mut self.rng)];
        self.make_move(action);
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

    pub fn new(state: State, initial_policy_value: f32) -> Self {
        Self {
            state,
            parent: Weak::new(),
            visit_count: 0,
            q_sum: 0.0,
            initial_policy_value,
            children: None,
        }
    }

    /// The exploitation component of the UCT value.
    pub fn q_value(&self) -> f32 {
        self.q_sum / self.visit_count as f32
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
    fn get_terminal_q_value(&self, max_ply: usize, avg_ply: usize) -> Option<QValue> {
        if self.state.is_terminal().is_some() {
            return None;
        }
        let ply = self.state.ply as usize;

        // For wins/losses, map ply from [0, max_ply] to [-1, 1]
        // with avg_ply mapping to 0
        let score = if ply <= avg_ply {
            // Early segment: map [0, avg_ply] to [1, 0]
            1.0 - (ply as f32 / avg_ply as f32)
        } else {
            // Late segment: map [avg_ply, max_ply] to [0, -1]
            -((ply - avg_ply) as f32) / ((max_ply - avg_ply) as f32)
        };
        Some(score)
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
}

/// Softmax function for a policy.
fn softmax(policy_logprobs: Policy) -> Policy {
    let max = policy_logprobs
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        // If the policy is all negative infinity, we fall back to uniform policy.
        // This can happen if the NN dramatically underflows.
        // We panic as this is an issue that should be fixed in the NN.
        panic!("softmax: policy is all negative infinity, debug NN on why this is happening.");
    }
    let exps = policy_logprobs
        .iter()
        // Subtract max value to avoid overflow
        .map(|p| (p - max).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    array::from_fn(|i| exps[i] / sum)
}

/// Applies temperature scaling to a policy.
/// Expects the policy to be in [0-1] (non-log) space.
/// Temperature=0.0 is argmax, temperature=1.0 is a noop.
pub fn apply_temperature(policy: &Policy, temperature: f32) -> Policy {
    if temperature == 1.0 || policy.iter().all(|&p| p == policy[0]) {
        // Temp 1.0 or uniform policy is noop
        return policy.clone();
    } else if temperature == 0.0 {
        // Temp 0.0 is argmax
        let max = policy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ret = policy.map(|p| if p == max { 1.0 } else { 0.0 });
        let sum = ret.iter().sum::<f32>();
        return ret.map(|p| p / sum); // Potentially multiple argmaxes
    }

    let policy_log = policy.map(|p| p.ln() / temperature);
    let policy_log_sum_exp = policy_log.map(|p| p.exp()).iter().sum::<f32>().ln();
    policy_log.map(|p| (p - policy_log_sum_exp).exp().clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const CONST_COL_WEIGHT: f32 = 1.0 / N_ACTIONS as f32;

    /// Strategy for generating a policy with at least one non-zero value.
    fn policy_strategy() -> impl Strategy<Value = Policy> {
        let min = 0.0f32;
        let max = 10.0f32;
        let positive_strategy = min..max;
        let neg_inf_strategy = Just(f32::NEG_INFINITY);
        prop::array::uniform3(prop_oneof![positive_strategy, neg_inf_strategy])
            .prop_filter("all neg infinity not allowed", |policy_logits| {
                !policy_logits.iter().all(|&p| p == f32::NEG_INFINITY)
            })
            .prop_map(|policy_log| softmax(policy_log))
    }

    proptest! {
        /// Softmax policies should sum up to one.
        #[test]
        fn softmax_sum_1(policy in policy_strategy()) {
            assert_policy_sum_1(&policy);
        }

        /// Temperature of 1.0 should not affect the policy.
        #[test]
        fn temperature_1(policy in policy_strategy()) {
            let policy_with_temp = apply_temperature(&policy, 1.0);
            assert_policy_eq(&policy, &policy_with_temp, 1e-5);
        }

        /// Temperature of 2.0 should change the policy.
        #[test]
        fn temperature_2(policy in policy_strategy()) {
            let policy_with_temp = apply_temperature(&policy, 2.0);
            assert_policy_sum_1(&policy_with_temp);
            // If policy is nonuniform and there are at least two non-zero probabilities, the
            // policy with temperature should be different from the original policy
            if policy.iter().filter(|&&p| p != CONST_COL_WEIGHT && p > 0.0).count() >= 2 {
                assert_policy_ne(&policy, &policy_with_temp, Node::EPS);
            }
        }

        /// Temperature of 0.0 should be argmax.
        #[test]
        fn temperature_0(policy in policy_strategy()) {
            let policy_with_temp = apply_temperature(&policy, 0.0);
            let max = policy_with_temp.iter().fold(f32::NEG_INFINITY, |a, &b| f32::max(a, b));
            let max_count = policy_with_temp.iter().filter(|&&p| p == max).count() as f32;
            assert_policy_sum_1(&policy_with_temp);
            for p in policy_with_temp {
                if p == max {
                    assert_eq!(1.0 / max_count, p);
                }
            }
        }
    }

    fn assert_policy_sum_1(policy: &Policy) {
        let sum = policy.iter().sum::<f32>();
        if (sum - 1.0).abs() > 1e-5 {
            panic!("policy sum {:?} is not 1.0: {:?}", sum, policy);
        }
    }

    fn assert_policy_eq(p1: &Policy, p2: &Policy, epsilon: f32) {
        let eq = p1
            .iter()
            .zip(p2.iter())
            .all(|(a, b)| (a - b).abs() < epsilon);
        if !eq {
            panic!("policies are not equal: {:?} {:?}", p1, p2);
        }
    }

    fn assert_policy_ne(p1: &Policy, p2: &Policy, epsilon: f32) {
        let ne = p1
            .iter()
            .zip(p2.iter())
            .any(|(a, b)| (a - b).abs() > epsilon);
        if !ne {
            panic!("policies are equal: {:?} {:?}", p1, p2);
        }
    }
}
