use std::{
    array,
    cell::RefCell,
    rc::{Rc, Weak},
};

use rand::{distributions::WeightedIndex, prelude::Distribution, rngs::SmallRng};

use crate::{
    actions::Action,
    policy::{Policy, PolicyExt},
    state::State,
    types::{GameMetadata, GameResult, NNEst, QValue, Sample},
    utils::OrdF32,
};

#[derive(Debug)]
pub struct MCTSAction {
    prev_node: Rc<RefCell<Node>>,
    pub action: Action,
}

#[derive(Debug)]
pub struct MCTS {
    metadata: GameMetadata,
    root: Rc<RefCell<Node>>,
    leaf: Rc<RefCell<Node>>,
    actions: Vec<MCTSAction>,
    rng: SmallRng,
}

/// SAFETY: MCTS is Send because it doesn't have any public methods that expose the Rc/RefCell
/// which would allow for illegal cross-thread mutation.
unsafe impl Send for MCTS {}

impl MCTS {
    pub const UNIFORM_POLICY: Policy = [1.0 / Action::N_ACTIONS as f32; Action::N_ACTIONS];

    pub fn new(metadata: GameMetadata, state: State, rng: SmallRng) -> Self {
        let root = Rc::new(RefCell::new(Node::new(Weak::new(), state, 0.0)));
        Self {
            metadata,
            root: root.clone(),
            leaf: root,
            actions: Vec::new(),
            rng,
        }
    }

    pub fn root_visit_count(&self) -> usize {
        self.root.borrow().visit_count
    }

    pub fn on_received_nn_est(&mut self, mut est: NNEst, c_exploration: f32) {
        let mut leaf = self.leaf.borrow_mut();
        if let Some(q_actual) = leaf.get_terminal_q_value() {
            // If we've reached a terminal state, backpropagate the actual q value and attempt
            // to select a new leaf node.
            leaf.backpropagate(q_actual);
            drop(leaf); // Drop leaf borrow so we can reassign self.leaf

            self.select_new_leaf(c_exploration);
        } else {
            // Non-terminal state found, proceed with normal expansion
            leaf.state.mask_policy(&mut est.policy_logprobs);
            let policy_logprob_est = est.policy_logprobs.log_softmax();
            leaf.expand(self.leaf.clone(), policy_logprob_est, &mut self.rng);
            // TODO: If we've expanded a leaf that has a single child (only one valid action), then
            // we can preemptively take that action and select that child as the new leaf.
            leaf.backpropagate(est.ply1_log_neg);

            drop(leaf); // Drop leaf borrow so we can reassign self.leaf
            self.select_new_leaf(c_exploration);
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
        let original_root_node = Rc::clone(&self.root);
        let root = self.root.borrow_mut();
        let child_idx = action.to_idx();

        let child = root
            .children
            .as_ref()
            .expect("apply_action called on leaf with no children")[child_idx]
            .as_ref()
            .expect("illegal action");

        // Eliminate the child's parent reference
        child.borrow_mut().parent = Weak::new();

        let child = Rc::clone(child);
        drop(root); // Drop root borrow so we can reassign self.root
        self.root = child;

        self.select_new_leaf(c_exploration);
        self.actions.push(MCTSAction {
            prev_node: original_root_node,
            action,
        });
    }

    /// Makes a move probabalistically based on the root node's policy.
    ///
    /// The temperature parameter scales the policy probabilities, with values > 1.0 making the
    /// sampled distribution more uniform and values < 1.0 making the sampled distribution favor
    /// the most lucrative moves.
    pub fn make_random_move(&mut self, temperature: f32, c_exploration: f32) {
        let policy = self.root.borrow().policy_logprobs().exp();
        let policy = policy.apply_temperature(temperature);
        let dist = WeightedIndex::new(policy).unwrap();
        let action = Action::ALL[dist.sample(&mut self.rng)];
        self.make_move(action, c_exploration);
    }

    /// If the root position is a terminal state, returns the Q value of the terminal state.
    pub fn get_root_terminal_q(&self) -> Option<f32> {
        self.root.borrow().get_terminal_q_value()
    }

    /// Returns a clone of the current leaf node's state.
    pub fn leaf_state_cloned(&self) -> State {
        self.leaf.borrow().state.clone()
    }

    /// Converts the current sequence of actions into a [GameResult] for training.
    pub fn to_result(&self) -> GameResult {
        let terminal_ply = self
            .root
            .borrow()
            .state
            .is_terminal()
            .expect("Calling to_result on non-terminal state");

        GameResult {
            metadata: self.metadata.clone(),
            samples: self
                .actions
                .iter()
                .map(
                    |MCTSAction {
                         prev_node,
                         action: _,
                     }| {
                        Sample::new_from_terminal_ply(
                            prev_node.borrow().state.clone(),
                            prev_node.borrow().policy_logprobs(),
                            terminal_ply,
                        )
                    },
                )
                .collect(),
        }
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
    children: Option<[Option<Rc<RefCell<Node>>>; Action::N_ACTIONS]>,
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
    pub fn q_value(&self) -> QValue {
        self.q_sum / (self.visit_count as f32 + Self::EPS)
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

    /// Returns the Q value of the terminal state as a log-ply1 value.
    /// This Q value is always zero as -log(ply=0 + 1) = 0.
    /// Returns None if the state is not terminal.
    fn get_terminal_q_value(&self) -> Option<QValue> {
        if self.state.is_terminal().is_none() {
            return None;
        }
        Some(0.0)
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
    fn policy_logprobs(&self) -> Policy {
        if let Some(children) = &self.children {
            let child_counts = array::from_fn(|i| {
                children[i]
                    .as_ref()
                    .map_or(0.0, |child| child.borrow().visit_count as f32)
            });
            child_counts.log_softmax()
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

    fn expand(
        &mut self,
        parent_ref: Rc<RefCell<Node>>,
        policy_logprobs: Policy,
        rng: &mut SmallRng,
    ) {
        if self.children.is_some() {
            panic!("expand called on node with children");
        }

        let legal_moves = self.state.valid_actions();
        let children: [Option<Rc<RefCell<Node>>>; Action::N_ACTIONS] = std::array::from_fn(|i| {
            let (action, can_play) = legal_moves[i];
            if can_play {
                let child_state = self.state.apply_action(action, rng);
                let child = Node::new(Rc::downgrade(&parent_ref), child_state, policy_logprobs[i]);
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
        cards::Card::*,
        state::{tests::assert_can_play_action, StateBuilder, WinCondition},
        types::NNEst,
    };

    use super::*;
    use more_asserts::assert_gt;
    use rand::SeedableRng;

    const GAME_METADATA: GameMetadata = GameMetadata {
        game_id: 0,
        player0_id: 0,
        player1_id: 1,
    };

    /// If we can buy an estate to win, we should do so.
    #[test]
    fn test_obvious_win_buy_estate() {
        let c_exploration = 2.0;
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_discard(&[(Copper, 5)])
            .with_kingdom(&[(Copper, 1), (Estate, 1)])
            .with_win_conditions(&[WinCondition::VictoryPoints(1)])
            .build(&mut rng);
        assert_eq!(state.is_terminal(), None);
        let mut mcts = MCTS::new(GAME_METADATA, state, rng);

        while mcts.root_visit_count() < 100 {
            mcts.on_received_nn_est(NNEst::new_from_ply(2, MCTS::UNIFORM_POLICY), c_exploration);
        }
        let policy = mcts.root.borrow().policy_logprobs().exp();
        assert_gt!(policy.value_for_action(Action::Buy(Estate)), 0.99);

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
            .with_discard(&[(Copper, 4), (Silver, 1)])
            .with_kingdom(&[(Copper, 5), (Silver, 5), (Gold, 5), (Province, 1)])
            .with_win_conditions(&[WinCondition::VictoryPoints(6)])
            .build(&mut rng);
        assert_eq!(state.is_terminal(), None);
        let mut mcts = MCTS::new(GAME_METADATA, state, rng);

        while mcts.root_visit_count() < 100 {
            mcts.on_received_nn_est(NNEst::new_from_ply(2, MCTS::UNIFORM_POLICY), c_exploration);
        }
        let policy = mcts.root.borrow().policy_logprobs().exp();
        assert_gt!(policy.value_for_action(Action::Buy(Gold)), 0.99);
        assert_can_play_action(&mcts.root.borrow().state, Action::Buy(Province), false);
    }

    /// If we can buy a province and gold and the win target is 7vp, we should buy the province.
    #[test]
    fn test_buy_province_over_gold() {
        let c_exploration = 2.0;
        let mut rng = SmallRng::seed_from_u64(1);
        let state = StateBuilder::new()
            .with_discard(&[(Gold, 2), (Silver, 1)])
            .with_kingdom(&[
                (Copper, 5),
                (Silver, 5),
                (Gold, 5),
                (Estate, 5),
                (Duchy, 5),
                (Province, 5),
            ])
            .with_win_conditions(&[WinCondition::VictoryPoints(7)])
            .build(&mut rng);
        assert_eq!(state.is_terminal(), None);
        let mut mcts = MCTS::new(GAME_METADATA, state, rng);

        while mcts.root_visit_count() < 100 {
            mcts.on_received_nn_est(NNEst::new_from_ply(3, MCTS::UNIFORM_POLICY), c_exploration);
        }
        let policy = mcts.root.borrow().policy_logprobs().exp();
        assert_gt!(policy.value_for_action(Action::Buy(Province)), 0.99);
    }

    #[test]
    fn test_terminal_q_value() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Non-terminal state should return None
        let non_terminal_state = StateBuilder::new()
            .with_discard(&[(Copper, 5)])
            .with_kingdom(&[(Copper, 1), (Estate, 1)])
            .with_win_conditions(&[WinCondition::VictoryPoints(100)])
            .build(&mut rng);
        let node = Node::new(Weak::new(), non_terminal_state, 0.0);
        assert_eq!(node.get_terminal_q_value(), None);

        // Terminal state should return 0.0
        let terminal_state = StateBuilder::new()
            .with_discard(&[(Copper, 5)])
            .with_kingdom(&[(Copper, 1), (Estate, 1)])
            .with_win_conditions(&[WinCondition::VictoryPoints(0)])
            .build(&mut rng);
        let node = Node::new(Weak::new(), terminal_state, 0.0);
        assert_eq!(node.get_terminal_q_value(), Some(0.0));
    }
}
