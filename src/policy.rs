use std::array;

use crate::actions::Action;

pub type Policy = [f32; Action::N_ACTIONS];

pub trait PolicyExt {
    /// Creates a policy from an iterator of logits.
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self;

    /// Returns the value of the policy for a given action.
    fn value_for_action(&self, action: Action) -> f32;

    /// Applies softmax to the policy logits returning a policy in [0-1] space.
    fn softmax(&self) -> Policy;

    /// Applies temperature scaling to a policy.
    /// Expects the policy to be in [0-1] (non-log/prob) space.
    /// Temperature=0.0 is argmax, temperature=1.0 is a noop.
    fn apply_temperature(&self, temperature: f32) -> Policy;
}

impl PolicyExt for Policy {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut policy = [0.0; Action::N_ACTIONS];
        for (i, p) in iter.into_iter().enumerate() {
            policy[i] = p;
        }
        policy
    }

    fn value_for_action(&self, action: Action) -> f32 {
        self[action.to_idx()]
    }

    fn softmax(&self) -> Policy {
        let max = self.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max.is_infinite() {
            // If the policy is all negative infinity, we fall back to uniform policy.
            // This can happen if the NN dramatically underflows.
            // We panic as this is an issue that should be fixed in the NN.
            panic!("softmax: policy is all negative infinity, debug NN on why this is happening.");
        }
        let exps = self
            .iter()
            // Subtract max value to avoid overflow
            .map(|p| (p - max).exp())
            .collect::<Vec<_>>();
        let sum = exps.iter().sum::<f32>();
        array::from_fn(|i| exps[i] / sum)
    }

    fn apply_temperature(&self, temperature: f32) -> Policy {
        if temperature == 1.0 || self.iter().all(|&p| p == self[0]) {
            // Temp 1.0 or uniform policy is noop
            return self.clone();
        } else if temperature == 0.0 {
            // Temp 0.0 is argmax
            let max = self.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ret = self.map(|p| if p == max { 1.0 } else { 0.0 });
            let sum = ret.iter().sum::<f32>();
            return ret.map(|p| p / sum); // Potentially multiple argmaxes
        }

        let policy_log = self.map(|p| p.ln() / temperature);
        let policy_log_sum_exp = policy_log.map(|p| p.exp()).iter().sum::<f32>().ln();
        policy_log.map(|p| (p - policy_log_sum_exp).exp().clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prop::collection::vec;
    use proptest::prelude::*;

    const CONST_COL_WEIGHT: f32 = 1.0 / Action::N_ACTIONS as f32;

    /// Strategy for generating a policy with at least one non-zero value.
    fn policy_strategy() -> impl Strategy<Value = Policy> {
        let min = 0.0f32;
        let max = 10.0f32;
        let positive_strategy = min..max;
        let neg_inf_strategy = Just(f32::NEG_INFINITY);
        vec(
            prop_oneof![positive_strategy, neg_inf_strategy],
            Action::N_ACTIONS..Action::N_ACTIONS + 1,
        )
        .prop_filter("all neg infinity not allowed", |policy_logits| {
            !policy_logits.iter().all(|&p| p == f32::NEG_INFINITY)
        })
        .prop_map(|v| PolicyExt::from_iter(v.into_iter()))
        .prop_map(|policy: Policy| policy.softmax())
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
            let policy_with_temp = policy.apply_temperature(1.0);
            assert_policy_eq(&policy, &policy_with_temp, 1e-5);
        }

        /// Temperature of 2.0 should change the policy.
        #[test]
        fn temperature_2(policy in policy_strategy()) {
            let policy_with_temp = policy.apply_temperature(2.0);
            assert_policy_sum_1(&policy_with_temp);
            // If policy is nonuniform and there are at least two non-zero probabilities, the
            // policy with temperature should be different from the original policy
            if policy.iter().filter(|&&p| p != CONST_COL_WEIGHT && p > 0.0).count() >= 2 {
                assert_policy_ne(&policy, &policy_with_temp, 1e-5);
            }
        }

        /// Temperature of 0.0 should be argmax.
        #[test]
        fn temperature_0(policy in policy_strategy()) {
            let policy_with_temp = policy.apply_temperature(0.0);
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
