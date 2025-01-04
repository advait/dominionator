use crate::actions::Action;

pub type Policy = [f32; Action::N_ACTIONS];

pub trait PolicyExt {
    /// Creates a policy from an iterator of logits.
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self;

    /// Returns the value of the policy for a given action.
    fn value_for_action(&self, action: Action) -> f32;

    /// Applies log softmax to the policy logits returning a policy in logprob space.
    fn log_softmax(&self) -> Policy;

    /// Applies temperature scaling to a policy in logprob space.
    /// Temperature=0.0 is argmax, temperature=1.0 is a noop.
    fn apply_temperature(&self, temperature: f32) -> Policy;

    /// Applies exp to the policy logprobs returning a policy in [0-1] space.
    fn exp(&self) -> Policy;
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

    fn log_softmax(&self) -> Policy {
        let max = self.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max.is_infinite() {
            panic!(
                "log_softmax: policy is all negative infinity, debug NN on why this is happening."
            );
        }
        // Subtract max value to avoid overflow when computing exp
        let shifted = self.map(|p| p - max);
        let log_sum_exp = shifted.iter().map(|p| p.exp()).sum::<f32>().ln();
        shifted.map(|p| p - log_sum_exp)
    }

    fn apply_temperature(&self, temperature: f32) -> Policy {
        if temperature == 0.0 {
            // Temp 0.0 is argmax
            let max = self.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ret = self.map(|p| if p == max { 0.0 } else { f32::NEG_INFINITY });
            return ret;
        }

        // Scale logprobs by temperature and reapply log_softmax
        let scaled = self.map(|p| p / temperature);
        scaled.log_softmax()
    }

    fn exp(&self) -> Policy {
        self.map(|p| p.exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prop::collection::vec;
    use proptest::prelude::*;

    const CONST_COL_WEIGHT: f32 = 1.0 / Action::N_ACTIONS as f32;

    /// Strategy for generating a policy with at least one non-zero value.
    fn policy_logprob_strategy() -> impl Strategy<Value = Policy> {
        let min = -8f32;
        let max = 10f32;
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
        .prop_map(|policy: Policy| policy.log_softmax())
    }

    proptest! {
        /// Softmax policies should sum up to one.
        #[test]
        fn softmax_sum_1(policy_logprobs in policy_logprob_strategy()) {
            assert_policy_sum_1(&policy_logprobs.exp());
        }

        /// Temperature of 1.0 should not affect the policy.
        #[test]
        fn temperature_1(policy_logprobs in policy_logprob_strategy()) {
            let policy_with_temp = policy_logprobs.apply_temperature(1.0);
            assert_policy_eq(&policy_logprobs, &policy_with_temp, 1e-5);
        }

        /// Temperature of 2.0 should change the policy.
        #[test]
        fn temperature_2(policy_logprobs in policy_logprob_strategy()) {
            let policy_with_temp = policy_logprobs.apply_temperature(2.0);
            let policy = policy_logprobs.exp();
            let policy_with_temp_prob = policy_with_temp.exp();
            assert_policy_sum_1(&policy_with_temp_prob);
            // If policy is nonuniform and there are at least two non-zero probabilities, the
            // policy with temperature should be different from the original policy
            if policy.iter().filter(|&&p| p != CONST_COL_WEIGHT && p > 0.0).count() >= 2 {
                assert_policy_ne(&policy, &policy_with_temp_prob, 1e-5);
            }
        }

        /// Temperature of 0.0 should be argmax.
        #[test]
        fn temperature_0(policy_logprobs in policy_logprob_strategy()) {
            let policy_with_temp = policy_logprobs.apply_temperature(0.0).exp();
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
            .all(|(a, b)| a.is_infinite() && b.is_infinite() || (a - b).abs() < epsilon);
        if !eq {
            panic!("policies are not equal: {:?} {:?}", p1, p2);
        }
    }

    fn assert_policy_ne(p1: &Policy, p2: &Policy, epsilon: f32) {
        let ne = p1
            .iter()
            .zip(p2.iter())
            .any(|(a, b)| a.is_infinite() ^ b.is_infinite() || (a - b).abs() > epsilon);
        if !ne {
            panic!("policies are equal: {:?} {:?}", p1, p2);
        }
    }
}
