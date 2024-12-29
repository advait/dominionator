use crate::actions::N_ACTIONS;

pub type Ply = u8;

pub type QValue = f32;

pub type Policy = [f32; N_ACTIONS];

pub fn policy_from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Policy {
    let mut policy = [0.0; N_ACTIONS];
    for (i, p) in iter.into_iter().enumerate() {
        policy[i] = p;
    }
    policy
}
