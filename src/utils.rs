use std::cmp::Ordering;

/// A wrapper around f32 that implements Ord.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrdF32(pub f32);

impl Eq for OrdF32 {}

/// Panics if the f32 is NaN.
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0.is_nan() {
            panic!("NaN value in OrdF32");
        } else if other.0.is_nan() {
            panic!("NaN value in OrdF32");
        } else {
            self.partial_cmp(other).unwrap()
        }
    }
}
