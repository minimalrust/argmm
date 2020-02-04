mod utils;

#[cfg(target_feature = "sse")]
mod simd;
#[cfg(target_feature = "sse")]
pub use simd::{argmax_f32, argmin_f32};
#[cfg(not(target_feature = "sse"))]
pub use self::{argmin as argmin_f32, argmax as argmax_f32};



pub trait ArgMinMax {
    fn argmin(&self) -> Option<usize>;
    fn argmax(&self) -> Option<usize>;
}

macro_rules! impl_argmm {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(argmin(self));
            #[cfg(target_feature = "sse")] return argmin_f32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(argmax(self));
            #[cfg(target_feature = "sse")] return argmax_f32(self);
            }
        })*
    }
}

impl_argmm!(Vec<f32>, &[f32], [f32]);

#[inline]
pub fn argmin<T: Copy + PartialOrd>(arr: &[T]) -> usize {
    let mut low_index = 0usize;
    let mut low = arr[low_index];
    for i in 0..arr.len() {
        if arr[i] < low {
            low = arr[i];
            low_index = i;
        }
    }
    low_index
}

#[inline]
pub fn argmax<T: Copy + PartialOrd>(arr: &[T]) -> usize {
    let mut high_index = 0usize;
    let mut high = arr[high_index];
    for i in 0..arr.len() {
        if arr[i] > high {
            high = arr[i];
            high_index = i;
        }
    }
    high_index
}
