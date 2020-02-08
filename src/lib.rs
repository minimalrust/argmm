pub mod generic;
#[cfg(target_feature = "sse")]
mod simd;
#[cfg(target_feature = "sse")]
mod task;
mod typed;

#[cfg(target_feature = "sse")]
pub use simd::{simd_f32, simd_i32};
#[cfg(not(target_feature = "sse"))]
pub use typed::{simple_argmax_f32, simple_argmax_i32, simple_argmin_f32, simple_argmin_i32};

pub trait ArgMinMax {
    fn argmin(&self) -> Option<usize>;
    fn argmax(&self) -> Option<usize>;
}

macro_rules! impl_argmm_f32 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin_f32(self));
            #[cfg(target_feature = "sse")] return simd_f32::argmin_f32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax_f32(self));
            #[cfg(target_feature = "sse")] return simd_f32::argmax_f32(self);
            }
        })*
    }
}

macro_rules! impl_argmm_i32 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin_i32(self));
            #[cfg(target_feature = "sse")] return simd_i32::argmin_i32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax_i32(self));
            #[cfg(target_feature = "sse")] return simd_i32::argmax_i32(self);
            }
        })*
    }
}

impl_argmm_f32!(Vec<f32>, &[f32], [f32]);
impl_argmm_i32!(Vec<i32>, &[i32], [i32]);
