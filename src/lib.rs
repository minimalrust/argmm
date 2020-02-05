mod generic;
#[cfg(target_feature = "sse")]
mod simd;
#[cfg(target_feature = "sse")]
mod tasks;

#[cfg(not(target_feature = "sse"))]
pub use generic::{argmax as argmax_f32, argmin as argmin_f32};
pub use generic::{argmax, argmin};
#[cfg(target_feature = "sse")]
pub use simd::{
    simd_f32,
    simd_i32
};

pub trait ArgMinMax {
    fn argmin(&self) -> Option<usize>;
    fn argmax(&self) -> Option<usize>;
}

macro_rules! impl_argmm_f32 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(argmin(self));
            #[cfg(target_feature = "sse")] return simd_f32::argmin_f32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(argmax(self));
            #[cfg(target_feature = "sse")] return simd_f32::argmax_f32(self);
            }
        })*
    }
}

macro_rules! impl_argmm_i32 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(argmin(self));
            #[cfg(target_feature = "sse")] return simd_i32::argmin_i32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(argmax(self));
            #[cfg(target_feature = "sse")] return simd_i32::argmax_i32(self);
            }
        })*
    }
}

impl_argmm_f32!(Vec<f32>, &[f32], [f32]);
impl_argmm_i32!(Vec<i32>, &[i32], [i32]);
