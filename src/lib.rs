pub mod generic;
#[cfg(target_feature = "sse")]
mod simd;
#[cfg(target_feature = "sse")]
mod task;

#[cfg(not(target_feature = "sse"))]
pub use generic::{simple_argmax, simple_argmin};
#[cfg(target_feature = "sse")]
pub use simd::{simd_f32, simd_i16, simd_i32, simd_u16, simd_u8};

pub trait ArgMinMax {
    fn argmin(&self) -> Option<usize>;
    fn argmax(&self) -> Option<usize>;
}

macro_rules! impl_argmm_f32 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin(self));
            #[cfg(target_feature = "sse")] return simd_f32::argmin_f32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax(self));
            #[cfg(target_feature = "sse")] return simd_f32::argmax_f32(self);
            }
        })*
    }
}

macro_rules! impl_argmm_i32 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin(self));
            #[cfg(target_feature = "sse")] return simd_i32::argmin_i32(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax(self));
            #[cfg(target_feature = "sse")] return simd_i32::argmax_i32(self);
            }
        })*
    }
}

macro_rules! impl_argmm_i16 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin(self));
            #[cfg(target_feature = "sse")] return simd_i16::argmin_i16(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax(self));
            #[cfg(target_feature = "sse")] return simd_i16::argmax_i16(self);
            }
        })*
    }
}

macro_rules! impl_argmm_u16 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin(self));
            #[cfg(target_feature = "sse")] return simd_u16::argmin_u16(self);
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax(self));
            #[cfg(target_feature = "sse")] return simd_u16::argmax_u16(self);
            }
        })*
    }
}

macro_rules! impl_argmm_u8 {
    ($($b:ty),*) => {
        $(impl ArgMinMax for $b {

            fn argmin(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmin(self));
            #[cfg(target_feature = "sse")] return simd_u8::argmin_u8(self) ;
            }

            fn argmax(&self) -> Option<usize> {
            #[cfg(not(target_feature = "sse"))] return Some(simple_argmax(self));
            #[cfg(target_feature = "sse")] return simd_u8::argmax_u8(self);
            }
        })*
    }
}

impl_argmm_f32!(Vec<f32>, &[f32], [f32]);
impl_argmm_i32!(Vec<i32>, &[i32], [i32]);
impl_argmm_i16!(Vec<i16>, &[i16], [i16]);
impl_argmm_u16!(Vec<u16>, &[u16], [u16]);
impl_argmm_u8!(Vec<u8>, &[u8], [u8]);
