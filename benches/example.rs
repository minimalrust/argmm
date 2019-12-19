#![allow(unused)]

#[macro_use]
extern crate criterion;

use argmm;
use criterion::Criterion;

fn get_array() -> Vec<i32> {
    vec![1i32; 1024]
}

fn min(c: &mut Criterion) {
    let data = get_array();
    c.bench_function("argmin", |b| b.iter(|| argmm::argmin(data.as_slice())));
}

//fn max(c: &mut Criterion) {
//    let data = get_array();
//    c.bench_function("argmax", |b| b.iter(|| argmm::argmax(data.as_slice())));
//}

fn simd_min_i32(c: &mut Criterion) {
    let data = get_array();
    c.bench_function("argmin_simd i32", |b| {
        b.iter(|| argmm::argmin_i32(data.as_slice()))
    });
}
//
//fn simd_max_f32(c: &mut Criterion) {
//    let data = get_array();
//    c.bench_function("argmax_simd f32", |b| {
//        b.iter(|| {
//            argmm::argmax_f32(data.as_slice());
//        })
//    });
//}

criterion_group!(benches, simd_min_i32, min);
criterion_main!(benches);
