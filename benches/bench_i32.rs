#[macro_use]
extern crate criterion;

use rand::{thread_rng, Rng};
use rand_distr::Uniform;

use argmm::ArgMinMax;
use criterion::{black_box, Criterion};

fn get_array_i32() -> Vec<i32> {
    let rng = thread_rng();
    let uni = Uniform::from(0..1000);
    rng.sample_iter(uni).take(512).collect()
}

fn max_i32(c: &mut Criterion) {
    let data = get_array_i32();
    c.bench_function("simple_argmax_i32", |b| {
        b.iter(|| argmm::generic::simple_argmax(black_box(data.as_slice())))
    });
    let data = get_array_i32();
    c.bench_function("argmax_simd_i32", |b| {
        b.iter(|| black_box(data.as_slice().argmax()))
    });
}

fn min_i32(c: &mut Criterion) {
    let data = get_array_i32();
    c.bench_function("simple_argmin_i32", |b| {
        b.iter(|| argmm::generic::simple_argmin(black_box(data.as_slice())))
    });
    let data = get_array_i32();
    c.bench_function("argmin_simd_i32", |b| {
        b.iter(|| black_box(data.as_slice().argmin()))
    });
}

criterion_group!(benches, max_i32, min_i32);
criterion_main!(benches);
