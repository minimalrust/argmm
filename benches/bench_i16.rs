#[macro_use]
extern crate criterion;

use rand::{thread_rng, Rng};
use rand_distr::Uniform;

use argmm::ArgMinMax;
use criterion::{black_box, Criterion};

fn get_array_i16() -> Vec<i16> {
    let rng = thread_rng();
    let uni = Uniform::from(std::i16::MIN..std::i16::MAX);
    rng.sample_iter(uni).take(512).collect()
}

fn max_i16(c: &mut Criterion) {
    let data = get_array_i16();
    c.bench_function("simple_argmax_i16", |b| {
        b.iter(|| argmm::generic::simple_argmax(black_box(data.as_slice())))
    });
    let data = get_array_i16();
    c.bench_function("argmax_simd_i16", |b| {
        b.iter(|| black_box(data.as_slice().argmax()))
    });
}

fn min_i16(c: &mut Criterion) {
    let data = get_array_i16();
    c.bench_function("simple_argmin_i16", |b| {
        b.iter(|| argmm::generic::simple_argmin(black_box(data.as_slice())))
    });
    let data = get_array_i16();
    c.bench_function("argmin_simd_i16", |b| {
        b.iter(|| black_box(data.as_slice().argmin()))
    });
}

criterion_group!(benches, max_i16, min_i16);
criterion_main!(benches);
