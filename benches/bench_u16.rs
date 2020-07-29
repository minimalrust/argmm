#[macro_use]
extern crate criterion;

use rand::{thread_rng, Rng};
use rand_distr::Uniform;

use argmm::ArgMinMax;
use criterion::{black_box, Criterion};

fn get_array_u16() -> Vec<u16> {
    let rng = thread_rng();
    let uni = Uniform::from(std::u16::MIN..std::u16::MAX);
    rng.sample_iter(uni).take(512).collect()
}

fn max_u16(c: &mut Criterion) {
    let data = get_array_u16();
    c.bench_function("simple_argmax_u16", |b| {
        b.iter(|| argmm::generic::simple_argmax(black_box(data.as_slice())))
    });
    let data = get_array_u16();
    c.bench_function("argmax_simd_u16", |b| {
        b.iter(|| black_box(data.as_slice().argmax()))
    });
}

fn min_u16(c: &mut Criterion) {
    let data = get_array_u16();
    c.bench_function("simple_argmin_u16", |b| {
        b.iter(|| argmm::generic::simple_argmin(black_box(data.as_slice())))
    });
    let data = get_array_u16();
    c.bench_function("argmin_simd_u16", |b| {
        b.iter(|| black_box(data.as_slice().argmin()))
    });
}

criterion_group!(benches, max_u16, min_u16);
criterion_main!(benches);
