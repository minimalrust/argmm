#[macro_use]
extern crate criterion;

use rand::{thread_rng, Rng};
use rand_distr::Exp;

use argmm::ArgMinMax;
use criterion::{black_box, Criterion};

fn get_array_f32() -> Vec<f32> {
    let rng = thread_rng();
    let exp = Exp::new(1.0).unwrap();
    rng.sample_iter(exp).take(512).collect()
}

fn max_f32(c: &mut Criterion) {
    let data = get_array_f32();
    c.bench_function("argmax_f32", |b| {
        b.iter(|| argmm::typed::simple_argmin_f32(black_box(data.as_slice())))
    });
    let data = get_array_f32();
    c.bench_function("argmax_simd f32", |b| {
        b.iter(|| black_box(data.as_slice().argmax()))
    });
}

fn min_f32(c: &mut Criterion) {
    let data = get_array_f32();
    c.bench_function("argmin_f32", |b| {
        b.iter(|| argmm::typed::simple_argmax_f32(black_box(data.as_slice())))
    });
    let data = get_array_f32();
    c.bench_function("argmin_simd f32", |b| {
        b.iter(|| black_box(data.as_slice().argmin()))
    });
}

criterion_group!(benches, max_f32, min_f32);
criterion_main!(benches);
