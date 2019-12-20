#![allow(unused)]

#[macro_use]
extern crate criterion;

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Exp, Uniform};

use argmm;
use criterion::{black_box, Criterion};

fn get_array_f32() -> Vec<f32> {
    let rng = thread_rng();
    let exp = Exp::new(1.0).unwrap();
    rng.sample_iter(exp).take(1024).collect()
}

fn min_f32(c: &mut Criterion) {
    let data = get_array_f32();
    c.bench_function("argmin_f32", |b| {
        b.iter(|| argmm::argmin(black_box(data.as_slice())))
    });
}

fn simd_min_f32(c: &mut Criterion) {
    let data = get_array_f32();
    c.bench_function("argmin_simd f32", |b| {
        b.iter(|| argmm::argmin_f32(black_box(data.as_slice())))
    });
}

criterion_group!(benches, min_f32, simd_min_f32);
criterion_main!(benches);
