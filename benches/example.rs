#[macro_use]
extern crate criterion;

use argmm;
use criterion::Criterion;

fn get_array() -> Vec<f32> {
    vec![1f32; 64]
}

fn min(c: &mut Criterion) {
    let data = get_array();
    c.bench_function("argmin", |b| b.iter(|| argmm::argmin(data.as_slice())));
}

fn max(c: &mut Criterion) {
    let data = get_array();
    c.bench_function("argmax", |b| b.iter(|| argmm::argmax(data.as_slice())));
}

fn simd_min_f32(c: &mut Criterion) {
    let data = get_array();
    c.bench_function("argmin_simd f32", |b| {
        b.iter(|| argmm::argmin_simd_f32(data.as_slice()))
    });
}

fn simd_max_f32(c: &mut Criterion) {
    let data = get_array();
    c.bench_function("argmax_simd f32", |b| {
        b.iter(|| {
            argmm::argmax_simd_f32(data.as_slice());
        })
    });
}

criterion_group!(benches, simd_min_f32, min);
criterion_main!(benches);
