use argmm::{argmin_i32, argmax_i32};
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn get_array_i32() -> Vec<i32> {
    let rng = thread_rng();
    let uni = Uniform::from(0..100_000);
    rng.sample_iter(uni).take(512).collect()
}

fn main() {
    let data = get_array_i32();
    let res = argmin_i32(&data).unwrap();
    let res2 = argmax_i32(&data).unwrap();
    println!("{:?} / {:?}", res, res2);
}
