use argmm::ArgMinMax;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;

fn get_array_i32() -> Vec<i32> {
    let rng = thread_rng();
    let uni = Uniform::from(0..1000);
    rng.sample_iter(uni).take(512).collect()
}

fn main() {
    let data = get_array_i32();
    let res = data.argmin();
    let res2 = data.argmax();
    println!("{:?} / {:?}", res, res2);
}
