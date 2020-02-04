use argmm::ArgMinMax;
use rand::{thread_rng, Rng};
use rand_distr::Exp;

fn get_array_f32() -> Vec<f32> {
    let rng = thread_rng();
    let exp = Exp::new(1.0).unwrap();
    rng.sample_iter(exp).take(512).collect()
}

fn main() {
    let data = get_array_f32();
    let res = data.argmin();
    let res2 = data.argmax();
    println!("{:?} / {:?}", res, res2);
}
