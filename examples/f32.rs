use argmm::ArgMinMax;

fn main() {
    let data = vec![1f32, 7., -4., 13., 20., -4., -8., 0., 15., 2.];
    let min_index = data.argmin().unwrap();
    let max_index = data.argmax().unwrap();
    assert_eq!(data[min_index], -8.);
    assert_eq!(data[max_index], 20.);
}
