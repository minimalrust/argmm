use argmm::ArgMinMax;

fn main() {
    let data = vec![-4i32, -8, 0, 15, 2, 1, 7, -4, 13, 20];
    let min_index = data.argmin().unwrap();
    let max_index = data.argmax().unwrap();
    assert_eq!(data[min_index], -8);
    assert_eq!(data[max_index], 20);
}
