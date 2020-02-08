use argmm::ArgMinMax;

fn main() {
    let data = vec![-4, -8, 0, 15, 2, 1, 7, -4, 13, 20];
    let min_index = data.argmin();
    let max_index = data.argmax();
    assert_eq!(min_index, Some(1));
    assert_eq!(max_index, Some(9));
}
