#[inline]
pub fn argmin<T: Copy + PartialOrd>(arr: &[T]) -> usize {
    let mut low_index = 0usize;
    let mut low = arr[low_index];
    for i in 0..arr.len() {
        if arr[i] < low {
            low = arr[i];
            low_index = i;
        }
    }
    low_index
}

#[inline]
pub fn argmax<T: Copy + PartialOrd>(arr: &[T]) -> usize {
    let mut high_index = 0usize;
    let mut high = arr[high_index];
    for i in 0..arr.len() {
        if arr[i] > high {
            high = arr[i];
            high_index = i;
        }
    }
    high_index
}
