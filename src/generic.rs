#[inline]
pub fn simple_argmin<T: Copy + PartialOrd>(arr: &[T]) -> usize {
    let mut low_index = 0usize;
    let mut low = arr[low_index];
    for (i, item) in arr.iter().enumerate() {
        if *item < low {
            low = *item;
            low_index = i;
        }
    }
    low_index
}

#[inline]
pub fn simple_argmax<T: Copy + PartialOrd>(arr: &[T]) -> usize {
    let mut high_index = 0usize;
    let mut high = arr[high_index];
    for (i, item) in arr.iter().enumerate() {
        if *item > high {
            high = *item;
            high_index = i;
        }
    }
    high_index
}
