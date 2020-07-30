use std::cmp::Ordering;

#[inline]
pub(crate) fn split_array<T: Copy>(arr: &[T], lane_size: usize) -> (Option<&[T]>, Option<&[T]>) {
    let n = arr.len();

    if n < lane_size * 2 {
        return (Some(arr), None);
    };

    let (left_arr, right_arr) = arr.split_at(n % lane_size);

    match (left_arr.is_empty(), right_arr.is_empty()) {
        (true, true) => (None, None),
        (false, false) => (Some(left_arr), Some(right_arr)),
        (true, false) => (None, Some(right_arr)),
        (false, true) => (Some(left_arr), None),
    }
}

pub fn find_final_index_min<T: PartialOrd>(
    remainder_result: (T, usize),
    simd_result: (T, usize),
) -> Option<usize> {
    let result = match remainder_result.0.partial_cmp(&simd_result.0).unwrap() {
        Ordering::Less => remainder_result.1,
        Ordering::Equal => std::cmp::min(remainder_result.1, simd_result.1),
        Ordering::Greater => simd_result.1,
    };
    Some(result)
}

pub fn find_final_index_max<T: PartialOrd>(
    remainder_result: (T, usize),
    simd_result: (T, usize),
) -> Option<usize> {
    let result = match simd_result.0.partial_cmp(&remainder_result.0).unwrap() {
        Ordering::Less => remainder_result.1,
        Ordering::Equal => std::cmp::min(remainder_result.1, simd_result.1),
        Ordering::Greater => simd_result.1,
    };
    Some(result)
}
