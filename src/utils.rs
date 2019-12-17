#[inline]
/// Helper function to divide the array into two segments. The first being the elements which cannot
/// be ran within the SIMD function as the array size is not divisible by the SIMD vector length.
/// The second being the elements which will be vectorized.
pub fn split_array<T: Copy>(arr: &[T]) -> (Option<&[T]>, Option<&[T]>) {
    let n = arr.len();

    // Return all of the array for the non-vectorized function
    if n < 8 {
        return (Some(arr), None);
    };

    // left_array will contain remainder elements
    let (left_arr, right_arr) = arr.split_at(n % 4);

    let left_arr = match left_arr.is_empty() {
        true => None,
        false => Some(left_arr),
    };

    let right_arr = match right_arr.is_empty() {
        true => None,
        false => Some(right_arr),
    };

    (left_arr, right_arr)
}
