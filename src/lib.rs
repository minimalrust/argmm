use std::arch::x86_64::*;

#[inline]
/// Locates the index of the smallest value in the array
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
/// Locates the index of the largest value in the array
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

/// Locate the index of the smallest `f32` in the array using SIMD.
///
/// # Vectorization conditions
/// The function compares 2 lanes of 4 elements on each loop. Any elements that don't fit into the
/// SIMD lane will have the remaining elements ran using a non-vectorized `argmin` function.
/// The smallest of the two results are then returned.
///
/// # Examples
/// `len` >= 8 - At worst 3 elements ran using a non-vectorized `argmin` function.
/// ```rust
/// # use argmm::argmin_simd_f32;
/// # fn main() {
/// let arr = [6., 7., 4., 2., 5., 9., 3., 9., 2., 3., 8.];
/// // [6., 7., 4.] will be run using argmin internally
/// // [2., 5., 9., 3., 9., 2., 3., 8.] using argmin_simd version
/// // [4.] and [.2] will be compared and the index of the lowest returned
/// let idx = argmin_simd_f32(&arr).unwrap();
/// assert_eq!(idx, 3);
/// assert_eq!(arr[idx], 2.);
/// # }
/// ```
/// `len` < 8 - All elements ran using a non-vectorized `argmin` function.
/// ```rust
/// # use argmm::argmin_simd_f32;
/// # fn main() {
/// let arr = [5., 5., 9., 7., 9., 1., 3., 1.];
/// // All elements will be run using argmin internally
/// let idx = argmin_simd_f32(&arr).unwrap();
/// assert_eq!(idx, 5);
/// assert_eq!(arr[idx], 1.);
/// # }
/// ```
#[inline]
pub fn argmin_simd_f32(arr: &[f32]) -> Option<usize> {
    match split_array(arr) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = argmin(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { min_simd_f32(sim, rem.len()) };
            let final_test = &[rem_result, sim_result];
            let final_index = argmin(final_test);
            Some(final_test[final_index].1)
        }
        (Some(rem), None) => {
            let rem_min_index = argmin(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            Some(rem_result.1)
        }
        (None, Some(sim)) => {
            let sim_result = unsafe { min_simd_f32(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

/// Unsafe SIMD function to locate smallest `f32`. The function should only be called via a safe API
/// which passes a correctly sized array as the first argument and any offset to the result as the
/// second.
#[inline]
unsafe fn min_simd_f32(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    // Create a SIMD array with an offset if the reminder array was used
    let offset = _mm_set1_ps(rem_offset as f32);

    // Put the first 4 indexes into a SIMD array with
    let mut index_low = _mm_set_ps(3.0, 2.0, 1.0, 0.0);

    index_low = _mm_add_ps(index_low, offset);

    let mut new_index_low = index_low;

    // Setup a new SIMD array that will be used to increase the indexes
    let increment = _mm_set1_ps(4.0);

    // Initialise a new SIMD array with the first 4 values
    let mut values_low = _mm_loadu_ps(sim_arr.get_unchecked(0));

    // Iterate over the rest of the sim_arr in chunks of 4
    for i in (0..sim_arr.len()).step_by(4).skip(1) {
        // Put the values into the 4 registers
        let new_values = _mm_loadu_ps(sim_arr.get_unchecked(i));

        // Create a mask where any 'left' values lower than 'right' are true
        let lt_mask = _mm_cmplt_ps(new_values, values_low);

        // Compare the new and previous low values
        values_low = _mm_min_ps(new_values, values_low);

        // Increment the index for this chunk of values
        new_index_low = _mm_add_ps(new_index_low, increment);

        // Instrinsic hack that improves conditional selection - blend far too slow
        index_low = _mm_or_ps(
            _mm_and_ps(new_index_low, lt_mask), // the new values that are lower
            _mm_andnot_ps(lt_mask, index_low),  // false and old_index
        );
    }

    // Find the smallest value in the array by stacking and comparing
    let highpack = _mm_unpackhi_ps(values_low, values_low);
    let lopack = _mm_unpacklo_ps(values_low, values_low);

    let mut lowest = _mm_min_ps(highpack, lopack);

    let highestpack = _mm_unpackhi_ps(lowest, lowest);
    let lowestpack = _mm_unpacklo_ps(lowest, lowest);
    lowest = _mm_min_ps(highestpack, lowestpack);

    // Create a mask for the lowest value
    let low_mask = _mm_cmpeq_ps(lowest, values_low);

    // Replace indexes that are not the related to the lowest value with the f32::MAX
    index_low = _mm_or_ps(
        _mm_and_ps(index_low, low_mask), // the new values that are lower
        _mm_andnot_ps(low_mask, _mm_set1_ps(std::f32::MAX)),  // false and old_index
    );

    // Convert values back into arrays
    let value_array = std::mem::transmute::<__m128, [f32; 4]>(values_low);
    let index_array = std::mem::transmute::<__m128, [f32; 4]>(index_low);

    // Find the lowest index in the available indexes that match the lowest value
    let min_index = argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

/// Locate the index of the largest `f32` in the array using SIMD.
///
/// # Vectorization conditions
/// The function compares 2 lanes of 4 elements on each loop. Any elements that don't fit into the
/// SIMD lane will have the remaining elements ran using a non-vectorized `argmax` function.
/// The largest of the two results are then returned.
///
/// # Examples
/// `len` >= 8 - At worst 3 elements ran using a non-vectorized `argmax` function.
/// ```rust
/// # use argmm::argmax_simd_f32;
/// # fn main() {
/// let arr = [6., 7., 4., 2., 5., 9., 3., 9., 2., 3., 8.];
/// // [6., 7., 4.] will be run using argmax internally
/// // [2., 5., 9., 3., 9., 2., 3., 8.] using argmax_simd version
/// // [4.] and [.2] will be compared and the index of the largest returned
/// let idx = argmax_simd_f32(&arr).unwrap();
/// assert_eq!(idx, 5);
/// assert_eq!(arr[idx], 9.);
/// # }
/// ```
/// `len` < 8 - All elements ran using a non-vectorized `argmax` function.
/// ```rust
/// # use argmm::argmax_simd_f32;
/// # use argmm::argmax;
/// # fn main() {
/// let arr = [5., 5., 9., 7., 9., 1., 3., 4.];
/// // All elements will be run using argmax internally
/// let idx = argmax_simd_f32(&arr).unwrap();
/// let idx2 = argmax(&arr);
/// assert_eq!(idx, 2);
/// assert_eq!(idx2, 2);
/// assert_eq!(arr[idx], 9.);
/// # }
/// ```
#[inline]
pub fn argmax_simd_f32(arr: &[f32]) -> Option<usize> {
    match split_array(arr) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = argmax(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { max_simd_f32(sim, rem.len()) };
            let final_test = &[rem_result, sim_result];
            let final_index = argmax(final_test);
            Some(final_test[final_index].1)
        }
        (Some(rem), None) => {
            let rem_min_index = argmax(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            Some(rem_result.1)
        }
        (None, Some(sim)) => {
            let sim_result = unsafe { max_simd_f32(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

/// Unsafe SIMD function to locate largest `f32`. The function should only be called via a safe API
/// which passes a correctly sized array as the first argument and any offset to the result as the
/// second.
#[inline]
unsafe fn max_simd_f32(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    // Create a SIMD array with an offset if the reminder array was used
    let offset = _mm_set1_ps(rem_offset as f32);

    // Put the first 4 indexes into a SIMD array with
    let mut index_high = _mm_set_ps(3.0, 2.0, 1.0, 0.0);

    index_high = _mm_add_ps(index_high, offset);

    let mut new_index_high = index_high;

    // Setup a new SIMD array that will be used to increase the indexes
    let increment = _mm_set1_ps(4.0);

    // Initialise a new SIMD array with the first 4 values
    let mut values_high = _mm_loadu_ps(sim_arr.get_unchecked(0));

    // Iterate over the rest of the sim_arr in chunks of 4
    for i in (0..sim_arr.len()).step_by(4).skip(1) {
        // Put the values into the 4 registers
        let new_values = _mm_loadu_ps(sim_arr.get_unchecked(i));

        // Create a mask where any 'left' values lower than 'right' are true
        let gt_mask = _mm_cmpgt_ps(new_values, values_high);

        // Compare the new and previous low values
        values_high = _mm_max_ps(new_values, values_high);

        // Increment the index for this chunk of values
        new_index_high = _mm_add_ps(new_index_high, increment);

        // Instrinsic hack that improves conditional selection - blend far too slow
        index_high = _mm_or_ps(
            _mm_and_ps(new_index_high, gt_mask), // the new values that are lower
            _mm_andnot_ps(gt_mask, index_high),  // false and old_index
        );
    }

    // Find the largest value in the array by stacking and comparing
    let highpack = _mm_unpackhi_ps(values_high, values_high);
    let lopack = _mm_unpacklo_ps(values_high, values_high);

    let mut highest = _mm_max_ps(highpack, lopack);

    let highestpack = _mm_unpackhi_ps(highpack, highpack);
    let lowestpack = _mm_unpacklo_ps(lopack, lopack);
    highest = _mm_max_ps(highestpack, lowestpack);

    // Create a mask for the highest value
    let high_mask = _mm_cmpeq_ps(highest, values_high);

    // Replace indexes that are not the related to the highest value with the f32::MAX
    index_high = _mm_or_ps(
        _mm_and_ps(index_high, high_mask), // the new values that are lower
        _mm_andnot_ps(high_mask, _mm_set1_ps(std::f32::MAX)),  // false and old_index
    );

    // Convert values back into arrays
    let value_array = std::mem::transmute::<__m128, [f32; 4]>(values_high);
    let index_array = std::mem::transmute::<__m128, [f32; 4]>(index_high);

    // Find the lowest index in the available indexes that match the highest value
    let max_index = argmin(&index_array);
    let value = *value_array.get_unchecked(max_index);
    let index = *index_array.get_unchecked(max_index);

    (value, index as usize)
}

#[cfg(test)]
mod tests {
    use super::{argmax, argmax_simd_f32, argmin, argmin_simd_f32};
    #[test]
    fn basic_argx_test() {
        let arr = [10, 2, 10, 32, 47, 3, 22];
        assert_eq!(argmin(&arr), 1);
        assert_eq!(argmax(&arr), 4);

        let arr = [9.1, 19.9, 5.2];
        assert_eq!(argmin(&arr), 2);
        assert_eq!(argmax(&arr), 1);

        let arr = [std::f32::INFINITY, std::f32::NAN, std::f32::NEG_INFINITY];
        assert_eq!(argmin(&arr), 2);
        assert_eq!(argmax(&arr), 0);
    }

    #[test]
    fn test_argx_and_simdx_get_same_result() {
        let data = vec![
            2924.92, 2941.76, 2964.33, 2973.01, 2995.82, 2990.41, 2975.95, 2979.63, 2993.07,
            2999.91, 3013.77, 3014.3, 3004.04, 2984.42, 2995.11, 2976.61, 2985.03, 3005.47,
            3019.56, 3003.67, 3025.86, 3020.97, 3013.18, 2980.38, 2953.56, 2932.05, 2844.74,
            2881.77, 2883.98, 2938.09, 2918.65, 2882.7, 2926.32, 2840.6, 2847.6, 2888.68, 2923.65,
            2900.51, 2924.43, 2922.95, 2847.11, 2878.38, 2869.16, 2887.94, 2924.58, 2926.46,
            2906.27, 2937.78, 2976.0, 2978.71, 2978.43, 2979.39, 3000.93, 3009.57, 3007.39,
            2997.96, 3005.7, 3006.73, 3006.79, 2992.07, 2991.78, 2966.6, 2984.87, 2977.62, 2961.79,
        ];

        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_simd_f32(&data).unwrap(); // 33
        let max_index = argmax_simd_f32(&data).unwrap(); // 20
        let argmin_index = argmin(&data);
        let argmax_index = argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
        assert_eq!(data[min_index], 2840.6);
        assert_eq!(data[max_index], 3025.86);
    }
}
