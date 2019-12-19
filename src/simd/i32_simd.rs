use crate::utils::split_array;
use crate::{argmax as simple_argmax, argmin as simple_argmin};
use std::arch::x86_64::*;

#[inline]
pub fn argmin_i32(arr: &[i32]) -> Option<usize> {
    match split_array(arr) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = simple_argmin(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { core_argmin(sim, rem.len()) };
            let final_test = &[rem_result, sim_result];
            let final_index = simple_argmin(final_test);
            Some(final_test[final_index].1)
        }
        (Some(rem), None) => {
            let rem_min_index = simple_argmin(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            Some(rem_result.1)
        }
        (None, Some(sim)) => {
            let sim_result = unsafe { core_argmin(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

#[inline]
unsafe fn core_argmin(sim_arr: &[i32], rem_offset: usize) -> (i32, usize) {
    // Create a SIMD array with an offset if the reminder array was used
    let offset = _mm_set1_epi32(rem_offset as i32);

    // Put the first 4 indexes into a SIMD array with
    let mut index_low = _mm_set_epi32(3, 2, 1, 0);

    index_low = _mm_add_epi32(index_low, offset);

    let mut new_index_low = index_low;

    // Setup a new SIMD array that will be used to increase the indexes
    let increment = _mm_set1_epi32(4);

    // Initialise a new SIMD array with the first 4 values
    let mut values_low = _mm_lddqu_si128(sim_arr.get_unchecked(0..4).as_ptr() as *const __m128i);

    // Iterate over the rest of the sim_arr in chunks of 4
    for i in (0..sim_arr.len()).step_by(4).skip(1) {
        // Put the values into the 4 registers
        let new_values =
            _mm_lddqu_si128(sim_arr.get_unchecked(i..i + 4).as_ptr() as *const __m128i);

        // Create a mask where any 'left' values lower than 'right' are true
        let lt_mask = _mm_cmplt_epi32(new_values, values_low);

        // Compare the new and previous low values
        values_low = _mm_min_epi16(new_values, values_low);

        // Increment the index for this chunk of values
        new_index_low = _mm_add_epi32(new_index_low, increment);

        // Instrinsic hack that improves conditional selection - blend far too slow
        index_low = _mm_or_si128(
            _mm_and_si128(new_index_low, lt_mask), // the new values that are lower
            _mm_andnot_si128(lt_mask, index_low),  // false and old_index
        );
    }

    // Find the smallest value in the array by stacking and comparing
    let highpack = _mm_unpackhi_epi32(values_low, values_low);
    let lowpack = _mm_unpacklo_epi32(values_low, values_low);

    let mut lowest = _mm_min_epi32(highpack, lowpack);

    let highestpack = _mm_unpackhi_epi32(lowest, lowest);
    let lowestpack = _mm_unpacklo_epi32(lowest, lowest);

    lowest = _mm_min_epi32(highestpack, lowestpack);

    // Create a mask for the lowest value
    let low_mask = _mm_cmpeq_epi32(lowest, values_low);

    // Replace indexes that are not the related to the lowest value with the f32::MAX
    index_low = _mm_or_si128(
        _mm_and_si128(index_low, low_mask), // the new values that are lower
        _mm_andnot_si128(low_mask, _mm_set1_epi32(std::i32::MAX)), // false and old_index
    );

    // Convert values back into arrays
    let value_array = std::mem::transmute::<__m128i, [i32; 4]>(values_low);
    let index_array = std::mem::transmute::<__m128i, [i32; 4]>(index_low);

    // Find the lowest index in the available indexes that match the lowest value
    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

#[cfg(test)]
mod tests {
    use super::{argmin_i32, simple_argmin};

    #[test]
    fn test_argmin_and_argmax_and_simd_versions_return_the_same_results_i32() {
        let data = vec![100, 5, 3, 7, 8, 9, 9, 5, 12, 5, 3, 2, 90];

        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_i32(&data).unwrap();
        let argmin_index = simple_argmin(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(data[min_index], 2);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [10, 4, 6, 9, 9, 22, 22, 4];
        let argmin_index = simple_argmin(&data);
        let argmin_simd_index = argmin_i32(&data).unwrap();
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, 1);
    }
}
