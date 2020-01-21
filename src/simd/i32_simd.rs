use crate::utils::split_array;
use crate::{argmax as simple_argmax, argmin as simple_argmin};
use std::arch::x86_64::*;

#[inline]
pub fn argmin_i32(arr: &[i32]) -> Option<usize> {
    match split_array(arr, 4) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = simple_argmin(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { core_argmin(sim, rem.len()) };
            let final_test = &[rem_result, sim_result];
            let final_index = simple_argmin(final_test);
            Some(final_test[final_index].1)
        }
        (Some(rem), None) => Some(simple_argmin(rem)),
        (None, Some(sim)) => {
            let sim_result = unsafe { core_argmin(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

#[inline]
unsafe fn core_argmin(sim_arr: &[i32], rem_offset: usize) -> (i32, usize) {
    let offset = _mm_set1_epi32(rem_offset as i32);
    let mut index_low = _mm_add_epi32(_mm_set_epi32(3, 2, 1, 0), offset);

    let increment = _mm_set1_epi32(4);
    let mut new_index_low = index_low;

    let mut values_low = _mm_loadu_si128(sim_arr.get_unchecked(0..4).as_ptr() as *const __m128i);

    for i in (0..sim_arr.len()).step_by(4).skip(1) {
        new_index_low = _mm_add_epi32(new_index_low, increment);

        let new_values =
            _mm_loadu_si128(sim_arr.get_unchecked(i..i + 4).as_ptr() as *const __m128i);
        let lt_mask = _mm_cmplt_epi32(new_values, values_low);

        values_low = _mm_or_si128(
            _mm_and_si128(new_values, lt_mask),
            _mm_andnot_si128(lt_mask, values_low),
        );
        index_low = _mm_or_si128(
            _mm_and_si128(new_index_low, lt_mask),
            _mm_andnot_si128(lt_mask, index_low),
        );
    }

    let highpack = _mm_unpackhi_epi32(values_low, values_low);
    let lowpack = _mm_unpacklo_epi32(values_low, values_low);

    let mut lowest = _mm_min_epi32(highpack, lowpack);

    let highestpack = _mm_unpackhi_epi32(lowest, lowest);
    let lowestpack = _mm_unpacklo_epi32(lowest, lowest);

    lowest = _mm_min_epi32(highestpack, lowestpack);

    let low_mask = _mm_cmpeq_epi32(lowest, values_low);

    index_low = _mm_or_si128(
        _mm_and_si128(index_low, low_mask),
        _mm_andnot_si128(low_mask, _mm_set1_epi32(std::i32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128i, [i32; 4]>(values_low);
    let index_array = std::mem::transmute::<__m128i, [i32; 4]>(index_low);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

#[inline]
pub fn argmax_i32(arr: &[i32]) -> Option<usize> {
    match split_array(arr, 4) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = simple_argmax(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { core_argmax(sim, rem.len()) };
            let final_test = &[rem_result, sim_result];
            let final_index = simple_argmax(final_test);
            Some(final_test[final_index].1)
        }
        (Some(rem), None) => Some(simple_argmax(rem)),
        (None, Some(sim)) => {
            let sim_result = unsafe { core_argmax(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

#[inline]
unsafe fn core_argmax(sim_arr: &[i32], rem_offset: usize) -> (i32, usize) {
    let offset = _mm_set1_epi32(rem_offset as i32);
    let mut index_high = _mm_add_epi32(_mm_set_epi32(3, 2, 1, 0), offset);
    let mut new_index_high = index_high;

    let increment = _mm_set1_epi32(4);

    let mut values_high = _mm_loadu_si128(sim_arr.get_unchecked(0..4).as_ptr() as *const __m128i);

    for i in (0..sim_arr.len()).step_by(4).skip(1) {
        new_index_high = _mm_add_epi32(new_index_high, increment);

        let new_values =
            _mm_loadu_si128(sim_arr.get_unchecked(i..i + 4).as_ptr() as *const __m128i);
        let gt_mask = _mm_cmpgt_epi32(new_values, values_high);

        values_high = _mm_or_si128(
            _mm_and_si128(new_values, gt_mask),
            _mm_andnot_si128(gt_mask, values_high),
        );
        index_high = _mm_or_si128(
            _mm_and_si128(new_index_high, gt_mask),
            _mm_andnot_si128(gt_mask, index_high),
        );
    }

    let highpack = _mm_unpackhi_epi32(values_high, values_high);
    let lowpack = _mm_unpacklo_epi32(values_high, values_high);

    let mut highest = _mm_max_epi32(highpack, lowpack);

    let highestpack = _mm_unpackhi_epi32(highest, highest);
    let lowestpack = _mm_unpacklo_epi32(highest, highest);

    highest = _mm_max_epi32(highestpack, lowestpack);

    let high_mask = _mm_cmpeq_epi32(highest, values_high);

    index_high = _mm_or_si128(
        _mm_and_si128(index_high, high_mask),
        _mm_andnot_si128(high_mask, _mm_set1_epi32(std::i32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128i, [i32; 4]>(values_high);
    let index_array = std::mem::transmute::<__m128i, [i32; 4]>(index_high);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

#[cfg(test)]
mod tests {
    use super::{argmax_i32, argmin_i32, simple_argmax, simple_argmin};
    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    fn get_array_i32() -> Vec<i32> {
        let rng = thread_rng();
        let uni = Uniform::new_inclusive(-100_000, 100_000);
        rng.sample_iter(uni).take(1025).collect()
    }

    #[test]
    fn test_using_a_random_input_returns_the_same_result() {
        let data = get_array_i32();
        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_i32(&data).unwrap();
        let max_index = argmax_i32(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = vec![100, 5, 3, 7, 8, 9, 9, 5, 12, 5, 3, 2, 909];
        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_i32(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let max_index = argmax_i32(&data).unwrap();
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [10, 4, 6, 9, 9, 22, 22, 4];
        let argmin_index = simple_argmin(&data);
        let argmin_simd_index = argmin_i32(&data).unwrap();
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, 1);

        let argmax_index = simple_argmax(&data);
        let argmax_simd_index = argmax_i32(&data).unwrap();
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, 5);
    }
}
