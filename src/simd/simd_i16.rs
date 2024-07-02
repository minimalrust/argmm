use crate::generic::{simple_argmax, simple_argmin};
use crate::task::{find_final_index_max, find_final_index_min, split_array};
use std::arch::x86_64::*;

pub fn argmin_i16(arr: &[i16]) -> Option<usize> {
    let n = arr.len();
    let mut simd_func: unsafe fn(&[i16], usize) -> (i16, usize) = core_argmin;
    let mut mod_size = 4;

    if (17..std::i16::MAX).contains(&(n as i16)) {
        simd_func = core_argmin_ext;
        mod_size = 8;
    };

    match split_array(arr, mod_size) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = simple_argmin(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { simd_func(sim, rem.len()) };
            find_final_index_min(rem_result, sim_result)
        }
        (Some(rem), None) => Some(simple_argmin(rem)),
        (None, Some(sim)) => {
            let sim_result = unsafe { simd_func(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

unsafe fn core_argmin(sim_arr: &[i16], rem_offset: usize) -> (i16, usize) {
    let offset = _mm_set1_epi32(rem_offset as i32);
    let mut index_low = _mm_add_epi32(_mm_set_epi32(3, 2, 1, 0), offset);

    let increment = _mm_set1_epi32(4);
    let mut new_index_low = index_low;

    let mut values_low = _mm_set_epi32(
        sim_arr[3] as i32,
        sim_arr[2] as i32,
        sim_arr[1] as i32,
        sim_arr[0] as i32,
    );

    sim_arr.chunks_exact(4).skip(1).for_each(|step| {
        new_index_low = _mm_add_epi32(new_index_low, increment);

        let new_values = _mm_set_epi32(
            step[3] as i32,
            step[2] as i32,
            step[1] as i32,
            step[0] as i32,
        );

        let lt_mask = _mm_cmplt_epi32(new_values, values_low);

        values_low = _mm_or_si128(
            _mm_and_si128(new_values, lt_mask),
            _mm_andnot_si128(lt_mask, values_low),
        );
        index_low = _mm_or_si128(
            _mm_and_si128(new_index_low, lt_mask),
            _mm_andnot_si128(lt_mask, index_low),
        );
    });

    let highpack = _mm_unpackhi_epi32(values_low, values_low);
    let lowpack = _mm_unpacklo_epi32(values_low, values_low);
    let lowest = _mm_min_epi32(highpack, lowpack);

    let highpack = _mm_unpackhi_epi32(lowest, lowest);
    let lowpack = _mm_unpacklo_epi32(lowest, lowest);
    let lowest = _mm_min_epi32(highpack, lowpack);

    let low_mask = _mm_cmpeq_epi32(lowest, values_low);

    index_low = _mm_or_si128(
        _mm_and_si128(index_low, low_mask),
        _mm_andnot_si128(low_mask, _mm_set1_epi32(std::i32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128i, [u32; 4]>(values_low);
    let index_array = std::mem::transmute::<__m128i, [i32; 4]>(index_low);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value as i16, index as usize)
}

unsafe fn core_argmin_ext(sim_arr: &[i16], rem_offset: usize) -> (i16, usize) {
    let offset = _mm_set1_epi16(rem_offset as i16);
    let mut index_low = _mm_add_epi16(_mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0), offset);

    let increment = _mm_set1_epi16(8);
    let mut new_index_low = index_low;

    let mut values_low = _mm_set_epi16(
        sim_arr[7], sim_arr[6], sim_arr[5], sim_arr[4], sim_arr[3], sim_arr[2], sim_arr[1],
        sim_arr[0],
    );

    sim_arr.chunks_exact(8).skip(1).for_each(|step| {
        new_index_low = _mm_add_epi16(new_index_low, increment);

        let new_values = _mm_set_epi16(
            step[7], step[6], step[5], step[4], step[3], step[2], step[1], step[0],
        );

        let lt_mask = _mm_cmplt_epi16(new_values, values_low);

        values_low = _mm_or_si128(
            _mm_and_si128(new_values, lt_mask),
            _mm_andnot_si128(lt_mask, values_low),
        );
        index_low = _mm_or_si128(
            _mm_and_si128(new_index_low, lt_mask),
            _mm_andnot_si128(lt_mask, index_low),
        );
    });

    let highpack = _mm_unpackhi_epi16(values_low, values_low);
    let lowpack = _mm_unpacklo_epi16(values_low, values_low);
    let lowest = _mm_min_epi16(highpack, lowpack);

    let highpack = _mm_unpackhi_epi16(lowest, lowest);
    let lowpack = _mm_unpacklo_epi16(lowest, lowest);
    let lowest = _mm_min_epi16(highpack, lowpack);

    let highpack = _mm_unpackhi_epi16(lowest, lowest);
    let lowpack = _mm_unpacklo_epi16(lowest, lowest);
    let lowest = _mm_min_epi16(highpack, lowpack);

    let low_mask = _mm_cmpeq_epi16(lowest, values_low);

    index_low = _mm_or_si128(
        _mm_and_si128(index_low, low_mask),
        _mm_andnot_si128(low_mask, _mm_set1_epi16(std::i16::MAX)),
    );

    let value_array = std::mem::transmute::<__m128i, [i16; 8]>(values_low);
    let index_array = std::mem::transmute::<__m128i, [i16; 8]>(index_low);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

pub fn argmax_i16(arr: &[i16]) -> Option<usize> {
    let n = arr.len();
    let mut simd_func: unsafe fn(&[i16], usize) -> (i16, usize) = core_argmax;
    let mut mod_size = 4;

    if (17..=std::i16::MAX).contains(&(n as i16)) {
        simd_func = core_argmax_ext;
        mod_size = 8;
    };

    match split_array(arr, mod_size) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = simple_argmax(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { simd_func(sim, rem.len()) };
            find_final_index_max(rem_result, sim_result)
        }
        (Some(rem), None) => Some(simple_argmax(rem)),
        (None, Some(sim)) => {
            let sim_result = unsafe { simd_func(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

unsafe fn core_argmax(sim_arr: &[i16], rem_offset: usize) -> (i16, usize) {
    let offset = _mm_set1_epi32(rem_offset as i32);
    let mut index_high = _mm_add_epi32(_mm_set_epi32(3, 2, 1, 0), offset);
    let mut new_index_high = index_high;

    let increment = _mm_set1_epi32(4);

    let mut values_high = _mm_set_epi32(
        sim_arr[3] as i32,
        sim_arr[2] as i32,
        sim_arr[1] as i32,
        sim_arr[0] as i32,
    );

    sim_arr.chunks_exact(4).skip(1).for_each(|step| {
        new_index_high = _mm_add_epi32(new_index_high, increment);

        let new_values = _mm_set_epi32(
            step[3] as i32,
            step[2] as i32,
            step[1] as i32,
            step[0] as i32,
        );

        let gt_mask = _mm_cmpgt_epi32(new_values, values_high);

        values_high = _mm_or_si128(
            _mm_and_si128(new_values, gt_mask),
            _mm_andnot_si128(gt_mask, values_high),
        );
        index_high = _mm_or_si128(
            _mm_and_si128(new_index_high, gt_mask),
            _mm_andnot_si128(gt_mask, index_high),
        );
    });

    let highpack = _mm_unpackhi_epi32(values_high, values_high);
    let lowpack = _mm_unpacklo_epi32(values_high, values_high);
    let highest = _mm_max_epi32(highpack, lowpack);

    let highpack = _mm_unpackhi_epi32(highest, highest);
    let lowpack = _mm_unpacklo_epi32(highest, highest);
    let highest = _mm_max_epi32(highpack, lowpack);

    let high_mask = _mm_cmpeq_epi32(highest, values_high);

    index_high = _mm_or_si128(
        _mm_and_si128(index_high, high_mask),
        _mm_andnot_si128(high_mask, _mm_set1_epi32(std::i32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128i, [u32; 4]>(values_high);
    let index_array = std::mem::transmute::<__m128i, [i32; 4]>(index_high);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value as i16, index as usize)
}

unsafe fn core_argmax_ext(sim_arr: &[i16], rem_offset: usize) -> (i16, usize) {
    let offset = _mm_set1_epi16(rem_offset as i16);
    let mut index_high = _mm_add_epi16(_mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0), offset);
    let mut new_index_high = index_high;

    let increment = _mm_set1_epi16(8);

    let mut values_high = _mm_set_epi16(
        sim_arr[7], sim_arr[6], sim_arr[5], sim_arr[4], sim_arr[3], sim_arr[2], sim_arr[1],
        sim_arr[0],
    );

    sim_arr.chunks_exact(8).skip(1).for_each(|step| {
        new_index_high = _mm_add_epi16(new_index_high, increment);

        let new_values = _mm_set_epi16(
            step[7], step[6], step[5], step[4], step[3], step[2], step[1], step[0],
        );

        let gt_mask = _mm_cmpgt_epi16(new_values, values_high);

        values_high = _mm_or_si128(
            _mm_and_si128(new_values, gt_mask),
            _mm_andnot_si128(gt_mask, values_high),
        );
        index_high = _mm_or_si128(
            _mm_and_si128(new_index_high, gt_mask),
            _mm_andnot_si128(gt_mask, index_high),
        );
    });

    let highpack = _mm_unpackhi_epi16(values_high, values_high);
    let lowpack = _mm_unpacklo_epi16(values_high, values_high);
    let highest = _mm_max_epi16(highpack, lowpack);

    let highpack = _mm_unpackhi_epi16(highest, highest);
    let lowpack = _mm_unpacklo_epi16(highest, highest);
    let highest = _mm_max_epi16(highpack, lowpack);

    let highpack = _mm_unpackhi_epi16(highest, highest);
    let lowpack = _mm_unpacklo_epi16(highest, highest);
    let highest = _mm_max_epi16(highpack, lowpack);

    let high_mask = _mm_cmpeq_epi16(highest, values_high);

    index_high = _mm_or_si128(
        _mm_and_si128(index_high, high_mask),
        _mm_andnot_si128(high_mask, _mm_set1_epi16(std::i16::MAX)),
    );

    let value_array = std::mem::transmute::<__m128i, [i16; 8]>(values_high);
    let index_array = std::mem::transmute::<__m128i, [i16; 8]>(index_high);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

#[cfg(test)]
mod tests {
    use super::{argmax_i16, argmin_i16, simple_argmax, simple_argmin};
    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    fn get_array_i16(n: usize) -> Vec<i16> {
        let rng = thread_rng();
        let uni = Uniform::new_inclusive(std::i16::MIN, std::i16::MAX);
        rng.sample_iter(uni).take(n).collect()
    }

    #[test]
    fn test_both_versions_return_the_same_results_long_array() {
        let data = get_array_i16(1025);
        assert_eq!(data.len() % 8, 1);

        let min_index = argmin_i16(&data).unwrap();
        let max_index = argmax_i16(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
    }

    #[test]
    fn test_both_versions_return_the_same_results_short_array() {
        let data = get_array_i16(9);
        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_i16(&data).unwrap();
        let max_index = argmax_i16(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [
            10,
            std::i16::MIN,
            6,
            9,
            9,
            22,
            std::i16::MAX,
            4,
            std::i16::MAX,
        ];
        let argmin_index = simple_argmin(&data);
        let argmin_simd_index = argmin_i16(&data).unwrap();
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, 1);

        let argmax_index = simple_argmax(&data);
        let argmax_simd_index = argmax_i16(&data).unwrap();
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, 6);
    }
}
