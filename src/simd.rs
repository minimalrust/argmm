use super::generic::{argmax as simple_argmax, argmin as simple_argmin};
use super::tasks::split_array;
use std::arch::x86_64::*;

pub fn argmin_f32(arr: &[f32]) -> Option<usize> {
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

unsafe fn core_argmin(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    let offset = _mm_set1_ps(rem_offset as f32);
    let mut index_low = _mm_add_ps(_mm_set_ps(3.0, 2.0, 1.0, 0.0), offset);

    let increment = _mm_set1_ps(4.0);
    let mut new_index_low = index_low;

    let mut values_low = _mm_loadu_ps(sim_arr.get_unchecked(0));

    sim_arr.chunks(4).skip(1).for_each(|step| {
        new_index_low = _mm_add_ps(new_index_low, increment);

        let new_values = _mm_loadu_ps(&step[0] as *const _);
        let lt_mask = _mm_cmplt_ps(new_values, values_low);

        values_low = _mm_min_ps(new_values, values_low);
        index_low = _mm_or_ps(
            _mm_and_ps(new_index_low, lt_mask),
            _mm_andnot_ps(lt_mask, index_low),
        );
    });

    let highpack = _mm_unpackhi_ps(values_low, values_low);
    let lowpack = _mm_unpacklo_ps(values_low, values_low);

    let mut lowest = _mm_min_ps(highpack, lowpack);

    let highestpack = _mm_unpackhi_ps(lowest, lowest);
    let lowestpack = _mm_unpacklo_ps(lowest, lowest);

    lowest = _mm_min_ps(highestpack, lowestpack);

    let low_mask = _mm_cmpeq_ps(lowest, values_low);

    index_low = _mm_or_ps(
        _mm_and_ps(index_low, low_mask),
        _mm_andnot_ps(low_mask, _mm_set1_ps(std::f32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128, [f32; 4]>(values_low);
    let index_array = std::mem::transmute::<__m128, [f32; 4]>(index_low);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

pub fn argmax_f32(arr: &[f32]) -> Option<usize> {
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

unsafe fn core_argmax(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    let offset = _mm_set1_ps(rem_offset as f32);
    let mut index_high = _mm_add_ps(_mm_set_ps(3.0, 2.0, 1.0, 0.0), offset);
    let mut new_index_high = index_high;

    let increment = _mm_set1_ps(4.0);

    let mut values_high = _mm_loadu_ps(sim_arr.get_unchecked(0));

    sim_arr.chunks(4).skip(1).for_each(|step| {
        new_index_high = _mm_add_ps(new_index_high, increment);

        let new_values = _mm_loadu_ps(&step[0] as *const _);
        let gt_mask = _mm_cmpgt_ps(new_values, values_high);

        values_high = _mm_max_ps(new_values, values_high);
        index_high = _mm_or_ps(
            _mm_and_ps(new_index_high, gt_mask),
            _mm_andnot_ps(gt_mask, index_high),
        );
    });

    let highpack = _mm_unpackhi_ps(values_high, values_high);
    let lowpack = _mm_unpacklo_ps(values_high, values_high);

    let mut highest = _mm_max_ps(highpack, lowpack);

    let highestpack = _mm_unpackhi_ps(highest, highest);
    let lowestpack = _mm_unpacklo_ps(highest, highest);

    highest = _mm_max_ps(highestpack, lowestpack);

    let high_mask = _mm_cmpeq_ps(highest, values_high);

    index_high = _mm_or_ps(
        _mm_and_ps(index_high, high_mask),
        _mm_andnot_ps(high_mask, _mm_set1_ps(std::f32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128, [f32; 4]>(values_high);
    let index_array = std::mem::transmute::<__m128, [f32; 4]>(index_high);

    let max_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(max_index);
    let index = *index_array.get_unchecked(max_index);

    (value, index as usize)
}

#[cfg(test)]
mod tests {
    use super::{argmax_f32, argmin_f32, simple_argmax, simple_argmin};
    use rand::{thread_rng, Rng};
    use rand_distr::Exp;

    fn get_array_f32() -> Vec<f32> {
        let rng = thread_rng();
        let exp = Exp::new(1.0).unwrap();
        rng.sample_iter(exp).take(1025).collect()
    }

    #[test]
    fn test_using_a_random_input_returns_the_same_result() {
        let data = get_array_f32();
        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_f32(&data).unwrap();
        let max_index = argmax_f32(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
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

        let min_index = argmin_f32(&data).unwrap();
        let max_index = argmax_f32(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
        assert_eq!(data[min_index], 2840.6);
        assert_eq!(data[max_index], 3025.86);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [10., 4., 6., 9., 9., 22., 22., 4.];
        let argmin_index = simple_argmin(&data);
        let argmin_simd_index = argmin_f32(&data).unwrap();
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, 1);

        let argmax_index = simple_argmax(&data);
        let argmax_simd_index = argmax_f32(&data).unwrap();
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, 5);
    }

    #[test]
    fn test_infinity_and_nans_are_sorted_correctly() {
        let data = [std::f32::INFINITY, std::f32::NAN, std::f32::NEG_INFINITY];

        let argmin_index = simple_argmin(&data);
        let argmin_simd_index = argmin_f32(&data).unwrap();
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, 2);

        let argmax_index = simple_argmax(&data);
        let argmax_simd_index = argmax_f32(&data).unwrap();
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, 0);
    }
}
