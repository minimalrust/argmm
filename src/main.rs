use argmm::{argmax_f32, argmin_f32};
use std::collections::HashMap;

fn main() {
    let data = vec![
        2924.92, 2941.76, 2964.33, 2973.01, 2995.82, 2990.41, 2975.95, 2979.63, 2993.07, 2999.91,
        3013.77, 3014.3, 3004.04, 2984.42, 2995.11, 2976.61, 2985.03, 3005.47, 3019.56, 3003.67,
        3025.86, 3020.97, 3013.18, 2980.38, 2953.56, 2932.05, 2844.74, 2881.77, 2883.98, 2938.09,
        2918.65, 2882.7, 2926.32, 2840.6, 2847.6, 2888.68, 2923.65, 2900.51, 2924.43, 2922.95,
        2847.11, 2878.38, 2869.16, 2887.94, 2924.58, 2926.46, 2906.27, 2937.78, 2976.0, 2978.71,
        2978.43, 2979.39, 3000.93, 3009.57, 3007.39, 2997.96, 3005.7, 3006.73, 3006.79, 2992.07,
        2991.78, 2966.6, 2984.87, 2977.62, 2961.79,
    ];

    fn find_price_rises(arr: &[f32], window: usize) -> Vec<(usize, usize)> {
        assert!(window > 1, "Please apply a window size of 2 or more");

        let mut hm: HashMap<usize, usize> = HashMap::new();
        let n = arr.len();
        let mut ext = 1;
        let mut last_iteration = false;
        let mut start_index = 0usize;
        let mut end_index;
        let mut low_index;
        let mut high_index = 0usize;

        while !last_iteration {
            // Basic calculation to extend window on each iteration by factor of ext * window
            end_index = start_index + (window * ext);

            // Base case - either return end of array as end_index then continue or return hm
            if (end_index > n) & !last_iteration {
                //                println!("Outside array range");
                if ext > 1 {
                    end_index = n - 1;
                    last_iteration = true;
                } else {
                    let mut results: Vec<(usize, usize)> =
                        hm.iter().map(|v| (*v.0, *v.1)).collect();
                    results.sort_by(|a, b| a.cmp(&b));
                    return results;
                }
            }

            // The current array space we will be calculating the min or max over
            let current_window = &arr[start_index..end_index];

            // The low of the current window
            low_index = start_index + argmin_f32(&current_window).unwrap();

            if low_index == start_index {
                high_index = start_index + argmax_f32(&current_window).unwrap();
                hm.insert(low_index, high_index);
                ext += 1;
                continue;
            } else if low_index != start_index {
                if Some(&high_index) == hm.get(&start_index) {
                    start_index = high_index;
                    ext = 1;
                    continue;
                }

                start_index = low_index;
                ext = 1;
                continue;
            }
        }

        let mut results: Vec<(usize, usize)> = hm.iter().map(|v| (*v.0, *v.1)).collect();
        results.sort_by(|a, b| a.cmp(&b));
        results
    }

    dbg!(find_price_rises(&data, 2));
}
