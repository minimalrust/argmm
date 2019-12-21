use crate::{argmax_f32, argmin_f32};
use std::collections::HashMap;

#[inline]
pub(crate) fn split_array<T: Copy>(arr: &[T]) -> (Option<&[T]>, Option<&[T]>) {
    let n = arr.len();

    // Return all of the array for the non-vectorized function
    if n < 8 {
        return (Some(arr), None);
    };

    // left_array will contain remainder elements
    let (left_arr, right_arr) = arr.split_at(n % 4);

    match (left_arr.is_empty(), right_arr.is_empty()) {
        (true, true) => (None, None),
        (false, false) => (Some(left_arr), Some(right_arr)),
        (true, false) => (None, Some(right_arr)),
        (false, true) => (Some(left_arr), None),
    }
}

pub(crate) fn find_price_rises(arr: &[f32], window: usize) -> Vec<(usize, usize)> {
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
                let mut results: Vec<(usize, usize)> = hm.iter().map(|v| (*v.0, *v.1)).collect();
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
