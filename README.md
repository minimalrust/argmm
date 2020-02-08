# Argmm

Argmin/max with SIMD support for i32 and f32 arrays and vectors.

## Getting started

You can use the extention trait which will take advantage of SIMD if available
```rust
use argmm::ArgMinMax;

fn main() {
    let v = vec![1., 3., -20., 50., -82., 9., -53., 60., 0.];
    let min_index = v.argmin();
    let max_index = v.argmax();
    assert_eq!(min_index, Some(4));
    assert_eq!(max_index, Some(7));
}
```

Alternatively, the generic function can be used if you require non-SIMD support for other types

```rust
use argmm::generic::{simple_argmin, simple_argmax};

fn main() {
    let v = vec![1u8, 3, 20, 50, 82, 9, 53, 60, 0];
    let min_index = simple_argmin(&v);
    let max_index = simple_argmax(&v);
    assert_eq!(min_index, 8);
    assert_eq!(max_index, 4);
}
```

## Benchmarks

MacBook Pro (Retina, 13-inch, Early 2015) Processor 2.7 GHz Dual-Core Intel Core i5
with an array size of 512.

|Type|Function|Time|Factor|
|---|---|---|---|
|f32|simple_argmin|536.91 ns|1|
|f32|argmin_simd  |157.30 ns|3.14|

|Type|Function|Time|Factor|
|---|---|---|---|
|f32|simple_argmax|531.54 ns|1|
|f32|argmax_simd  |157.08 ns|3.38|

|Type|Function|Time|Factor|
|---|---|---|---|
|i32|simple_argmin|343.02 ns|1|
|i32|argmin_simd  |203.30 ns|1.68|

|Type|Function|Time|Factor|
|---|---|---|---|
|i32|simple_argmax|350.06 ns|1|
|i32|argmax_simd  |191.45 ns|1.82|

## Warning

NAN values are not supported.

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
