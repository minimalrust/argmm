# Argmm

Argmin/max with SIMD support for u8, i16, u16, i32 and f32 arrays and vectors.

## Installing

Add the following to your Cargo.toml

```
argmm = "0.1.2"
```

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
    let v = vec![1u64, 3, 20, 50, 82, 9, 53, 60, 0];
    let min_index = simple_argmin(&v);
    let max_index = simple_argmax(&v);
    assert_eq!(min_index, 8);
    assert_eq!(max_index, 4);
}
```

## Benchmarks

Using a MacBook Pro (Retina, 13-inch, Early 2015) Processor 2.7 GHz Dual-Core Intel Core i5
with an array size of 512.

See `/benches/results`.

## Warning

NAN values are not supported.

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
