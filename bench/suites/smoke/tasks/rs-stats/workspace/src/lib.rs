//! Small statistics helpers over `f64` slices.

/// Arithmetic mean. Returns `None` for empty input.
pub fn mean(xs: &[f64]) -> Option<f64> {
    todo!()
}

/// Median: the middle value of the sorted data, or the average of the two
/// middle values for even lengths. The input does not need to be sorted.
/// Returns `None` for empty input.
pub fn median(xs: &[f64]) -> Option<f64> {
    todo!()
}

/// The p-th percentile (0.0 ≤ p ≤ 100.0) using linear interpolation between
/// closest ranks: for sorted values v[0..n], the rank is `p/100 * (n-1)`,
/// and a fractional rank interpolates linearly between the two neighbors.
/// So `percentile(xs, 0.0)` is the minimum, `percentile(xs, 100.0)` the
/// maximum, and `percentile(xs, 50.0)` equals the median.
/// Returns `None` for empty input or p outside [0, 100].
/// The input does not need to be sorted.
pub fn percentile(xs: &[f64], p: f64) -> Option<f64> {
    todo!()
}
