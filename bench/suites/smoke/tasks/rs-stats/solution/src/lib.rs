//! Small statistics helpers over `f64` slices.

/// Arithmetic mean. Returns `None` for empty input.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Median: the middle value of the sorted data, or the average of the two
/// middle values for even lengths. The input does not need to be sorted.
/// Returns `None` for empty input.
pub fn median(xs: &[f64]) -> Option<f64> {
    percentile(xs, 50.0)
}

/// The p-th percentile (0.0 ≤ p ≤ 100.0) using linear interpolation between
/// closest ranks: for sorted values v[0..n], the rank is `p/100 * (n-1)`,
/// and a fractional rank interpolates linearly between the two neighbors.
/// So `percentile(xs, 0.0)` is the minimum, `percentile(xs, 100.0)` the
/// maximum, and `percentile(xs, 50.0)` equals the median.
/// Returns `None` for empty input or p outside [0, 100].
/// The input does not need to be sorted.
pub fn percentile(xs: &[f64], p: f64) -> Option<f64> {
    if xs.is_empty() || !(0.0..=100.0).contains(&p) {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = p / 100.0 * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return Some(v[lo]);
    }
    let frac = rank - lo as f64;
    Some(v[lo] + (v[hi] - v[lo]) * frac)
}
