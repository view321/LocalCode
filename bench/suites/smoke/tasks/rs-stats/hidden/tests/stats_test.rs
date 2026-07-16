use stats::{mean, median, percentile};

fn close(a: Option<f64>, b: f64) {
    let a = a.expect("expected Some");
    assert!((a - b).abs() < 1e-9, "got {a}, expected {b}");
}

#[test]
fn mean_basic_and_empty() {
    close(mean(&[1.0, 2.0, 3.0]), 2.0);
    close(mean(&[5.0]), 5.0);
    close(mean(&[-2.0, 2.0]), 0.0);
    assert!(mean(&[]).is_none());
}

#[test]
fn median_odd_even_unsorted() {
    close(median(&[3.0, 1.0, 2.0]), 2.0);
    close(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    close(median(&[7.0]), 7.0);
    close(median(&[2.0, 1.0]), 1.5);
    assert!(median(&[]).is_none());
}

#[test]
fn percentile_endpoints_are_min_and_max() {
    let xs = [9.0, 1.0, 5.0];
    close(percentile(&xs, 0.0), 1.0);
    close(percentile(&xs, 100.0), 9.0);
}

#[test]
fn percentile_interpolates_linearly() {
    let xs: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    // rank = 0.9 * 9 = 8.1 → between v[8]=9 and v[9]=10.
    close(percentile(&xs, 90.0), 9.1);
    close(percentile(&xs, 50.0), 5.5);
    close(percentile(&[1.0, 3.0], 25.0), 1.5);
}

#[test]
fn percentile_matches_median() {
    let xs = [10.0, 30.0, 20.0, 40.0];
    close(percentile(&xs, 50.0), 25.0);
    close(median(&xs), 25.0);
}

#[test]
fn percentile_rejects_bad_inputs() {
    assert!(percentile(&[], 50.0).is_none());
    assert!(percentile(&[1.0], -0.1).is_none());
    assert!(percentile(&[1.0], 100.1).is_none());
}

#[test]
fn unsorted_input_is_handled() {
    close(percentile(&[5.0, 1.0, 4.0, 2.0, 3.0], 25.0), 2.0);
}
