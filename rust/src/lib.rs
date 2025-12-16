//! Monte Carlo engine for scenario fan charts.
//!
//! Simulates n paths of an ARX(p) model over a horizon of h quarters,
//!     y_t = c + sum_i phi_i y_{t-i} + beta' x_t + e_t,
//! where the shocks e_t are drawn from historical residuals with the
//! stationary block bootstrap of Politis and Romano (1994): at each step a new
//! block starts with probability 1 / mean_block (always at the first step),
//! otherwise the next residual in the sample is used (circularly).
//!
//! Uniform random numbers are supplied by the caller, two per path and step,
//! so the Rust and the NumPy reference implementations produce identical
//! output for identical inputs (tested bit-for-bit in tests/test_scenarios_pipeline.py).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Core recursion, independent of Python (unit-tested in Rust).
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    c: f64,
    phi: &[f64],
    beta: &[f64],
    x_path: &[f64],
    y_hist: &[f64],
    resid: &[f64],
    mean_block: f64,
    uniforms: &[f64],
    n: usize,
    h: usize,
) -> Result<Vec<f64>, String> {
    let p = phi.len();
    let k = beta.len();
    let m = resid.len();
    if y_hist.len() != p {
        return Err(format!(
            "y_hist has {} values, expected p = {}",
            y_hist.len(),
            p
        ));
    }
    if x_path.len() != h * k {
        return Err(format!(
            "x_path has {} values, expected h * k = {}",
            x_path.len(),
            h * k
        ));
    }
    if uniforms.len() != n * h * 2 {
        return Err(format!(
            "uniforms has {} values, expected 2 n h = {}",
            uniforms.len(),
            2 * n * h
        ));
    }
    if m == 0 || mean_block < 1.0 {
        return Err("need at least one residual and mean_block >= 1".to_string());
    }
    let p_new = 1.0 / mean_block;
    let mut out = vec![0.0; n * h];
    let mut lags = vec![0.0; p];
    for path in 0..n {
        // lags[0] = y_{t-1}, lags[1] = y_{t-2}, ...
        for i in 0..p {
            lags[i] = y_hist[p - 1 - i];
        }
        let mut idx: usize = 0;
        for t in 0..h {
            let u_new = uniforms[(path * h + t) * 2];
            let u_pos = uniforms[(path * h + t) * 2 + 1];
            if t == 0 || u_new < p_new {
                idx = ((u_pos * m as f64) as usize).min(m - 1);
            } else {
                idx = (idx + 1) % m;
            }
            let mut y = c + resid[idx];
            for i in 0..p {
                y += phi[i] * lags[i];
            }
            for j in 0..k {
                y += beta[j] * x_path[t * k + j];
            }
            out[path * h + t] = y;
            for i in (1..p).rev() {
                lags[i] = lags[i - 1];
            }
            if p > 0 {
                lags[0] = y;
            }
        }
    }
    Ok(out)
}

/// Python entry point. NumPy arrays are borrowed without copying; the
/// result is a flat array of length n * h (row-major, path by path).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (c, phi, beta, x_path, y_hist, resid, mean_block, uniforms, n, h))]
fn simulate_arx<'py>(
    py: Python<'py>,
    c: f64,
    phi: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
    x_path: PyReadonlyArray1<'py, f64>,
    y_hist: PyReadonlyArray1<'py, f64>,
    resid: PyReadonlyArray1<'py, f64>,
    mean_block: f64,
    uniforms: PyReadonlyArray1<'py, f64>,
    n: usize,
    h: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = simulate(
        c,
        phi.as_slice()?,
        beta.as_slice()?,
        x_path.as_slice()?,
        y_hist.as_slice()?,
        resid.as_slice()?,
        mean_block,
        uniforms.as_slice()?,
        n,
        h,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn plancast_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_arx, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_ar1_with_zero_residuals() {
        // y_t = 1 + 0.5 y_{t-1}, starting from 0: 1, 1.5, 1.75
        let out = simulate(1.0, &[0.5], &[], &[], &[0.0], &[0.0], 4.0, &[0.1; 6], 1, 3).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 1.5).abs() < 1e-12);
        assert!((out[2] - 1.75).abs() < 1e-12);
    }

    #[test]
    fn exogenous_driver_enters_linearly() {
        let out = simulate(
            0.0,
            &[],
            &[2.0],
            &[1.0, -1.0],
            &[],
            &[0.0],
            1.0,
            &[0.5; 4],
            1,
            2,
        )
        .unwrap();
        assert_eq!(out, vec![2.0, -2.0]);
    }

    #[test]
    fn block_continues_when_no_restart() {
        // mean_block huge -> never restart after t = 0; start index floor(0.0 * 3) = 0
        let r = [10.0, 20.0, 30.0];
        let u = [0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0];
        let out = simulate(0.0, &[], &[], &[], &[], &r, 1e9, &u, 1, 4).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 30.0, 10.0]);
    }

    #[test]
    fn shape_errors_are_reported() {
        assert!(simulate(0.0, &[0.5], &[], &[], &[], &[1.0], 2.0, &[0.5; 2], 1, 1).is_err());
    }
}
