use nalgebra as na;
use na::{DMatrix, DVector};

fn max_abs_vec(v: &DVector<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn max_abs_mat(m: &DMatrix<f64>) -> f64 {
    m.iter().fold(0.0_f64, |mm, &x| mm.max(x.abs()))
}

fn block_norm(m: &DMatrix<f64>, r0: usize, c0: usize, r: usize, c: usize) -> f64 {
    m.view((r0, c0), (r, c)).norm()
}

fn fmt_vec_compact(v: &DVector<f64>) -> String {
    // Compact "[a b c]" formatting (good for 3-vectors).
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 { s.push(' '); }
        s.push_str(&format!("{:+.6e}", x));
    }
    s.push(']');
    s
}

fn dump_csv(path: &std::path::Path, r: &DVector<f64>, j: Option<&DMatrix<f64>>) -> std::io::Result<()> {
    use std::io::Write;

    let mut f = std::fs::File::create(path)?;
    writeln!(f, "# residuals (len={})", r.len())?;
    for i in 0..r.len() {
        writeln!(f, "{},{}", i, r[i])?;
    }
    if let Some(J) = j {
        writeln!(f, "\n# jacobian ({}x{})", J.nrows(), J.ncols())?;
        for rr in 0..J.nrows() {
            for cc in 0..J.ncols() {
                if cc > 0 { write!(f, ",")?; }
                write!(f, "{:.17e}", J[(rr, cc)])?;
            }
            writeln!(f)?;
        }
    }
    Ok(())
}

/// Logs residual/J blocks for your 15x30 IMU factor:
/// residual layout: [rR(3), rv(3), rp(3), r_ba(3), r_bg(3)] matching Eq. 45 + Eq. 48. [file:1]
/// jac layout: state blocks [phi_i,v_i,p_i,ba_i,bg_i, phi_j,v_j,p_j,ba_j,bg_j] (each 3).
pub fn log_imu_linearization(
    tag: &str,
    residuals: &DVector<f64>,
    jacobian: Option<&DMatrix<f64>>,
    dump_dir: Option<&std::path::Path>,
) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    // Basic shape checks (don’t panic in logger; just report).
    if residuals.len() != 15 {
        log::warn!("{} residual len {} != 15", tag, residuals.len());
        return;
    }
    if let Some(J) = jacobian {
        if J.nrows() != 15 || J.ncols() != 30 {
            log::warn!("{} jac shape {}x{} != 15x30", tag, J.nrows(), J.ncols());
        }
    }

    let r_r = residuals.rows(0, 3).into_owned();
    let r_v = residuals.rows(3, 3).into_owned();
    let r_p = residuals.rows(6, 3).into_owned();
    let r_ba = residuals.rows(9, 3).into_owned();
    let r_bg = residuals.rows(12, 3).into_owned();

    log::debug!(
        "{} |r|={:.3e} max|r|={:.3e}  rR={} (|.|={:.3e}) rv={} (|.|={:.3e}) rp={} (|.|={:.3e}) rba={} (|.|={:.3e}) rbg={} (|.|={:.3e})",
        tag,
        residuals.norm(),
        max_abs_vec(residuals),
        fmt_vec_compact(&r_r), r_r.norm(),
        fmt_vec_compact(&r_v), r_v.norm(),
        fmt_vec_compact(&r_p), r_p.norm(),
        fmt_vec_compact(&r_ba), r_ba.norm(),
        fmt_vec_compact(&r_bg), r_bg.norm(),
    );

    if let Some(J) = jacobian {
        // Column block offsets (each block is 3 columns)
        const PHI_I: usize = 0;
        const V_I: usize   = 3;
        const P_I: usize   = 6;
        const BA_I: usize  = 9;
        const BG_I: usize  = 12;

        const PHI_J: usize = 15;
        const V_J: usize   = 18;
        const P_J: usize   = 21;
        const BA_J: usize  = 24;
        const BG_J: usize  = 27;

        // Row block offsets (each block is 3 rows)
        const RR: usize = 0;
        const RV: usize = 3;
        const RP: usize = 6;
        const RBA: usize = 9;
        const RBG: usize = 12;

        let jmax = max_abs_mat(J);

        // Print a compact “block norm table” (Frobenius norms).
        log::debug!(
            "{tag} J max|J|={jmax:.3e}  \
             ||J_rR_phi_i||={:.3e} ||J_rR_bg_i||={:.3e} ||J_rR_phi_j||={:.3e}  \
             ||J_rv_phi_i||={:.3e} ||J_rv_v_i||={:.3e} ||J_rv_ba_i||={:.3e} ||J_rv_bg_i||={:.3e} ||J_rv_v_j||={:.3e}  \
             ||J_rp_phi_i||={:.3e} ||J_rp_v_i||={:.3e} ||J_rp_p_i||={:.3e} ||J_rp_ba_i||={:.3e} ||J_rp_bg_i||={:.3e} ||J_rp_p_j||={:.3e}",
            block_norm(J, RR, PHI_I, 3, 3),
            block_norm(J, RR, BG_I,  3, 3),
            block_norm(J, RR, PHI_J, 3, 3),
            block_norm(J, RV, PHI_I, 3, 3),
            block_norm(J, RV, V_I,   3, 3),
            block_norm(J, RV, BA_I,  3, 3),
            block_norm(J, RV, BG_I,  3, 3),
            block_norm(J, RV, V_J,   3, 3),
            block_norm(J, RP, PHI_I, 3, 3),
            block_norm(J, RP, V_I,   3, 3),
            block_norm(J, RP, P_I,   3, 3),
            block_norm(J, RP, BA_I,  3, 3),
            block_norm(J, RP, BG_I,  3, 3),
            block_norm(J, RP, P_J,   3, 3),
        );

        // Bias random-walk rows: should look like [-I, +I] blocks (up to whitening).
        log::debug!(
            "{} bias rows norms: ||J_rba_ba_i||={:.3e} ||J_rba_ba_j||={:.3e} ||J_rbg_bg_i||={:.3e} ||J_rbg_bg_j||={:.3e}",
            tag,
            block_norm(J, RBA, BA_I, 3, 3),
            block_norm(J, RBA, BA_J, 3, 3),
            block_norm(J, RBG, BG_I, 3, 3),
            block_norm(J, RBG, BG_J, 3, 3),
        );

        // Optional dump to disk for offline inspection.
        if let Some(dir) = dump_dir {
            let _ = std::fs::create_dir_all(dir);
            let fname = format!("{}_lin.csv", tag.replace(['/', ' ', ':'], "_"));
            let path = dir.join(fname);
            if let Err(e) = dump_csv(&path, residuals, Some(J)) {
                log::warn!("{} failed to dump csv: {}", tag, e);
            } else {
                log::debug!("{} dumped linearization to {}", tag, path.display());
            }
        }
    }
}
