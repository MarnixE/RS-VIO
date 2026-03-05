#[cfg(test)]
mod tests {
    use apex_solver::manifold::{LieGroup, so3::SO3};
    use nalgebra as na;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{datasets::ImuData, imu::piecewise_integration::{ImuPiecewiseIntegration, PreInt}};

    fn assert_vector_near(v1: &na::Vector3<f64>, v2: &na::Vector3<f64>, epsilon: f64) {
        assert!(
            (v1 - v2).norm() < epsilon,
            "assertion failed: vectors not approximately equal\n  left: `{:?}`,\n right: `{:?}`",
            v1, v2
        );
    }

    pub fn make_random_imu_slice(n: usize, dt: f64) -> Vec<ImuData> {
        assert!(n >= 2);
        assert!(dt > 0.0);

        let mut rng = StdRng::seed_from_u64(42);

        let t0_ns = 1e9;
        let dt_ns: i64 = (dt * t0_ns) as i64;

        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let timestamp = t0_ns as i64 + (k as i64) * dt_ns;

            // Moderate random motion; keep it small-ish so tests aren't dominated by numerical issues.
            let gyro = na::Vector3::new(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
            );

            let accel = na::Vector3::new(
                rng.gen_range(-5.0..5.0),
                rng.gen_range(-5.0..5.0),
                rng.gen_range(-5.0..5.0),
            );

            out.push(ImuData { timestamp, gyro, accel });
        }
        out
    }

    fn elapsed_time(imus: &[ImuData]) -> f64 {
        let t0 = imus.first().unwrap().timestamp as f64 * 1e-9;
        let t1 = imus.last().unwrap().timestamp as f64 * 1e-9;
        t1 - t0
    }

    fn propagate_slice(
        integrator: &mut ImuPiecewiseIntegration,
        imu_slice: &[ImuData],
        bias_a: &na::Vector3<f64>,
        bias_g: &na::Vector3<f64>,
    ) -> PreInt {
        if imu_slice.is_empty() {
            return PreInt::identity();
        }

        let mut preint = PreInt::identity();
        preint.linearized_ba = *bias_a;
        preint.linearized_bg = *bias_g;

        let mut prev_ts = imu_slice[0].timestamp as f64 * 1e-9;

        for imu in imu_slice.iter() {
            let ts = imu.timestamp as f64 * 1e-9;
            let dt = ts - prev_ts;
            prev_ts = ts;

            if dt <= 0.0 {
                continue;
            }

            preint.dt += dt;
            integrator.propagate(&mut preint, dt, &imu.accel, &imu.gyro, bias_a, bias_g);
        }

        preint
    }

    // Reference implementation following the paper’s definitions of Eq. (33),
    // using the same “recursive” accumulation but kept separate from your production code.
    fn reference_preint(
        imu_slice: &[ImuData],
        bias_a: &na::Vector3<f64>,
        bias_g: &na::Vector3<f64>,
    ) -> (SO3, na::Vector3<f64>, na::Vector3<f64>) {
        let mut prev_ts = imu_slice[0].timestamp as f64 * 1e-9;

        let mut d_r = SO3::identity();
        let mut dv = na::Vector3::zeros();
        let mut dp = na::Vector3::zeros();

        for imu in imu_slice.iter() {
            let ts = imu.timestamp as f64 * 1e-9;
            let dt = ts - prev_ts;
            prev_ts = ts;
            if dt <= 0.0 { continue; }

            let omega = imu.gyro - bias_g;
            let acc   = imu.accel - bias_a;

            let dphi = omega * dt;
            let d_r_kkp1 = SO3::from_scaled_axis(dphi);

            let dv_prev = dv;
            dv += d_r.rotation_matrix() * acc * dt;
            dp += dv_prev * dt + 0.5 * d_r.rotation_matrix() * acc * dt * dt;

            d_r = SO3::from_quaternion(d_r.quaternion() * d_r_kkp1.quaternion());
        }

        (d_r, dv, dp)
    }

    #[test]
    fn preint_matches_reference_deltas() {
        // Build a synthetic IMU slice with monotonic timestamps.
        let dt = 0.005; // 200 hz in seconds
        let imu_slice = make_random_imu_slice(/*N=*/200, /*dt=*/dt); // you implement
        let bias_a = na::Vector3::new(0.01, -0.02, 0.005);
        let bias_g = na::Vector3::new(0.001, 0.002, -0.001);

        let mut integrator = ImuPiecewiseIntegration::new(/* noise params etc */);

        let out = propagate_slice(&mut integrator, &imu_slice, &bias_a, &bias_g);
        let (d_r_ref, dv_ref, dp_ref) = reference_preint(&imu_slice, &bias_a, &bias_g);

        // Rotation: compare on tangent spacee
        let dphi = d_r_ref.inverse(None).compose(&out.delta_R, None, None).log(None); // implement log()->Vector3
        assert!(dphi.coeffs().norm() < 1e-10);
        
        assert!((out.delta_v - dv_ref).norm() < 1e-10);
        assert!((out.delta_p - dp_ref).norm() < 1e-10);

        // Time bookkeeping
        let dt_true = elapsed_time(&imu_slice);
        assert!((out.dt - dt_true).abs() < 1e-9, "PreInt.dt should be elapsed time");
    }

    #[test]
    fn jacobian_wrt_bias_matches_finite_difference() {
        let dt = 0.005; // 200 hz in seconds
        let imu_slice = make_random_imu_slice(200, dt);
        let bias_a = na::Vector3::new(0.01, -0.02, 0.005);
        let bias_g = na::Vector3::new(0.001, 0.002, -0.001);
        let eps = 1e-6;

        let mut integrator = ImuPiecewiseIntegration::new();

        let base = propagate_slice(&mut integrator, &imu_slice, &bias_a, &bias_g);

        for axis in 0..3 {
            let mut bias_a2 = bias_a;
            bias_a2[axis] += eps;

            let out2 = propagate_slice(&mut integrator, &imu_slice, &bias_a2, &bias_g);

            let fd_dv = (out2.delta_v - base.delta_v) / eps;
            let jv_ba = base.jacobian.fixed_view::<3, 3>(3, 9);
            let col = jv_ba.column(axis);

            println!("Finite difference dv wrt bias_a[{}]: {:?}", axis, fd_dv);
            assert!((fd_dv - col).norm() < 1e-4, "Jv_ba col {} mismatch", axis);

            let fd_dp = (out2.delta_p - base.delta_p) / eps;
            let jp_ba = base.jacobian.fixed_view::<3, 3>(6, 9);
            let colp = jp_ba.column(axis);
            assert!((fd_dp - colp).norm() < 1e-4, "Jp_ba col {} mismatch", axis);
        }

        for axis in 0..3 {
            let mut bias_g2 = bias_g;
            bias_g2[axis] += eps;

            let out2 = propagate_slice(&mut integrator, &imu_slice, &bias_a, &bias_g2);

            let fd_dv = (out2.delta_v - base.delta_v) / eps;
            let jv_bg = base.jacobian.fixed_view::<3, 3>(3, 12);
            let col = jv_bg.column(axis);

            println!("Finite difference dv wrt bias_g[{}]: {:?}", axis, fd_dv);
            assert!((fd_dv - col).norm() < 1e-4, "Jv_bg col {} mismatch", axis);

            let fd_dp = (out2.delta_p - base.delta_p) / eps;
            let jp_bg = base.jacobian.fixed_view::<3, 3>(6, 12);
            let colp = jp_bg.column(axis);
            assert!((fd_dp - colp).norm() < 1e-4, "Jp_bg col {} mismatch", axis);
        }
    }
}