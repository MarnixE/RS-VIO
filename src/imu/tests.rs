#[cfg(test)]
mod tests {
    use apex_solver::manifold::{LieGroup, so3::SO3};
    use nalgebra as na;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{datasets::ImuData, imu::{self, piecewise_integration::{ImuPiecewiseIntegration, PreInt}}};

    fn assert_vector_near(v1: &na::Vector3<f64>, v2: &na::Vector3<f64>, epsilon: f64) {
        assert!(
            (v1 - v2).norm() < epsilon,
            "assertion failed: vectors not approximately equal\n  left: `{:?}`,\n right: `{:?}`",
            v1, v2
        );
    }

    #[test]
    fn test_stationary_imu_zero_motion() {
        let mut integrator = ImuPiecewiseIntegration::new();

        // Simulate 100 stationary measurements at 200 Hz (0.5 seconds)
        let imu_data: Vec<ImuData> = (0..100)
            .map(|i| ImuData {
                timestamp: (i * 5000000), // 5ms = 200 Hz
                gyro: na::Vector3::zeros(),
                accel: na::Vector3::new(0.0, 0.0, 9.81), // gravity only
            })
            .collect();

        let bias_g = na::Vector3::zeros();
        let mut bias_a = na::Vector3::zeros();
        bias_a[2] = 9.81;
        let result = integrator.integrate(&imu_data, &bias_a, &bias_g);

        // Rotation should remain identity
        let q = result.dR;
        let is_identity = (q.w().abs() - 1.0).abs() < 1e-6
                    && q.x().abs() < 1e-6
                    && q.y().abs() < 1e-6
                    && q.z().abs() < 1e-6;
        assert!(is_identity, "Expected identity quaternion, got: {:?}", q.coeffs());

        // Velocity should remain zero (gravity cancels with bias)
        assert_vector_near(&result.dv, &na::Vector3::zeros(), 1e-3);

        // Position should have small gravity-induced drift
        // dp ≈ 0.5 * g * t^2 ≈ 0.5 * 9.81 * 0.25 ≈ 1.2m (check order of magnitude)
        assert!(result.dp.norm() < 2.0);
    }

    pub fn make_random_imu_slice(n: usize, dt: f64) -> Vec<ImuData> {
        assert!(n >= 2);
        assert!(dt > 0.0);

        let mut rng = StdRng::seed_from_u64(42);

        // let t0_ns: i64 = 1; // 1s
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

    // Reference implementation following the paper’s definitions of Eq. (33),
    // using the same “recursive” accumulation but kept separate from your production code.
    fn reference_preint(
        imu_slice: &[ImuData],
        bias_a: &na::Vector3<f64>,
        bias_g: &na::Vector3<f64>,
    ) -> (SO3, na::Vector3<f64>, na::Vector3<f64>) {
        let mut prev_ts = imu_slice[0].timestamp as f64 * 1e-9;

        let mut dR = SO3::identity();
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
            let dR_kkp1 = SO3::from_scaled_axis(dphi);

            let dv_prev = dv;
            dv += dR.rotation_matrix() * acc * dt;
            dp += dv_prev * dt + 0.5 * dR.rotation_matrix() * acc * dt * dt;

            dR = SO3::from_quaternion(dR.quaternion() * dR_kkp1.quaternion());
        }

        (dR, dv, dp)
    }

    #[test]
    fn preint_matches_reference_deltas() {
        // Build a synthetic IMU slice with monotonic timestamps.
        let dt = 0.005; // 200 hz in seconds
        let imu_slice = make_random_imu_slice(/*N=*/200, /*dt=*/dt); // you implement
        let bias_a = na::Vector3::new(0.01, -0.02, 0.005);
        let bias_g = na::Vector3::new(0.001, 0.002, -0.001);

        let mut integrator = ImuPiecewiseIntegration::new(/* noise params etc */);

        let out = integrator.integrate(&imu_slice, &bias_a, &bias_g);
        let (dR_ref, dv_ref, dp_ref) = reference_preint(&imu_slice, &bias_a, &bias_g);

        // Rotation: compare on tangent spacee
        let dphi = dR_ref.inverse(None).compose(&out.dR, None, None).log(None); // implement log()->Vector3
        assert!(dphi.coeffs().norm() < 1e-10);

        assert!((out.dv - dv_ref).norm() < 1e-10);
        assert!((out.dp - dp_ref).norm() < 1e-10);

        // Time bookkeeping
        let dt_true = elapsed_time(&imu_slice);
        assert!((out.dt - dt_true).abs() < 1e-9, "PreInt.dt should be elapsed time");
    }

    #[test]
    fn jacobian_wrt_accel_bias_matches_finite_difference() {
        let dt = 0.005; // 200 hz in seconds
        let imu_slice = make_random_imu_slice(200, dt);
        let bias_a = na::Vector3::new(0.01, -0.02, 0.005);
        let bias_g = na::Vector3::new(0.001, 0.002, -0.001);
        let eps = 1e-6;

        let mut integrator = ImuPiecewiseIntegration::new();

        let base = integrator.integrate(&imu_slice, &bias_a, &bias_g);

        for axis in 0..3 {
            let mut bias_a2 = bias_a;
            bias_a2[axis] += eps;

            let out2 = integrator.integrate(&imu_slice, &bias_a2, &bias_g);

            let fd_dv = (out2.dv - base.dv) / eps;
            let col = base.Jv_ba.column(axis);

            println!("Finite difference dv wrt bias_a[{}]: {:?}", axis, fd_dv);
            assert!((fd_dv - col).norm() < 1e-4, "Jv_ba col {} mismatch", axis);

            let fd_dp = (out2.dp - base.dp) / eps;
            let colp = base.Jp_ba.column(axis);
            assert!((fd_dp - colp).norm() < 1e-4, "Jp_ba col {} mismatch", axis);
        }
    }
}