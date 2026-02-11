use apex_solver::factors::camera::kannala_brandt;
use apex_solver::factors::Factor;


pub struct IMUFactor {
    // Define the structure of the IMU factor here
}

impl IMUFactor {
    pub fn new() -> Self {
        // Initialize the IMU factor
        IMUFactor {
            // Initialize fields
        }
    }

    pub fn from_measurements(/* inputs */) -> Self {
        // Create an IMU factor from measurements
        // This is a placeholder implementation and should be replaced with actual logic
        log::info!("[IMUFactor] Creating IMU factor from measurements (placeholder)");
        IMUFactor {
            // Initialize fields based on measurements
        }
    }

    pub fn compute_residual(&self, /* inputs */) -> Vec<f64> {
        // Implement the residual computation for the IMU factor
        // This is a placeholder implementation and should be replaced with actual logic
        log::info!("[IMUFactor] Computing residual (placeholder)");
        vec![0.0; 6] // Return a zero residual for now (3 for position, 3 for orientation)
    }

    pub fn compute_jacobian(&self, /* inputs */) -> Vec<Vec<f64>> {
        // Implement the Jacobian computation for the IMU factor
        // This is a placeholder implementation and should be replaced with actual logic
        log::info!("[IMUFactor] Computing Jacobian (placeholder)");
        vec![vec![0.0; 6]; 6] // Return a zero Jacobian for now (6x6)
    }
}


use nalgebra as na;

// Minimal SO3 wrapper assumed:
// - SO3::identity()
// - SO3::from_scaled_axis(v: Vector3)  // Exp(v)
// - so3.inverse() / so3.transpose() or so3.inverse() giving SO3
// - so3.transform_vector(&v)           // R * v
// - so3 * so3                          // composition

fn hat(w: &na::Vector3<f64>) -> na::Matrix3<f64> {
    na::Matrix3::new(
        0.0, -w[2],  w[1],
        w[2],  0.0, -w[0],
       -w[1],  w[0],  0.0,
    )
}

// Right Jacobian Jr(phi) of SO(3), closed form (Forster Eq. (8), supp Eq. (A.42)).
fn right_jacobian_so3(phi: &na::Vector3<f64>) -> na::Matrix3<f64> {
    let a = phi.norm();
    let phi_hat = hat(phi);
    let phi_hat2 = phi_hat * phi_hat;

    if a < 1e-8 {
        // Jr ≈ I - 0.5*phi^ + 1/12*(phi^)2  (series)
        na::Matrix3::identity() - 0.5 * phi_hat + (1.0 / 12.0) * phi_hat2
    } else {
        let a2 = a * a;
        let a3 = a2 * a;
        let s = a.sin();
        let c = a.cos();

        na::Matrix3::identity()
            - ((1.0 - c) / a2) * phi_hat
            + ((a - s) / a3) * phi_hat2
    }
}

pub fn propagate_noise_recursions(
    imu_slice: &[ImuMeas],           // has timestamp, gyro: Vec3, accel: Vec3
    b_g_i: na::Vector3<f64>,
    b_a_i: na::Vector3<f64>,
) -> (SO3, na::Vector3<f64>, na::Vector3<f64>, na::Vector3<f64>) {

    let mut prev_ts = imu_slice.first().map_or(0.0, |m| m.timestamp as f64 * 1e-9);

    let mut delta_r_ik = SO3::identity(); // ΔR~_{i,k}
    let mut eta_phi = na::Vector3::zeros(); // δϕ_{i,k}
    let mut eta_v   = na::Vector3::zeros(); // δv_{i,k}
    let mut eta_p   = na::Vector3::zeros(); // δp_{i,k}

    for imu in imu_slice.iter() {
        let ts = imu.timestamp as f64 * 1e-9;
        let dt = ts - prev_ts;
        prev_ts = ts;

        // k = j-1 in your notation (current IMU sample used to propagate one step)
        let omega_unbiased = imu.gyro - b_g_i;
        let acc_unbiased   = imu.accel - b_a_i;

        // ΔR~_{k,k+1}
        let dphi = omega_unbiased * dt;                  // in so(3) coords
        let delta_r_kkp1 = SO3::from_scaled_axis(dphi);  // Exp(dphi)
        let jr = right_jacobian_so3(&dphi);              // Jr(dphi)

        // Cache ΔR~_{i,k} BEFORE updating it -> this is ΔR~_{i,j-1} in Eq. (60)/(61)
        let delta_r_i_k = delta_r_ik;

        // ---- Eq. (59): rotation noise recursion ----
        // δϕ_{i,k+1} ≈ ΔR~_{k,k+1}ᵀ δϕ_{i,k} + Jr_k η^g_{d,k} dt
        // (η^g_{d,k} is a random variable; here we only keep the linear form. If you are not sampling noise, omit the +noise term.)
        eta_phi = delta_r_kkp1.inverse().transform_vector(&eta_phi);
        // If you *sample* gyro noise n_g ~ N(0, σ_g^2 I): eta_phi += jr * n_g * dt;

        // ---- Eq. (60): velocity noise recursion ----
        // δv_{i,k+1} ≈ δv_{i,k} - ΔR~_{i,k}(a~_k - b^a_i)^ ∧ δϕ_{i,k} dt + ΔR~_{i,k} η^a_{d,k} dt
        let a_hat = hat(&acc_unbiased);
        eta_v = eta_v - (delta_r_i_k.transform_vector(&(a_hat * eta_phi))) * dt;
        // If sampling accel noise n_a: eta_v += delta_r_i_k.transform_vector(&(n_a * dt));

        // ---- Eq. (61): position noise recursion ----
        // δp_{i,k+1} ≈ δp_{i,k} + δv_{i,k} dt - 1/2 ΔR~_{i,k}(a~_k - b^a_i)^ ∧ δϕ_{i,k} dt^2 + 1/2 ΔR~_{i,k} η^a_{d,k} dt^2
        eta_p = eta_p
            + eta_v * dt
            - 0.5 * (delta_r_i_k.transform_vector(&(a_hat * eta_phi))) * (dt * dt);
        // If sampling accel noise n_a: eta_p += 0.5 * delta_r_i_k.transform_vector(&(n_a * dt * dt));

        // Finally update ΔR~_{i,k+1} = ΔR~_{i,k} * ΔR~_{k,k+1}
        delta_r_ik = delta_r_ik * delta_r_kkp1;
    }

    (delta_r_ik, eta_phi, eta_v, eta_p)
}


