use crate::{datasets::{ImuData, config}, imu};
use imageproc::noise;
use nalgebra as na;
use apex_solver::manifold::{LieGroup, so3::SO3};

#[allow(non_snake_case, dead_code)]
pub struct PreInt {
    dR: SO3,
    dv: na::Vector3<f64>,
    dp: na::Vector3<f64>,
    cov: na::SMatrix<f64, 9, 9>,
    dt: f64,
    bias_g: na::Vector3<f64>,
    bias_a: na::Vector3<f64>,
    Jr_bg: na::Matrix3<f64>,
    Jv_bg: na::Matrix3<f64>,
    Jv_ba: na::Matrix3<f64>,
    Jp_bg: na::Matrix3<f64>,
    Jp_ba: na::Matrix3<f64>,
}

pub struct ImuPiecewiseIntegration {
    // Fields for piecewise integration
    // prev_timestamp: f64,
    T_BS: na::Matrix4<f64>,
    accel_noise_density: f64,
    gyro_noise_density: f64,
    accel_random_walk: na::Vector3<f64>,
    gyro_random_walk: na::Vector3<f64>,
    // preintegrated_noise: na::SVector<f64, 9>,
}

#[allow(non_snake_case)]
impl ImuPiecewiseIntegration {
    pub fn new() -> Self {
        // Initialize the piecewise integration
        ImuPiecewiseIntegration {
            // Initialize fields
            // prev_timestamp: 0.0,
            T_BS: nalgebra::Matrix4::identity(),
            accel_noise_density: 0.0,
            gyro_noise_density: 0.0,
            accel_random_walk: na::Vector3::zeros(),
            gyro_random_walk: na::Vector3::zeros(),
            // preintegrated_noise: na::SVector::<f64, 9>::zeros(),
        }
    }

    pub fn from_config(config: config::ImuConfig) -> Self {
        ImuPiecewiseIntegration {
            // prev_timestamp: 0.0,
            T_BS: nalgebra::Matrix4::from_row_slice(&config.T_BS.data),
            accel_noise_density: config.accel_noise_density,
            gyro_noise_density: config.gyro_noise_density,
            accel_random_walk: na::Vector3::from_element(config.accel_random_walk),
            gyro_random_walk: na::Vector3::from_element(config.gyro_random_walk),
            // preintegrated_noise: na::SVector::<f64, 9>::from_element(1.0),
        }
    }

    #[allow(non_snake_case)]
    pub fn integrate(&mut self, imu_slice: &[ImuData]) -> PreInt {
        let mut prev_ts = imu_slice.first().map_or(0.0, |imu| imu.timestamp as f64 * 1e-9); // Initialize prev_timestamp to the timestamp of the first IMU measurement
        
        let mut dR_ik = SO3::identity(); // Initialize delta_R_j_i to identity
        let mut dv_ik = na::Vector3::zeros(); // Initialize delta_v_j_i to zero
        let mut dp_ik = na::Vector3::zeros(); // Initialize delta_p_j_i to zero

        let mut Jr_bg = na::Matrix3::zeros();
        let mut Jv_bg = na::Matrix3::zeros();
        let mut Jv_ba = na::Matrix3::zeros();
        let mut Jp_bg = na::Matrix3::zeros();
        let mut Jp_ba = na::Matrix3::zeros();

        // let mut dt = 0.0; // Initialize dt to zero
        // let mut dR_kkp1 = SO3::identity(); // Initialize delta_R_k_k+1 to identity
        // let mut dR_ikm1 = SO3::identity(); // Initialize delta_R_i_k+1 to identity

        // let mut eta_phi = na::Vector3::zeros();
        // let mut eta_v   = na::Vector3::zeros();
        // let mut eta_p   = na::Vector3::zeros();
        let mut cov_ik = na::SMatrix::<f64, 9, 9>::zeros(); 

        let bias_g = self.gyro_random_walk; // Placeholder for gyro bias
        let bias_a = self.accel_random_walk; // Placeholder for accel bias

        for (i, imu) in imu_slice.iter().enumerate() {
            let ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            let dt = ts - prev_ts;
            prev_ts = ts;

            let omega_unbiased = imu.gyro - bias_g;
            let acc_unbiased = imu.accel - bias_a;

            let dphi = omega_unbiased  * dt; // Angular increment
            let dR_kkp1 = SO3::from_scaled_axis(dphi);
            let J_r = self.right_jacobian_so3(&dphi);

            let dv_ikm1 = dv_ik.clone(); // Cache delta_v_i_k before updating it for the next iteration
            dv_ik += dR_ik.rotation_matrix() * acc_unbiased * dt;
            dp_ik += dv_ikm1 * dt + dR_ik.rotation_matrix() * acc_unbiased * dt * dt * 0.5;

            // let dR_ikm1 = dR_ik.clone(); // Cache delta_R_i_k before updating it for the next iteration
            
            // let dR_kp1k = dR_kkp1.rotation_matrix().try_inverse().unwrap();

            let A = self.construct_A(&dR_kkp1, &dR_ik, &acc_unbiased, dt);
            let B = self.construct_B(&J_r, &dR_ik, dt);
            
            let cov_eta = self.cov_eta(dt); // Assuming isotropic noise for simplicity
            cov_ik = A * cov_ik * A.transpose() + B * cov_eta * B.transpose(); // Propagate covariance

            Jp_ba += Jv_ba * dt - 0.5 * dR_ik.rotation_matrix() * (dt*dt);
            Jv_ba += -dR_ik.rotation_matrix() * dt;

            let a_hat = skew_symmetric(&acc_unbiased);
            Jp_bg += Jv_bg * dt - 0.5 * dR_ik.rotation_matrix() * a_hat * Jr_bg * (dt*dt);
            Jv_bg += -dR_ik.rotation_matrix() * a_hat * Jr_bg * dt;

            let R_kkp1 = dR_kkp1.rotation_matrix();
            Jr_bg = R_kkp1.transpose() * Jr_bg - J_r * dt;

            // let eta_phi_ikm1 = eta_phi.clone(); // Cache eta_phi_{i,k} before updating it for the next iteration
            // eta_phi = dR_kp1k.transpose() * eta_phi + J_r * noise_g * dt;

            // 
            
            // let eta_v_ikm1 = eta_v.clone();
            // eta_v = eta_v - dR_ikm1.rotation_matrix() * a_hat * eta_phi * dt 
            //     + dR_ikm1.rotation_matrix() * noise_a * dt;
            
            // eta_p = eta_p + eta_v_ikm1 * dt - 0.5 * dR_ikm1.rotation_matrix() * a_hat * eta_phi_ikm1 * (dt * dt) 
            //     + 0.5 * dR_ikm1.rotation_matrix() * (noise_a * dt * dt);

            dR_ik = SO3::from_quaternion(dR_ik.quaternion() * dR_kkp1.quaternion());
        }
        
        // log::debug!("Delta rotation (delta_q): {:?}", dR_ik.quaternion());
        // log::debug!("Delta velocity (delta_v): {:?}", dv_ik);
        // log::debug!("Delta position (delta_p): {:?}", dp_ik);

        PreInt {
            dR: dR_ik,
            dv: dv_ik,
            dp: dp_ik,
            cov: cov_ik,
            dt: prev_ts, // Total time from the first to the last IMU measurement
            bias_g,
            bias_a,
            Jr_bg: Jr_bg, // Placeholder for Jacobian w.r.t gyro bias
            Jv_bg: Jv_bg, // Placeholder for Jacobian w.r.t gyro bias
            Jv_ba: Jv_ba, // Placeholder for Jacobian w.r.t accel bias
            Jp_bg: Jp_bg, // Placeholder for Jacobian w.r.t gyro bias
            Jp_ba: Jp_ba, // Placeholder for Jacobian w.r.t accel bias
        }
    }

    // pub fn compute_bias(&self, imu_slice: &[ImuData]) -> [f64; 6] {
    //     // Implement bias computation logic here
    //     // This is a placeholder implementation and should be replaced with actual logic
    //     log::info!("[ImuMidpointIntegration] Computing IMU bias (placeholder)");
    //     [0.0; 6] // Return zero bias for now
    // }

    fn right_jacobian_so3(&self, phi: &na::Vector3<f64>) -> na::Matrix3<f64> {
        let a = phi.norm();
        let phi_hat = skew_symmetric(phi);
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

    fn construct_A(&self, dR_kkp1: &SO3, dR_ik: &SO3, acc_unbiased: &na::Vector3<f64>, dt: f64) -> na::SMatrix<f64, 9, 9> {
        let I3 = na::Matrix3::<f64>::identity();

        let R_ik = dR_ik.rotation_matrix();               // ΔR_{i,k}
        let R_kp1k_T = dR_kkp1.rotation_matrix().transpose(); // (ΔR_{k,k+1})^T

        let a_hat = skew_symmetric(&acc_unbiased);

        let A_phiphi = R_kp1k_T;
        let A_vphi = -R_ik * a_hat * dt;
        let A_pphi = -0.5 * R_ik * a_hat * (dt * dt);
        let A_pv = I3 * dt;
        
        // let A: na::SMatrix<f64, 9, 9> = 
        na::stack![
            A_phiphi, 0, 0;
            A_vphi, I3, 0;
            A_pphi, A_pv, I3
        ]
        // A
    }

    fn construct_B(&self, Jr: &na::Matrix3<f64>, dR_ik: &SO3, dt: f64) -> na::SMatrix<f64, 9, 6> {
        let R_ik = dR_ik.rotation_matrix();

        let B_phig = Jr * dt;
        let B_va   = R_ik * dt;
        let B_pa   = 0.5 * R_ik * (dt * dt);

        na::stack![
            B_phig, 0;
            0, B_va;
            0, B_pa
        ]
    }

    fn cov_eta(&self, dt: f64) -> na::SMatrix<f64, 6, 6> {
        let I3 = na::Matrix3::<f64>::identity();
        let sigma_g = self.gyro_noise_density;
        let sigma_a = self.accel_noise_density;
        let Qg = (sigma_g * sigma_g / dt) * I3; // Σ_gd
        let Qa = (sigma_a * sigma_a / dt) * I3; // Σ_ad
        na::stack![Qg, 0; 
            0, Qa]
    }

}

fn skew_symmetric(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
        na::Matrix3::new(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
    }