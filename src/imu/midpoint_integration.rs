use crate::{datasets::{ImuData, config}, imu};
use nalgebra as na;
use apex_solver::manifold::{LieGroup, so3::SO3};

pub struct ImuMidpointIntegration {
    // Fields for midpoint integration
    // prev_timestamp: f64,
    T_BS: na::Matrix4<f64>,
    accel_noise_density: na::Vector3<f64>,
    gyro_noise_density: na::Vector3<f64>,
    accel_random_walk: na::Vector3<f64>,
    gyro_random_walk: na::Vector3<f64>,
    preintegrated_noise: na::SVector<f64, 9>,
}

impl ImuMidpointIntegration {
    pub fn new() -> Self {
        // Initialize the midpoint integration
        ImuMidpointIntegration {
            // Initialize fields
            // prev_timestamp: 0.0,
            T_BS: nalgebra::Matrix4::identity(),
            accel_noise_density: na::Vector3::zeros(),
            gyro_noise_density: na::Vector3::zeros(),
            accel_random_walk: na::Vector3::zeros(),
            gyro_random_walk: na::Vector3::zeros(),
            preintegrated_noise: na::SVector::<f64, 9>::zeros(),
        }
    }

    pub fn from_config(config: config::ImuConfig) -> Self {
        ImuMidpointIntegration {
            // prev_timestamp: 0.0,
            T_BS: nalgebra::Matrix4::from_row_slice(&config.T_BS.data),
            accel_noise_density: na::Vector3::from_element(config.accel_noise_density),
            gyro_noise_density: na::Vector3::from_element(config.gyro_noise_density),
            accel_random_walk: na::Vector3::from_element(config.accel_random_walk),
            gyro_random_walk: na::Vector3::from_element(config.gyro_random_walk),
            preintegrated_noise: na::SVector::<f64, 9>::from_element(1.0),
        }
    }

    #[allow(non_snake_case)]
    pub fn integrate(&mut self, imu_slice: &[ImuData]) -> Vec<ImuData> {
        let mut prev_ts = imu_slice.first().map_or(0.0, |imu| imu.timestamp as f64 * 1e-9); // Initialize prev_timestamp to the timestamp of the first IMU measurement
        
        let mut dR_ik = SO3::identity(); // Initialize delta_R_j_i to identity
        let mut dv_ik = na::Vector3::zeros(); // Initialize delta_v_j_i to zero
        let mut dp_ik = na::Vector3::zeros(); // Initialize delta_p_j_i to zero

        let mut dt = 0.0; // Initialize dt to zero
        let mut dR_kkp1 = SO3::identity(); // Initialize delta_R_k_k+1 to identity
        // let mut dR_ikm1 = SO3::identity(); // Initialize delta_R_i_k+1 to identity

        let mut eta_phi = na::Vector3::zeros();
        let mut eta_v   = na::Vector3::zeros();
        let mut eta_p   = na::Vector3::zeros();

        for (i, imu) in imu_slice.iter().enumerate() {
            let ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            dt = ts - prev_ts;
            prev_ts = ts;

            let omega_unbiased = imu.gyro - self.gyro_random_walk;
            let acc_unbiased = imu.accel - self.accel_random_walk;

            let dphi = (omega_unbiased - self.gyro_noise_density)  * dt; // Angular increment
            dR_kkp1 = SO3::from_scaled_axis(dphi);
            dR_ik = SO3::from_quaternion(dR_ik.quaternion() * dR_kkp1.quaternion());

            dv_ik += dR_ik.quaternion() * (acc_unbiased - self.accel_noise_density) * dt;
            dp_ik += dR_ik.quaternion() * (acc_unbiased - self.accel_noise_density) * dt * dt * 1.5;

            let dR_ikm1 = dR_ik; // Cache delta_R_i_k before updating it for the next iteration
            
            let a_hat = skew_symmetric(&acc_unbiased);
            // eta_phi = dR_kkp1.rotation_matrix().inverse().transform_vector(&eta_phi);
            eta_phi = dR_kkp1.inverse().rotation_matrix().transform_vector(&eta_phi);
            eta_v = eta_v - (dR_ikm1.rotation_matrix().transform_vector(&(a_hat * eta_phi))) * dt;
            eta_p = eta_p + eta_v * dt - 0.5 * (dR_ikm1.rotation_matrix().transform_vector(&(a_hat * eta_phi))) * (dt * dt);   
        }
        
        log::debug!("Delta rotation (delta_q): {:?}", dR_ik.quaternion());
        log::debug!("Delta velocity (delta_v): {:?}", dv_ik);
        log::debug!("Delta position (delta_p): {:?}", dp_ik);
        // delta_q = 
        Vec::new() // Return an empty vector for now
    }

    pub fn compute_bias(&self, imu_slice: &[ImuData]) -> [f64; 6] {
        // Implement bias computation logic here
        // This is a placeholder implementation and should be replaced with actual logic
        log::info!("[ImuMidpointIntegration] Computing IMU bias (placeholder)");
        [0.0; 6] // Return zero bias for now
    }
}

fn skew_symmetric(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
        na::Matrix3::new(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
    }