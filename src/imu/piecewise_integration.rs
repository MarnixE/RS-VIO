use std::fmt;

use crate::{datasets::{ImuData, config}, imu, types::Vector3};
use faer::traits::math_utils::sqrt;
use imageproc::noise;
use nalgebra::{self as na, Cholesky, SymmetricEigen};
use apex_solver::manifold::{LieGroup, so3::SO3};

// type Matrix6 = na::SMatrix<f64, 6, 6>;
type Matrix15 = na::SMatrix<f64, 15, 15>;
type Matrix12 = na::SMatrix<f64, 12, 12>;
type Vector12 = na::SVector<f64, 12>;

#[derive(Clone)]
#[allow(non_snake_case, dead_code)]
pub struct PreInt {
    pub dR: SO3,
    pub dv: na::Vector3<f64>,
    pub dp: na::Vector3<f64>,
    pub cov: Matrix15,
    pub dt: f64,
    // pub bias_g: na::Vector3<f64>,
    // pub bias_a: na::Vector3<f64>,
    // pub Jr_bg: na::Matrix3<f64>,
    // pub Jv_bg: na::Matrix3<f64>,
    // pub Jv_ba: na::Matrix3<f64>,
    pub linearized_ba: na::Vector3<f64>,
    pub linearized_bg: na::Vector3<f64>,

    pub jacobian: na::SMatrix<f64, 15, 15>,
    // pub gyro_random_walk: na::Matrix3<f64>,
    // pub accel_random_walk: na::Matrix3<f64>,

    pub sqrt_info: Matrix15,
    // pub sqrt_info_bias: na::Matrix6<f64>,

    /// Buffers for repropagation
    pub imu_buffer: Vec<ImuData>,

    pub gravity: na::Vector3<f64>,

    pub idx_r: usize,
    pub idx_v: usize,
    pub idx_p: usize,
    pub idx_ba: usize,
    pub idx_bg: usize,
}

impl fmt::Debug for PreInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreInt")
            .field("dR", &"<SO3>")
            .field("dv", &self.dv)
            .field("dp", &self.dp)
            .field("dt", &self.dt)
            // .field("bias_g", &self.bias_g)
            // .field("bias_a", &self.bias_a)
            // .field("gyro_random_walk", &self.gyro_random_walk)
            // .field("accel_random_walk", &self.accel_random_walk)
            .finish()
    }
}

impl PreInt {
    #[allow(non_snake_case)]
    pub fn new(dR: SO3, dv: na::Vector3<f64>, dp: na::Vector3<f64>, cov: Matrix15, 
            dt_step: f64, bias_a: na::Vector3<f64>, bias_g: na::Vector3<f64>, jacobian: na::SMatrix<f64, 15, 15>, 
            sqrt_info: Matrix15, imu_slice: &[ImuData], gravity: na::Vector3<f64>) -> Self {
        Self {
            dR: dR,
            dv: dv,
            dp: dp,
            cov: cov,
            dt: dt_step, // Total time from the first to the last IMU measurement
            linearized_ba: bias_a, 
            linearized_bg: bias_g, 
            // gyro_random_walk: self.gyro_random_walk,
            // accel_random_walk: self.accel_random_walk,
            jacobian: jacobian,
            sqrt_info: sqrt_info,
            // sqrt_info_bias: na::Matrix6::identity(), // Identity for bias as well
            imu_buffer: imu_slice.to_vec(),
            gravity: gravity,
            idx_r: 0,
            idx_v: 3,
            idx_p: 6,
            idx_ba: 9,
            idx_bg: 12,
        }
    }

    pub fn finalize(&mut self) {
        // Symmetrize (numerical hygiene)
        let sigma = 0.5 * (self.cov + self.cov.transpose());

        // Cholesky Σ = L L^T
        let chol = Cholesky::new(sigma)
            .expect("Preint covariance not SPD; check propagation / add jitter");

        let L = chol.l();

        // W = L^{-1}  (triangular inverse is fine for 15x15)
        let W = L
            .try_inverse()
            .expect("Failed to invert Cholesky factor");

        self.sqrt_info = W.clone();
    }

    pub fn whiten_residual_15(&self, r: &na::SVector<f64, 15>) -> na::SVector<f64, 15> {
        self.sqrt_info * r
        // self.sqrt_info.solve_lower_triangular(&r).unwrap()
    }

    pub fn whiten_jacobian_15<const N: usize>(&self, J: &na::SMatrix<f64, 15, N>) -> na::SMatrix<f64, 15, N> {
        self.sqrt_info * J
        // self.sqrt_info.solve_lower_triangular(&J).unwrap()
    }
}

pub struct ImuPiecewiseIntegration {
    // Fields for piecewise integration
    // prev_timestamp: f64,
    T_BS: na::Matrix4<f64>,
    accel_noise_density: f64,
    gyro_noise_density: f64,
    accel_random_walk: f64,
    gyro_random_walk: f64,
    imu_buffer: Vec<ImuData>,
    // preintegrated_noise: na::SVector<f64, 9>,
    pub gravity: na::Vector3<f64>,
}

#[allow(non_snake_case)]
impl ImuPiecewiseIntegration {
    pub fn new() -> Self {
        // Initialize the piecewise integration
        ImuPiecewiseIntegration {
            // Initialize fields
            // prev_timestamp: 0.0,
            T_BS: nalgebra::Matrix4::identity(),
            accel_noise_density: 1.0,
            gyro_noise_density: 1.0,
            accel_random_walk: 1.0,
            gyro_random_walk: 1.0,
            imu_buffer: Vec::new(),
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            // preintegrated_noise: na::SVector::<f64, 9>::zeros(),
        }
    }

    pub fn from_config(config: config::ImuConfig) -> Self {
        ImuPiecewiseIntegration {
            // prev_timestamp: 0.0,
            T_BS: nalgebra::Matrix4::from_row_slice(&config.T_BS.data),
            accel_noise_density: config.accel_noise_density,
            gyro_noise_density: config.gyro_noise_density,
            accel_random_walk: config.accel_random_walk,
            gyro_random_walk: config.gyro_random_walk,
            imu_buffer: Vec::new(),
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            // preintegrated_noise: na::SVector::<f64, 9>::from_element(1.0),
        }
    }

    #[allow(non_snake_case)]
    pub fn propagate(&mut self, imu_slice: &[ImuData], bias_a: &Vector3, bias_g: &Vector3) -> PreInt {
        self.imu_buffer = imu_slice.to_vec(); // Store the IMU data for repropagation
        let mut prev_ts = imu_slice.first().map_or(0.0, |imu| imu.timestamp as f64 * 1e-9); // Initialize prev_timestamp to the timestamp of the first IMU measurement
        let first_ts = prev_ts.clone();
        let mut ts = 0.0;
        
        let mut dR_ik = SO3::identity(); // Initialize delta_R_j_i to identity
        let mut dv_ik = na::Vector3::zeros(); // Initialize delta_v_j_i to zero
        let mut dp_ik = na::Vector3::zeros(); // Initialize delta_p_j_i to zero

        let mut jacobian = na::SMatrix::<f64, 15, 15>::identity();

        let mut cov_ik = na::SMatrix::<f64, 15, 15>::zeros();
        

        for (i, imu) in imu_slice.iter().enumerate() {
            ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            let dt = ts - prev_ts;
            if dt <= 0.0 {
                continue;
            }
            prev_ts = ts;

            let acc_unbiased = imu.accel - bias_a;
            let omega_unbiased = imu.gyro - bias_g;

            let dphi = omega_unbiased  * dt; // Angular increment
            let dR_kkp1 = SO3::from_scaled_axis(dphi);
            let J_r = self.right_jacobian_so3(&dphi);

            let dv_ikm1 = dv_ik.clone(); // Cache delta_v_i_k before updating it for the next iteration
            dv_ik += dR_ik.rotation_matrix() * acc_unbiased * dt;
            dp_ik += dv_ikm1 * dt + dR_ik.rotation_matrix() * acc_unbiased * dt * dt * 0.5;

            let A = self.construct_A(&dR_kkp1, &dR_ik, &acc_unbiased, dt);
            let B = self.construct_B(&J_r, &dR_ik, dt);
            
            let noise = self.compute_noise(dt);

            cov_ik = A * cov_ik * A.transpose() + B * noise * B.transpose(); // Propagate covariance

            dR_ik = SO3::from_quaternion(dR_ik.quaternion() * dR_kkp1.quaternion());

            jacobian = A * jacobian; // Update the Jacobian of the preintegrated measurement w.r.t. biases
        }
        
        let dt_step = ts - first_ts;

        if dt_step <= 0.0 {
            log::warn!("Invalid time interval for preintegration: dt_step = {}", dt_step);
            return PreInt::new(
                SO3::identity(),
                na::Vector3::zeros(),
                na::Vector3::zeros(),
                Matrix15::identity() * 1e-6, // Small covariance to avoid singularity
                dt_step,
                na::Vector3::zeros(),
                na::Vector3::zeros(),
                na::SMatrix::<f64, 15, 15>::identity(),
                Matrix15::identity(), // Identity since covariance is near zero
                imu_slice, // Store the raw IMU data for potential repropagation
                self.gravity,
            )
        }

        // let sigma = self.bias_rw_cov(dt_step, 
        //     self.gyro_random_walk, 
        //     self.accel_random_walk,
        // );
        // print!("Sqrt info:\n{:?}", self.compute_sqrt_info(&cov_ik));
        PreInt::new(
            dR_ik,
            dv_ik,
            dp_ik,
            cov_ik,
            dt_step.clone(), // Total time from the first to the last IMU measurement
            // bias_g,
            // bias_a,
            bias_a.clone(), 
            bias_g.clone(), 
            // gyro_random_walk: self.gyro_random_walk,
            // accel_random_walk: self.accel_random_walk,
            jacobian,
            self.compute_sqrt_info(&cov_ik), // Compute square root information matrix
            // sqrt_info_bias: na::Matrix6::identity(), // Identity for bias as well
            imu_slice, // Store the raw IMU data for potential repropagation
            self.gravity, // Default gravity
        )
    }

    pub fn repropagate(&mut self, new_bias_a: &Vector3, new_bias_g: &Vector3) -> PreInt {
        let imu_slice = self.imu_buffer.clone(); // Get the original IMU data slice
        
        // Re-run the propagate function with the stored IMU data and new biases
        self.propagate(&imu_slice, new_bias_a, new_bias_g)
    }

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

    fn construct_A(&self, dR_kkp1: &SO3, dR_ik: &SO3, acc_unbiased: &na::Vector3<f64>, dt: f64) -> na::SMatrix<f64, 15, 15> {
        let I3 = na::Matrix3::<f64>::identity();

        let R_ik = dR_ik.rotation_matrix();               // ΔR_{i,k}
        let R_kp1k_T = dR_kkp1.rotation_matrix().transpose(); // (ΔR_{k,k+1})^T

        let a_hat = skew_symmetric(&acc_unbiased);

        let A_phiphi = R_kp1k_T;
        let A_vphi = -R_ik * a_hat * dt;
        let A_pphi = -0.5 * R_ik * a_hat * (dt * dt);
        let A_pv = I3 * dt;

        let A_phibg = -I3 * dt;
        let A_vba   = -R_ik * dt;
        let A_pba   = -0.5 * R_ik * dt * dt;
        
        // let A: na::SMatrix<f64, 9, 9> = 
        na::stack![
            A_phiphi, 0, 0, 0, A_phibg;
            A_vphi, I3, 0, A_vba, 0;
            A_pphi, A_pv, I3, A_pba, 0;
            0, 0, 0, I3, 0;
            0, 0, 0, 0, I3
        ]
        // A
    }

    fn construct_B(&self, Jr: &na::Matrix3<f64>, dR_ik: &SO3, dt: f64) -> na::SMatrix<f64, 15, 12> {
        let I3 = na::Matrix3::<f64>::identity();
        let R_ik = dR_ik.rotation_matrix();

        let B_phig = Jr * dt;
        let B_va   = R_ik * dt;
        let B_pa   = 0.5 * R_ik * (dt * dt);

        let B_ba = I3 * dt;                   // b_a <- accel-bias RW noise
        let B_bg = I3 * dt;                   // b_g <- gyro-bias  RW noise

        na::stack![
            0, B_phig, 0, 0;
            B_va, 0, 0, 0;
            B_pa, 0, 0, 0;
            0, 0, B_ba, 0;
            0, 0, 0, B_bg
        ]
    }

    fn compute_noise(&self, dt: f64) -> na::SMatrix<f64, 12, 12> {
        let I3 = na::Matrix3::<f64>::identity();

        let gyr_n = self.gyro_noise_density / sqrt(&dt);   // GYR_N in VINS config meaning
        let acc_n = self.accel_noise_density / sqrt(&dt);  // ACC_N
        let acc_w = self.accel_random_walk / sqrt(&dt);        // ACC_W
        let gyr_w = self.gyro_random_walk / sqrt(&dt);         // GYR_W

        let Qa = (acc_n * acc_n) * I3;  // n_a
        let Qg = (gyr_n * gyr_n) * I3;   // n_omega

        let Qba = (acc_w * acc_w) * I3;        // n_ba
        let Qbg = (gyr_w * gyr_w) * I3;        // n_bg

        na::stack![
            Qa, 0, 0, 0;
            0, Qg, 0, 0;
            0, 0, Qba, 0;
            0, 0, 0, Qbg
        ]
    }

    pub fn compute_sqrt_info(&self, sigma: &Matrix15) -> Matrix15 {
        let info = sigma.try_inverse().expect("Failed to invert bias covariance for sqrt info");
        let chol_info = na::Cholesky::new(info);
        let lw = chol_info.as_ref().unwrap().l();

        let sqrt_info = lw.transpose();
        let diff = sqrt_info.transpose() * sqrt_info - info;
        let max_diff = diff.iter().fold(0.0, |max, &val| val.abs().max(max));

        let err_f = diff.norm();
        let info_f = info.norm();
        assert!(err_f / info_f < 1e-10);

        sqrt_info
    }

    pub fn sqrt_info_bias(&self, sigma: &na::SMatrix<f64, 6, 6>) -> na::SMatrix<f64, 6, 6> {
        let info = sigma.try_inverse().expect("Failed to invert bias covariance for sqrt info");
        let chol_info = na::Cholesky::new(info);
        let lw = chol_info.as_ref().unwrap().l();
        lw.transpose() // Since info = L L^T, sqrt_info = L^T
    }

    fn bias_rw_cov(&self, dt: f64, q_ba: na::Matrix3<f64>, q_bg: na::Matrix3<f64>) -> na::Matrix6<f64> {
        // Cov = blockdiag( dt*Q_bg, dt*Q_ba )
        let mut sigma = na::Matrix6::<f64>::zeros();
        sigma.fixed_view_mut::<3,3>(0,0).copy_from(&(dt * q_ba));
        sigma.fixed_view_mut::<3,3>(3,3).copy_from(&(dt * q_bg));
        sigma
    }

}

fn skew_symmetric(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
        na::Matrix3::new(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
    }