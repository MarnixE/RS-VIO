use std::fmt;

use crate::{datasets::{ImuData, config}, estimator::state, imu, types::Vector3};
use faer::{row, traits::math_utils::sqrt};
use imageproc::noise;
use nalgebra::{self as na, Cholesky, SymmetricEigen};
use apex_solver::manifold::{LieGroup, so3::SO3};
use crate::estimator::Frame;

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
    pub fn propagate(&mut self, imu_slice: &[ImuData], current_frame: &mut Frame, new_bias_a: &Option<Vector3>, new_bias_g: &Option<Vector3>) {
        if (imu_slice.is_empty()) {
            log::warn!("No IMU data to propagate");
            return;
        }

        self.imu_buffer = imu_slice.to_vec(); // Store the IMU data for repropagation
        let mut prev_ts = imu_slice.first().map_or(0.0, |imu| imu.timestamp as f64 * 1e-9); // Initialize prev_timestamp to the timestamp of the first IMU measurement
        let first_ts = prev_ts.clone();
        let mut ts = 0.0;
        
        let mut dR_ik = SO3::identity(); // Initialize delta_R_j_i to identity
        let mut dv_ik = na::Vector3::zeros(); // Initialize delta_v_j_i to zero
        let mut dp_ik = na::Vector3::zeros(); // Initialize delta_p_j_i to zero

        let mut jacobian = na::SMatrix::<f64, 15, 15>::identity();

        let mut cov_ik = na::SMatrix::<f64, 15, 15>::zeros();

        let state = &current_frame.state;
        
        let mut R_wb = state.T_W_B.fixed_view::<3, 3>(0, 0).clone_owned(); // Rotation world from body
        let mut t_wb = state.T_W_B.fixed_view::<3, 1>(0, 3).clone_owned(); // Translation world from body
        let mut v = state.velocity; // Initial velocity in world frame
        let bias_a = new_bias_a.unwrap_or(state.accel_bias); // linearlized accel bias (assume constant during preintegration)
        let bias_g = new_bias_g.unwrap_or(state.gyro_bias); // linearlized gyro bias (assume constant during preintegration)

        let mut acc_0 = imu_slice.first().map_or(na::Vector3::zeros(), |imu| imu.accel);
        let mut gyr_0 = imu_slice.first().map_or(na::Vector3::zeros(), |imu| imu.gyro);

        for (i, imu) in imu_slice.iter().enumerate().skip(1) {
            ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            let dt = ts - prev_ts;
            if dt <= 0.0 {
                continue;
            }
            prev_ts = ts;

            // let acc_unbiased = imu.accel - bias_a;
            // let omega_unbiased = imu.gyro - bias_g;
            let acc_unbiased = acc_0 - bias_a; // Bias-corrected acceleration
            let omega_unbiased = gyr_0 - bias_g; // Bias-corrected

            let dphi = omega_unbiased  * dt; // Angular increment
            let axis = na::Unit::new_normalize(dphi);
            let angle = dphi.norm();
            let dR_kkp1 = SO3::from_quaternion(na::UnitQuaternion::from_axis_angle(&axis, angle));
            let J_r = self.right_jacobian_so3(&dphi);

            let dv_ikm1 = dv_ik.clone(); // Cache delta_v_i_k before updating it for the next iteration
            dv_ik += dR_ik.rotation_matrix() * acc_unbiased * dt;
            dp_ik += dv_ikm1 * dt + dR_ik.rotation_matrix() * acc_unbiased * dt * dt * 0.5;

            let A = self.construct_A(&dR_kkp1, &dR_ik, &acc_unbiased, dt);
            let B = self.construct_B(&J_r, &dR_ik, dt, &acc_unbiased);
            
            let noise = self.compute_noise(dt);

            cov_ik = A * cov_ik * A.transpose() + B * noise * B.transpose(); // Propagate covariance

            dR_ik = SO3::from_quaternion(dR_ik.quaternion() * dR_kkp1.quaternion());

            jacobian = A * jacobian; // Update the Jacobian of the preintegrated measurement w.r.t. biases

            // Update the state
            let acc_w_un = R_wb * (acc_0 - bias_a);
            let acc_unbiased_world = R_wb * (acc_0 - bias_a) + self.gravity; // Transform acceleration to world frame for bias correction
            let omega_unbiased_world = gyr_0 - bias_g; // Angular velocity in world frame for bias correction
            R_wb *= SO3::from_scaled_axis(omega_unbiased_world * dt).rotation_matrix();
            t_wb += v * dt + 0.5 * acc_unbiased_world * dt * dt;
            v += acc_unbiased_world * dt;
            acc_0 = imu.accel;
            gyr_0 = imu.gyro;
            let a = 10;
        }

        let mut T_W_B = na::Matrix4::<f64>::identity();
        T_W_B.fixed_view_mut::<3, 3>(0, 0).copy_from(&R_wb);
        T_W_B.fixed_view_mut::<3, 1>(0, 3).copy_from(&t_wb);

        print!("Final T_W_B: \n{:?}\n", T_W_B);
        print!("Final velocity: \n{:?}\n", v);

        // print!("Final T_W_B: {:?}", T_W_B);
        current_frame.state.T_W_B = T_W_B.try_into().expect("Failed to convert T_W_B to Matrix4x4");
        current_frame.state.velocity = v;

        
        let dt_step = ts - first_ts;

        if dt_step <= 0.0 {
            log::warn!("Invalid time interval for preintegration: dt_step = {}", dt_step);
            // return PreInt::new(
            //     SO3::identity(),
            //     na::Vector3::zeros(),
            //     na::Vector3::zeros(),
            //     Matrix15::identity() * 1e-6, // Small covariance to avoid singularity
            //     dt_step,
            //     na::Vector3::zeros(),
            //     na::Vector3::zeros(),
            //     na::SMatrix::<f64, 15, 15>::identity(),
            //     Matrix15::identity(), // Identity since covariance is near zero
            //     imu_slice, // Store the raw IMU data for potential repropagation
            //     self.gravity,
            // )
        }
        
        let preint = PreInt::new(
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
        );

        current_frame.imu_preintegration = Some(preint);
    }

    pub fn repropagate(&mut self, new_bias_a: &Vector3, new_bias_g: &Vector3, current_frame: &mut Frame) {
        let imu_slice = self.imu_buffer.clone(); // Get the original IMU data slice
        
        // Re-run the propagate function with the stored IMU data and new biases
        self.propagate(&imu_slice, current_frame, &Some(new_bias_a.clone()), &Some(new_bias_g.clone()))
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

    fn construct_B(&self, Jr: &na::Matrix3<f64>, dR_ik: &SO3, dt: f64, acc_unbiased: &na::Vector3<f64>) -> na::SMatrix<f64, 15, 12> {
        let I3 = na::Matrix3::<f64>::identity();
        let R_ik = dR_ik.rotation_matrix();

        let B_phig = I3 * dt; // Instead of Jr
        let B_va   = R_ik * dt;
        let B_pa   = 0.5 * R_ik * (dt * dt);
        let B_pg = -0.5 * R_ik * skew_symmetric(acc_unbiased) * dt * dt; // Gyro noise also affects position through rotation error

        let B_ba = I3 * dt;                   // b_a <- accel-bias RW noise
        let B_bg = I3 * dt;                   // b_g <- gyro-bias  RW noise

        na::stack![
            0, B_phig, 0, 0;
            B_va, 0, 0, 0;
            B_pa, B_pg, 0, 0;
            0, 0, B_ba, 0;
            0, 0, 0, B_bg
        ]
    }

    fn compute_noise(&self, dt: f64) -> na::SMatrix<f64, 12, 12> {
        let I3 = na::Matrix3::<f64>::identity();

        let gyr_n = self.gyro_noise_density / sqrt(&dt); // rad / s
        let acc_n = self.accel_noise_density / sqrt(&dt);  // m / s^2
        let acc_w = self.accel_random_walk / sqrt(&dt); // m / s^3
        let gyr_w = self.gyro_random_walk / sqrt(&dt); // rad / s^2    

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

    // pub fn compute_sqrt_info(&self, sigma: &Matrix15) -> Matrix15 {
    //     // let info = sigma.try_inverse().expect("Failed to invert bias covariance for sqrt info");
    //     let chol_info = na::Cholesky::new(sigma.clone());
    //     let lw = chol_info.as_ref().unwrap().l();

    //     let sqrt_info = lw.transpose();
    //     let diff = sqrt_info.transpose() * sqrt_info - sigma;
    //     let max_diff = diff.iter().fold(0.0, |max, &val| val.abs().max(max));

    //     let err_f = diff.norm();
    //     let info_f = sigma.norm();
    //     assert!(err_f / info_f < 1e-10);
    //     // print!("Sqrt info:\n{:?}", sqrt_info);
    //     let cov = sqrt_info.try_inverse().expect("Failed to invert sqrt_info");
    //     // print!("cov:\n{:?}", cov);
    //     sqrt_info
    // }

    pub fn compute_sqrt_info(&self, sigma: &Matrix15) -> Matrix15 {
        // Σ = L Lᵀ
        // for (i, row) in sigma.row_iter().enumerate() {
        //     print!("Sigma: Row {}: {:?}\n", i, row.clone_owned());
        // } 
        let chol = na::Cholesky::new(*sigma).expect("Matrix is not SPD"); // None if not SPD [page:1]
        let l: Matrix15 = chol.l();            // lower-triangular factor L [page:1]

        // Solve Lᵀ X = I  =>  X = L^{-T} = Σ^{-1/2}
        let i = Matrix15::identity();
        let sqrt_info = l.tr_solve_lower_triangular(&i).expect("Failed to solve"); // Option<Matrix15> [page:0]
        // for (i, row) in sqrt_info.row_iter().enumerate() {
        //     print!("Sqrt_info: Row {}: {:?}\n", i, row.clone_owned());
        // } 
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