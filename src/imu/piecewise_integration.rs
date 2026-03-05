use std::fmt;

use crate::{datasets::{ImuData, config}, estimator::state, imu, types::Vector3};
use faer::{row, traits::math_utils::sqrt};
use imageproc::noise;
use nalgebra::{self as na, Cholesky, SymmetricEigen};
use apex_solver::manifold::{LieGroup, so3::SO3};
use crate::estimator::Frame;

type Matrix15 = na::SMatrix<f64, 15, 15>;
type Matrix12 = na::SMatrix<f64, 12, 12>;
type Vector12 = na::SVector<f64, 12>;

#[derive(Clone)]
#[allow(non_snake_case, dead_code)]
pub struct PreInt {
    pub delta_R: SO3,
    pub delta_v: na::Vector3<f64>,
    pub delta_p: na::Vector3<f64>,
    pub cov: Matrix15,
    pub dt: f64,
    pub linearized_ba: na::Vector3<f64>,
    pub linearized_bg: na::Vector3<f64>,

    pub jacobian: na::SMatrix<f64, 15, 15>,
    pub sqrt_info: Matrix15,

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
            .field("delta_R", &"<SO3>")
            .field("delta_v", &self.delta_v)
            .field("delta_p", &self.delta_p)
            .field("dt", &self.dt)
            .finish()
    }
}

impl PreInt {
    #[allow(non_snake_case)]
    pub fn new(delta_R: SO3, delta_v: na::Vector3<f64>, delta_p: na::Vector3<f64>, cov: Matrix15, 
            dt_step: f64, bias_a: na::Vector3<f64>, bias_g: na::Vector3<f64>, jacobian: na::SMatrix<f64, 15, 15>, 
            sqrt_info: Matrix15, imu_slice: &[ImuData], gravity: na::Vector3<f64>) -> Self {
        Self {
            delta_R: delta_R,
            delta_v: delta_v,
            delta_p: delta_p,
            cov: cov,
            dt: dt_step, // Total time from the first to the last IMU measurement
            linearized_ba: bias_a, 
            linearized_bg: bias_g, 
            jacobian: jacobian,
            sqrt_info: sqrt_info,
            imu_buffer: imu_slice.to_vec(),
            gravity: gravity,
            idx_r: 3,
            idx_v: 6,
            idx_p: 0,
            idx_ba: 9,
            idx_bg: 12,
        }
    }

    pub fn identity() -> Self {
        Self {
            delta_R: SO3::identity(),
            delta_v: na::Vector3::zeros(),
            delta_p: na::Vector3::zeros(),
            cov: Matrix15::zeros(), // Small covariance to avoid singularity
            dt: 0.0,
            linearized_ba: na::Vector3::zeros(), 
            linearized_bg: na::Vector3::zeros(), 
            jacobian: na::SMatrix::<f64, 15, 15>::identity(),
            sqrt_info: Matrix15::identity(), // Identity since covariance is near zero
            imu_buffer: Vec::new(),
            gravity: na::Vector3::new(0.0, 0.0, 9.81007),
            idx_r: 3,
            idx_v: 6,
            idx_p: 0,
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
    }

    pub fn whiten_jacobian_15<const N: usize>(&self, J: &na::SMatrix<f64, 15, N>) -> na::SMatrix<f64, 15, N> {
        self.sqrt_info * J
    }
}

pub struct ImuPiecewiseIntegration {
    T_BS: na::Matrix4<f64>,
    accel_noise_density: f64,
    gyro_noise_density: f64,
    accel_random_walk: f64,
    gyro_random_walk: f64,
    imu_buffer: Vec<ImuData>,
    pub gravity: na::Vector3<f64>,
    continious_noise: Matrix12,
}

pub enum ImuPipeline {
    Enabled(ImuPiecewiseIntegration),
    Disabled,
}

impl ImuPipeline {
    pub fn from_config(config: Option<config::ImuConfig>) -> Self {
        match config {
            Some(cfg) => Self::Enabled(ImuPiecewiseIntegration::from_config(cfg)),
            None => Self::Disabled,
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }

    pub fn process_imu(&mut self, imu_slice: &[ImuData], current_frame: &mut Frame) -> bool {
        match self {
            Self::Enabled(integrator) => integrator.process_imu(imu_slice, current_frame),
            Self::Disabled => false,
        }
    }

    pub fn repropagate(&mut self, new_bias_a: &Vector3, new_bias_g: &Vector3) -> Option<PreInt> {
        match self {
            Self::Enabled(integrator) => Some(integrator.repropagate(new_bias_a, new_bias_g)),
            Self::Disabled => None,
        }
    }

    pub fn set_gravity(&mut self, gravity: na::Vector3<f64>) {
        if let Self::Enabled(integrator) = self {
            integrator.gravity = gravity;
        }
    }
}

#[allow(non_snake_case)]
impl ImuPiecewiseIntegration {
    pub fn new() -> Self {
        // Initialize the piecewise integration
        ImuPiecewiseIntegration {
            // Initialize fields
            T_BS: nalgebra::Matrix4::identity(),
            accel_noise_density: 1.0,
            gyro_noise_density: 1.0,
            accel_random_walk: 1.0,
            gyro_random_walk: 1.0,
            imu_buffer: Vec::new(),
            gravity: na::Vector3::new(0.0, 0.0, 9.81007),
            continious_noise: na::SMatrix::<f64, 12, 12>::zeros(), // Placeholder, should be set based on noise parameters
        }
    }

    pub fn from_config(config: config::ImuConfig) -> Self {
        ImuPiecewiseIntegration {
            T_BS: nalgebra::Matrix4::from_row_slice(&config.T_BS.data),
            accel_noise_density: config.accel_noise_density,
            gyro_noise_density: config.gyro_noise_density,
            accel_random_walk: config.accel_random_walk,
            gyro_random_walk: config.gyro_random_walk,
            imu_buffer: Vec::new(),
            gravity: na::Vector3::new(0.0, 0.0, 9.81007),
            continious_noise: na::SMatrix::<f64, 12, 12>::zeros(), // Placeholder, should be set based on noise parameters
        }
    }

    #[allow(non_snake_case)]
    pub fn process_imu(&mut self, imu_slice: &[ImuData], current_frame: &mut Frame) -> bool {
        if (imu_slice.is_empty()) {
            log::warn!("No IMU data to propagate");
            return false;
        }
        let total_ts = (imu_slice.last().unwrap().timestamp - imu_slice.first().unwrap().timestamp) as f64 * 1e-9;
        if total_ts >= 5.0 {
            log::warn!("Large time gap in IMU data: {} seconds", total_ts);
            return false;
        }

        self.imu_buffer = imu_slice.to_vec(); // Store the IMU data for repropagation

        let state = &current_frame.state;
        let mut R_wb = state.T_W_B.fixed_view::<3, 3>(0, 0).clone_owned(); // Rotation world from body
        let mut t_wb = state.T_W_B.fixed_view::<3, 1>(0, 3).clone_owned(); // Translation world from body
        let mut v = state.velocity; // Initial velocity in world frame

        let bias_a = state.accel_bias; // linearlized accel bias (assume constant during preintegration)
        let bias_g = state.gyro_bias; // linearlized gyro bias (assume constant during preintegration)

        let mut prev_ts = imu_slice.first().map_or(0.0, |imu| imu.timestamp as f64 * 1e-9);
        let t_wb_start = t_wb.clone(); // Store initial translation for sanity check after propagation

        // Initialize preintegration variables
        let mut preint = PreInt::identity();
        preint.linearized_ba = bias_a;
        preint.linearized_bg = bias_g;

        let mut acc_0 = imu_slice.first().map(|imu| imu.accel).unwrap();
        let mut gyr_0 = imu_slice.first().map(|imu| imu.gyro).unwrap();

        self.continious_noise = self.compute_noise();

        for (i, imu) in imu_slice.iter().enumerate().skip(1) {
            let ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            let dt = ts - prev_ts;
            preint.dt += dt; 
            if dt <= 0.0 {
                log::warn!("Non-positive time interval between IMU measurements: dt = {}", dt);
                continue;
            }

            self.propagate(&mut preint,
                dt,
                &acc_0,
                &gyr_0,
                &bias_a, 
                &bias_g
            );

            let acc_unbiased_world = R_wb * (acc_0 - bias_a) - self.gravity; // Transform acceleration to world frame for bias correction
            let omega_unbiased_world = gyr_0 - bias_g; // Angular velocity in world frame for bias correction
            R_wb *= SO3::from_scaled_axis(omega_unbiased_world * dt).rotation_matrix();
            t_wb += v * dt + 0.5 * acc_unbiased_world * dt * dt;
            v += acc_unbiased_world * dt;

            acc_0 = imu.accel;
            gyr_0 = imu.gyro;
            prev_ts = ts;
        }

        let t_wb_step = t_wb - t_wb_start;
        print!("Translation step norm after IMU propagation: {}\n", t_wb_step.norm());
        if t_wb_step.norm() > 10.0 {
            log::warn!("Large translation after IMU propagation: t_wb = {:?}", t_wb_step);
            return false;
        }
        
        // Update the state
        let mut T_W_B = na::Matrix4::<f64>::identity();
        T_W_B.fixed_view_mut::<3, 3>(0, 0).copy_from(&R_wb);
        T_W_B.fixed_view_mut::<3, 1>(0, 3).copy_from(&t_wb);

        // current_frame.state.T_W_B = T_W_B.try_into().expect("Failed to convert T_W_B to Matrix4x4");
        // current_frame.state.velocity = v;
        current_frame.imu_preintegration = Some(preint);
        true
    }

    #[allow(non_snake_case)]
    #[inline]
    pub fn propagate(&mut self, preint: &mut PreInt, dt: f64, acc_0: &Vector3, gyr_0: &Vector3, 
            bias_a_linearized: &Vector3, bias_g_linearized: &Vector3) {
        let acc_unbiased = acc_0 - bias_a_linearized; // Bias-corrected acceleration
        let omega_unbiased = gyr_0 - bias_g_linearized; // Bias-corrected angular velocity

        let dphi = omega_unbiased  * dt; // Angular increment
        let dR_kkp1 = SO3::from_quaternion(na::UnitQuaternion::from_scaled_axis(dphi));
        let J_r = self.right_jacobian_so3(&dphi);

        // let dv_ikm1 = preint.delta_v; // Cache delta_v_i_k before updating it for the next iteration
        preint.delta_p += preint.delta_v * dt + preint.delta_R.rotation_matrix() * acc_unbiased * dt * dt * 0.5;
        preint.delta_v += preint.delta_R.rotation_matrix() * acc_unbiased * dt;

        let A = self.construct_A(preint, &dR_kkp1, &preint.delta_R, &acc_unbiased, dt);
        let B = self.construct_B(preint, &J_r, &preint.delta_R, &acc_unbiased, dt);
        
        let noise = self.continious_noise / dt; // Discretize noise
        preint.cov = A * preint.cov * A.transpose() + B * noise * B.transpose(); // Propagate covariance

        preint.delta_R = SO3::from_quaternion(preint.delta_R.quaternion() * dR_kkp1.quaternion());

        preint.jacobian = A * preint.jacobian; // Update the Jacobian of the preintegrated measurement w.r.t. biases
    }

    pub fn repropagate(&mut self, new_bias_a: &Vector3, new_bias_g: &Vector3) -> PreInt {
        let imu_slice = self.imu_buffer.clone(); // Get the original IMU data slice

        let mut preint = PreInt::identity();

        let mut acc_0 = imu_slice.first().map(|imu| imu.accel).unwrap();
        let mut gyr_0 = imu_slice.first().map(|imu| imu.gyro).unwrap();
        let mut prev_ts = imu_slice.first().map(|imu| imu.timestamp as f64 * 1e-9).unwrap();
        
        // Re-run the propagate function with the stored IMU data and new biases
        for (i, imu) in imu_slice.iter().enumerate().skip(1) {
            let ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            let dt = ts - prev_ts;
            preint.dt += dt;

            if dt <= 0.0 {
                log::warn!("Non-positive time interval between IMU measurements: dt = {}", dt);
                continue;
            }

            self.propagate(
                &mut preint,
                dt,
                &acc_0,
                &gyr_0,
                &new_bias_a, 
                &new_bias_g
            );

            acc_0 = imu.accel;
            gyr_0 = imu.gyro;
            prev_ts = ts;
        }
        preint
    }

    #[inline]
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

    #[inline]
    fn construct_A(&self, preint: &PreInt, dR_kkp1: &SO3, dR_ik: &SO3, acc_unbiased: &na::Vector3<f64>, dt: f64) -> na::SMatrix<f64, 15, 15> {
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

        let mut A = na::SMatrix::<f64, 15, 15>::zeros();

        let idx_r = preint.idx_r;
        let idx_v = preint.idx_v;
        let idx_p = preint.idx_p;
        let idx_ba = preint.idx_ba;
        let idx_bg = preint.idx_bg;

        A.fixed_view_mut::<3, 3>(idx_r, idx_r).copy_from(&A_phiphi);
        A.fixed_view_mut::<3, 3>(idx_v, idx_r).copy_from(&A_vphi);
        A.fixed_view_mut::<3, 3>(idx_p, idx_r).copy_from(&A_pphi);
        A.fixed_view_mut::<3, 3>(idx_p, idx_v).copy_from(&A_pv);

        A.fixed_view_mut::<3, 3>(idx_v, idx_v).copy_from(&I3);
        A.fixed_view_mut::<3, 3>(idx_p, idx_p).copy_from(&I3);
        A.fixed_view_mut::<3, 3>(idx_ba, idx_ba).copy_from(&I3);
        A.fixed_view_mut::<3, 3>(idx_bg, idx_bg).copy_from(&I3);

        A.fixed_view_mut::<3, 3>(idx_r, idx_bg).copy_from(&A_phibg);
        A.fixed_view_mut::<3, 3>(idx_v, idx_ba).copy_from(&A_vba);
        A.fixed_view_mut::<3, 3>(idx_p, idx_ba).copy_from(&A_pba);

        A
    }

    #[inline]
    fn construct_B(&self, preint: &PreInt, Jr: &na::Matrix3<f64>, dR_ik: &SO3, acc_unbiased: &na::Vector3<f64>, dt: f64) -> na::SMatrix<f64, 15, 12> {
        let I3 = na::Matrix3::<f64>::identity();
        let R_ik = dR_ik.rotation_matrix();

        let B_phig = I3 * dt; // Instead of Jr
        let B_va   = R_ik * dt;
        let B_pa   = 0.5 * R_ik * (dt * dt);
        let B_pg = -0.5 * R_ik * skew_symmetric(acc_unbiased) * dt * dt; // Gyro noise also affects position through rotation error

        let B_ba = I3 * dt;                   // b_a <- accel-bias RW noise
        let B_bg = I3 * dt;                   // b_g <- gyro-bias  RW noise

        let mut B = na::SMatrix::<f64, 15, 12>::zeros();

        let idx_r = preint.idx_r;
        let idx_v = preint.idx_v;
        let idx_p = preint.idx_p;
        let idx_ba = preint.idx_ba;
        let idx_bg = preint.idx_bg;

        B.fixed_view_mut::<3, 3>(idx_r, 3).copy_from(&B_phig);
        B.fixed_view_mut::<3, 3>(idx_v, 0).copy_from(&B_va);
        B.fixed_view_mut::<3, 3>(idx_p, 0).copy_from(&B_pa);
        B.fixed_view_mut::<3, 3>(idx_p, 3).copy_from(&B_pg);
        B.fixed_view_mut::<3, 3>(idx_ba, 6).copy_from(&B_ba);
        B.fixed_view_mut::<3, 3>(idx_bg, 9).copy_from(&B_bg);

        B
    }

    fn compute_noise(&self) -> na::SMatrix<f64, 12, 12> {
        let I3 = na::Matrix3::<f64>::identity();

        let gyr_n = self.gyro_noise_density; // rad / s
        let acc_n = self.accel_noise_density;  // m / s^2
        let acc_w = self.accel_random_walk; // m / s^3
        let gyr_w = self.gyro_random_walk; // rad / s^2    

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
        // Σ = L Lᵀ
        let chol = na::Cholesky::new(*sigma).expect("Matrix is not SPD"); 
        let l: Matrix15 = chol.l(); 

        // Solve Lᵀ X = I  =>  X = L^{-T} = Σ^{-1/2}
        let i = Matrix15::identity();
        let sqrt_info = l.tr_solve_lower_triangular(&i).expect("Failed to solve"); 
        sqrt_info
    }

    pub fn sqrt_info_bias(&self, sigma: &na::SMatrix<f64, 6, 6>) -> na::SMatrix<f64, 6, 6> {
        let info = sigma.try_inverse().expect("Failed to invert bias covariance for sqrt info");
        let chol_info = na::Cholesky::new(info);
        let lw = chol_info.as_ref().unwrap().l();
        lw.transpose() // Since info = L L^T, sqrt_info = L^T
    }

    // fn bias_rw_cov(&self, dt: f64, q_ba: na::Matrix3<f64>, q_bg: na::Matrix3<f64>) -> na::Matrix6<f64> {
    //     // Cov = blockdiag( dt*Q_bg, dt*Q_ba )
    //     let mut sigma = na::Matrix6::<f64>::zeros();
    //     sigma.fixed_view_mut::<3,3>(0,0).copy_from(&(dt * q_ba));
    //     sigma.fixed_view_mut::<3,3>(3,3).copy_from(&(dt * q_bg));
    //     sigma
    // }

}

fn skew_symmetric(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
        na::Matrix3::new(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
    }