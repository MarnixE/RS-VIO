use std::fmt;

use crate::{datasets::{ImuData, config}, imu, types::Vector3};
use imageproc::noise;
use nalgebra::{self as na, Cholesky};
use apex_solver::manifold::{LieGroup, so3::SO3};

type Mat6 = na::SMatrix<f64, 6, 6>;
type Mat9 = na::SMatrix<f64, 9, 9>;
type Vec9 = na::SVector<f64, 9>;

#[derive(Clone)]
#[allow(non_snake_case, dead_code)]
pub struct PreInt {
    pub dR: SO3,
    pub dv: na::Vector3<f64>,
    pub dp: na::Vector3<f64>,
    pub cov: Mat9,
    pub dt: f64,
    // pub bias_g: na::Vector3<f64>,
    // pub bias_a: na::Vector3<f64>,
    pub Jr_bg: na::Matrix3<f64>,
    pub Jv_bg: na::Matrix3<f64>,
    pub Jv_ba: na::Matrix3<f64>,
    pub Jp_bg: na::Matrix3<f64>,
    pub Jp_ba: na::Matrix3<f64>,

    pub gyro_random_walk: na::Vector3<f64>,
    pub accel_random_walk: na::Vector3<f64>,

    pub inv_chol: Mat9,
    pub inv_chol_bias: Mat6,
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
            .field("gyro_random_walk", &self.gyro_random_walk)
            .field("accel_random_walk", &self.accel_random_walk)
            .finish()
    }
}

impl PreInt {
    pub fn finalize(&mut self) {
        // Symmetrize (numerical hygiene)
        let sigma = 0.5 * (self.cov + self.cov.transpose());

        // Cholesky Σ = L L^T
        let chol = Cholesky::new(sigma)
            .expect("Preint covariance not SPD; check propagation / add jitter");

        let L = chol.l();

        // W = L^{-1}  (triangular inverse is fine for 9x9)
        let W = L
            .try_inverse()
            .expect("Failed to invert Cholesky factor");

        self.inv_chol = W.clone();
    }

    pub fn whiten_residual_9(&self, r: &Vec9) -> Vec9 {
        self.inv_chol * r
    }
     
    pub fn whiten_residual_6(&self, r: &na::SVector<f64, 6>) -> na::SVector<f64, 6> {
        self.inv_chol_bias * r
    }

    pub fn whiten_jacobian_15<const N: usize>(&self, J: &na::SMatrix<f64, 15, N>) -> na::SMatrix<f64, 15, N> {
        // self.inv_chol * J
        let mat = na::stack![
            self.inv_chol, 0;
            0, self.inv_chol_bias
        ];
        mat * J
    }
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
            accel_noise_density: 1.0,
            gyro_noise_density: 1.0,
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
    pub fn integrate(&mut self, imu_slice: &[ImuData], bias_a: &Vector3, bias_g: &Vector3) -> PreInt {
        let mut prev_ts = imu_slice.first().map_or(0.0, |imu| imu.timestamp as f64 * 1e-9); // Initialize prev_timestamp to the timestamp of the first IMU measurement
        let first_ts = prev_ts.clone();
        let mut ts = 0.0;
        
        let mut dR_ik = SO3::identity(); // Initialize delta_R_j_i to identity
        let mut dv_ik = na::Vector3::zeros(); // Initialize delta_v_j_i to zero
        let mut dp_ik = na::Vector3::zeros(); // Initialize delta_p_j_i to zero

        let mut Jr_bg = na::Matrix3::zeros();
        let mut Jv_bg = na::Matrix3::zeros();
        let mut Jv_ba = na::Matrix3::zeros();
        let mut Jp_bg = na::Matrix3::zeros();
        let mut Jp_ba = na::Matrix3::zeros();

        let mut cov_ik = na::SMatrix::<f64, 9, 9>::zeros(); 
        let mut cov_eta = na::SMatrix::<f64, 6, 6>::zeros();

        // let bias_g = biases.rows(0, 3);
        // let bias_a = biases.rows(3, 3);
        // let bias_g = self.gyro_random_walk; // Placeholder for gyro bias
        // let bias_a = self.accel_random_walk; // Placeholder for accel bias

        for (i, imu) in imu_slice.iter().enumerate() {
            ts = imu.timestamp as f64 * 1e-9; // Convert nanoseconds to seconds
            // print!("Integrating IMU measurement {} at time {:.6} s\n", i, ts);
            let dt = ts - prev_ts;
            if dt <= 0.0 {
                continue;
            }
            prev_ts = ts;

            let omega_unbiased = imu.gyro - bias_g;
            let acc_unbiased = imu.accel - bias_a;

            let dphi = omega_unbiased  * dt; // Angular increment
            let dR_kkp1 = SO3::from_scaled_axis(dphi);
            let J_r = self.right_jacobian_so3(&dphi);
            // print!("dt: {:.6}, dphi: {:?}, Jr:\n{}", dt, dphi, J_r);

            let dv_ikm1 = dv_ik.clone(); // Cache delta_v_i_k before updating it for the next iteration
            dv_ik += dR_ik.rotation_matrix() * acc_unbiased * dt;
            dp_ik += dv_ikm1 * dt + dR_ik.rotation_matrix() * acc_unbiased * dt * dt * 0.5;

            let A = self.construct_A(&dR_kkp1, &dR_ik, &acc_unbiased, dt);
            let B = self.construct_B(&J_r, &dR_ik, dt);
            
            cov_eta = self.cov_eta(dt); // Assuming isotropic noise for simplicity
            // print!("dt: {:.6}, cov_eta:\n{}", dt, cov_eta);
            cov_ik = A * cov_ik * A.transpose() + B * cov_eta * B.transpose(); // Propagate covariance

            Jp_ba += Jv_ba * dt - 0.5 * dR_ik.rotation_matrix() * (dt*dt);
            Jv_ba += -dR_ik.rotation_matrix() * dt;

            let a_hat = skew_symmetric(&acc_unbiased);
            Jp_bg += Jv_bg * dt - 0.5 * dR_ik.rotation_matrix() * a_hat * Jr_bg * (dt*dt);
            Jv_bg += -dR_ik.rotation_matrix() * a_hat * Jr_bg * dt;

            let R_kkp1 = dR_kkp1.rotation_matrix();
            Jr_bg = R_kkp1.transpose() * Jr_bg - J_r * dt;

            dR_ik = SO3::from_quaternion(dR_ik.quaternion() * dR_kkp1.quaternion());
        }

        PreInt {
            dR: dR_ik,
            dv: dv_ik,
            dp: dp_ik,
            cov: cov_ik,
            dt: (ts - first_ts), // Total time from the first to the last IMU measurement
            // bias_g,
            // bias_a,
            Jr_bg: Jr_bg, 
            Jv_bg: Jv_bg,
            Jv_ba: Jv_ba,
            Jp_bg: Jp_bg, 
            Jp_ba: Jp_ba, 
            gyro_random_walk: self.gyro_random_walk,
            accel_random_walk: self.accel_random_walk,
            inv_chol: self.compute_inv_chol(&cov_ik), // Compute square root information matrix
            inv_chol_bias: self.inv_chol_bias(&cov_eta), // Compute square root information matrix for bias
        }
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

    pub fn compute_inv_chol(&self, sigma: &Mat9) -> Mat9 {
        // 1) Symmetrize (numerical hygiene)
        let sigma = 0.5 * (sigma + sigma.transpose());

        // 2) Cholesky: Σ = L L^T
        let sigma_reg = sigma + Mat9::identity() * 1e-6;
        let chol = na::Cholesky::new(sigma_reg);
        let L = chol.as_ref().unwrap().l();

        // 3) sqrt_info = L^{-1}
        L.try_inverse().expect("Failed to invert L")
    }

    pub fn inv_chol_bias(&self, sigma: &na::SMatrix<f64, 6, 6>) -> na::SMatrix<f64, 6, 6> {
        // 1) Symmetrize (numerical hygiene)
        let sigma = 0.5 * (sigma + sigma.transpose());

        // 2) Cholesky: Σ = L L^T
        let sigma_reg = sigma + na::SMatrix::<f64, 6, 6>::identity() * 1e-6;
        let chol = na::Cholesky::new(sigma_reg);
        let L = chol.as_ref().unwrap().l();

        // 3) sqrt_info = L^{-1}
        L.try_inverse().expect("Failed to invert L")
    }

}

fn skew_symmetric(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
        na::Matrix3::new(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
    }