use clap::Id;
use faer::linalg::jacobi;
use nalgebra::{self as na, Rotation3, dvector};
use na::{DVector, DMatrix, Vector3, Vector2, Matrix4, UnitQuaternion, Matrix3};
use apex_solver::factors::Factor;
use apex_solver::manifold::{LieGroup, Tangent, se3, so3};
use rerun::external::arrow::compute;

use crate::imu::piecewise_integration::PreInt;
use crate::optimization::logger::log_imu_linearization;

/// Pinhole projection factor for optimizing 3D point positions from camera observations.
///
/// This factor computes the reprojection error for a 3D point observed in a camera.
/// It optimizes only the 3D point position, with camera pose held fixed.
///
/// - Variables: 3D point in world/camera frame (3 params: x, y, z)
/// - Fixed parameters: Camera pose (T_world_to_camera, 4x4 matrix),
///                     Observation (2D normalized/undistorted)
///
/// The residual is 2D: [u, v] in normalized coordinates
///
/// # Mathematical Formulation
///
/// Given a 3D point `X` and camera pose `T_cam_world`, the residual is:
///
/// ```text
/// r = proj(T_cam_world * X) - obs
/// ```
///
/// where `proj` is the pinhole projection to normalized coordinates: `[x/z, y/z]`
#[derive(Debug, Clone)]
pub struct PinholeProjectionFactor {
    /// Observed 2D point in camera (normalized/undistorted coordinates: x, y)
    pub observation: Vector2<f64>,
    
    /// Transform from world to camera frame (T_world_to_camera, 4x4 matrix)
    pub T_C_W: Matrix4<f64>,
}

impl PinholeProjectionFactor {
    /// Create a new pinhole projection factor.
    ///
    /// # Arguments
    /// * `observation` - Observed 2D point in camera (normalized/undistorted: x, y)
    /// * `t_world_to_camera` - Transform from world to camera frame (4x4 matrix)
    pub fn new(
        observation: Vector2<f64>,
        T_C_W: Matrix4<f64>,
    ) -> Self {
        Self {
            observation,
            T_C_W,
        }
    }

    /// Project a 3D point in camera frame to normalized coordinates (simple pinhole: x/z, y/z).
    fn project_normalized(
        &self,
        point_3d_cam: Vector3<f64>,
    ) -> Vector2<f64> {
        let x = point_3d_cam[0] / point_3d_cam[2];
        let y = point_3d_cam[1] / point_3d_cam[2];
        Vector2::new(x, y)
    }

    /// Compute Jacobian of normalized projection w.r.t. 3D point in camera frame.
    /// For pinhole: [x/z, y/z], so ∂[x/z, y/z]/∂[x, y, z]
    fn jacobian_proj_wrt_point(
        &self,
        point_3d_cam: Vector3<f64>,
    ) -> na::Matrix2x3<f64> {
        let x = point_3d_cam[0];
        let y = point_3d_cam[1];
        let z = point_3d_cam[2];

        // ∂(x/z)/∂x = 1/z, ∂(x/z)/∂y = 0, ∂(x/z)/∂z = -x/z²
        // ∂(y/z)/∂x = 0, ∂(y/z)/∂y = 1/z, ∂(y/z)/∂z = -y/z²
        let inv_z = 1.0 / z;
        let inv_z_sq = inv_z * inv_z;
        
        let mut jac = na::Matrix2x3::zeros();
        jac[(0, 0)] = inv_z;           // ∂(x/z)/∂x
        jac[(0, 1)] = 0.0;             // ∂(x/z)/∂y
        jac[(0, 2)] = -x * inv_z_sq;   // ∂(x/z)/∂z
        jac[(1, 0)] = 0.0;             // ∂(y/z)/∂x
        jac[(1, 1)] = inv_z;           // ∂(y/z)/∂y
        jac[(1, 2)] = -y * inv_z_sq;   // ∂(y/z)/∂z

        jac
    }
}

impl Factor for PinholeProjectionFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // params[0] = 3D point in world frame (3 params: x, y, z)
        assert_eq!(params.len(), 1, "PinholeProjectionFactor requires 1 parameter vector");
        assert_eq!(params[0].len(), 3, "3D point must have 3 parameters");

        let point_world = Vector3::new(params[0][0], params[0][1], params[0][2]);


        // Transform 3D point from world to camera frame
        let R_C_W = self.T_C_W.fixed_view::<3, 3>(0, 0);
        let t_C_W = self.T_C_W.fixed_view::<3, 1>(0, 3);
        //println!("t_C_W: {:?}", t_C_W.to_owned().to_string());
        let point_camera = R_C_W * point_world + t_C_W;

        // Project to normalized coordinates (simple pinhole: x/z, y/z)
        let proj = self.project_normalized(point_camera);

        // Compute residuals (2D: u, v)
        let mut residuals = DVector::zeros(2);
        residuals[0] = proj[0] - self.observation[0];
        residuals[1] = proj[1] - self.observation[1];

        let jacobian_matrix = if compute_jacobian {
            let jac_proj_wrt_point_cam = self.jacobian_proj_wrt_point(point_camera);
            // Chain rule: ∂r/∂point_world = ∂proj/∂point_cam * R_world_to_camera
            let jac_wrt_point = jac_proj_wrt_point_cam * R_C_W;
            
            let mut jac = DMatrix::zeros(2, 3);
            jac.copy_from(&jac_wrt_point);
            Some(jac)
        } else {
            None
        };

        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        2 // 2D residual (u, v)
    }
}


#[inline]
pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// BA factor for optimizing only the translation of a camera pose - development purposes
/// Observation: 2D point
/// Data: Transform from camera to body
/// Variables: Translation of system pose t_B_W, p_W
/// Residual: 2D point - project(T_B_C^{-1} * T_B_W *p_W)
#[derive(Debug, Clone)]
pub struct BundleAdjustmentFactorTranslationOnly {
    /// Observed 2D point in camera (normalized/undistorted coordinates: x, y)
    pub observation: Vector2<f64>,
    
    pub T_C_B: Matrix4<f64>,

    pub fixed_position: Option<Vector3<f64>>,
}

impl BundleAdjustmentFactorTranslationOnly {
    pub fn new(
        observation: Vector2<f64>,
        T_C_B: Matrix4<f64>,
    ) -> Self {
        Self {
            observation,
            T_C_B,
            fixed_position: None,
        }
    }

    pub fn with_fixed_position(mut self, position: Vector3<f64>) -> Self {
        self.fixed_position = Some(position);
        self
    }

    /// Project a 3D point in camera frame to normalized coordinates (simple pinhole: x/z, y/z).
    fn project_normalized(
        &self,
        point_3d_cam: Vector3<f64>,
    ) -> Vector2<f64> {
        let x = point_3d_cam[0] / point_3d_cam[2];
        let y = point_3d_cam[1] / point_3d_cam[2];
        Vector2::new(x, y)
    }

    /// Compute Jacobian of normalized projection w.r.t. 3D point in camera frame.
    /// For pinhole: [x/z, y/z], so ∂[x/z, y/z]/∂[x, y, z]
    fn jacobian_r_wrt_p_C(
        &self,
        point_3d_cam: Vector3<f64>,
    ) -> na::Matrix2x3<f64> {
        let x = point_3d_cam[0];
        let y = point_3d_cam[1];
        let z = point_3d_cam[2];

        // ∂(x/z)/∂x = 1/z, ∂(x/z)/∂y = 0, ∂(x/z)/∂z = -x/z²
        // ∂(y/z)/∂x = 0, ∂(y/z)/∂y = 1/z, ∂(y/z)/∂z = -y/z²
        let inv_z = 1.0 / z;
        let inv_z_sq = inv_z * inv_z;
        
        let mut jac = na::Matrix2x3::zeros();
        jac[(0, 0)] = inv_z;           // ∂(x/z)/∂x
        jac[(0, 1)] = 0.0;             // ∂(x/z)/∂y
        jac[(0, 2)] = -x * inv_z_sq;   // ∂(x/z)/∂z
        jac[(1, 0)] = 0.0;             // ∂(y/z)/∂x
        jac[(1, 1)] = inv_z;           // ∂(y/z)/∂y
        jac[(1, 2)] = -y * inv_z_sq;   // ∂(y/z)/∂z

        jac
    }
}

impl Factor for BundleAdjustmentFactorTranslationOnly {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // params[0] = 3D point in world frame (3 params: x, y, z)

        let p_W = Vector3::new(params[0][0], params[0][1], params[0][2]);
        let mut t_B_W : Vector3<f64>;
        if let Some(fixed_position) = self.fixed_position {
            t_B_W = fixed_position.clone();
            assert_eq!(params.len(), 1, "BundleAdjustmentFactorTranslationOnly with fixed position requires 1 parameter vector");
            assert_eq!(params[0].len(), 3, "3D point must have 3 parameters");
        }
        else {
            t_B_W = Vector3::new(params[1][0], params[1][1], params[1][2]);
            assert_eq!(params.len(), 2, "BundleAdjustmentFactorTranslationOnly requires 2 parameter vectors");
            assert_eq!(params[0].len(), 3, "3D point must have 3 parameters");
            assert_eq!(params[1].len(), 3, "Translation must have 3 parameters");
        }

        // Transform 3D point from world to camera frame
        let R_C_B: nalgebra::Matrix<f64, nalgebra::Const<3>, nalgebra::Const<3>, nalgebra::ViewStorage<'_, f64, nalgebra::Const<3>, nalgebra::Const<3>, nalgebra::Const<1>, nalgebra::Const<4>>> = self.T_C_B.fixed_view::<3, 3>(0, 0);
        let t_C_B = self.T_C_B.fixed_view::<3, 1>(0, 3);
        //println!("t_C_W: {:?}", t_C_W.to_owned().to_string());
        let p_C = R_C_B * (p_W + t_B_W) + t_C_B;

        // Project to normalized coordinates (simple pinhole: x/z, y/z)
        let proj = self.project_normalized(p_C);

        // Compute residuals (2D: u, v)
        let mut residuals = DVector::zeros(2);
        residuals[0] = proj[0] - self.observation[0];
        residuals[1] = proj[1] - self.observation[1];

        let jacobian_matrix = if compute_jacobian {
            let jac_r_wrt_p_C = self.jacobian_r_wrt_p_C(p_C); // 2x3
            let jac_r_wrt_p_W = jac_r_wrt_p_C * R_C_B; // 2x3
            
            if self.fixed_position.is_some() {
                let mut jac = DMatrix::zeros(2, 3);
                jac.copy_from(&jac_r_wrt_p_W);
                Some(jac)
            } else {
                let jac_r_wrt_t_B_W = jac_r_wrt_p_C * R_C_B; 
                let mut jac = DMatrix::zeros(2, 6);
                jac.view_mut((0, 0), (2, 3)).copy_from(&jac_r_wrt_p_W);
                jac.view_mut((0, 3), (2, 3)).copy_from(&jac_r_wrt_t_B_W);
                Some(jac)
            }
        } else {
            None
        };
        // log::warn!("Residuals BA factor: {:?}", residuals);
        // log::warn!("Jacobian BA factor: {:?}", jacobian_matrix);
        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        2 // 2D residual (u, v)
    }
}



/// BA factor
/// Observation: 2D point
/// Data: Transform from camera to body (T_C_B)
/// Variables: System pose T_B_W (or t_B_W if rotation is fixed), p_W
/// Residual: 2D point - project(T_C_B * T_B_W * p_W)
#[derive(Debug, Clone)]
pub struct BundleAdjustmentFactor {
    /// Observed 2D point in camera (normalized/undistorted coordinates: x, y)
    pub observation: Vector2<f64>,
    
    /// Transform from body to camera (T_C_B: SE3 transform from B to C)
    pub T_C_B: Matrix4<f64>,

    /// Fixed pose T_B_W (SE3 transform from W to B) if provided, None if pose is optimized
    pub fixed_pose: Option<Matrix4<f64>>,

    // The sqrt information matrix for the residuals, if using a robust loss. If None, assume identity (no weighting).
    pub sqrt_info : Option<na::Matrix2<f64>>,
}

impl BundleAdjustmentFactor {
    pub fn new(
        observation: Vector2<f64>,
        T_C_B: Matrix4<f64>,
        sqrt_info: Option<na::Matrix2<f64>>,
    ) -> Self {
        Self {
            observation,
            T_C_B,
            fixed_pose: None,
            sqrt_info: sqrt_info,
        }
    }

    /// Set a fixed pose T_B_W (SE3 transform from W to B).
    /// When set, the pose is not optimized and only the 3D point is optimized.
    pub fn with_fixed_pose(mut self, T_B_W: Matrix4<f64>) -> Self {
        self.fixed_pose = Some(T_B_W);
        self
    }

    /// Project a 3D point in camera frame to normalized coordinates (simple pinhole: x/z, y/z).
    fn project_normalized(
        &self,
        point_3d_cam: Vector3<f64>,
    ) -> Vector2<f64> {
        let x = point_3d_cam[0] / point_3d_cam[2];
        let y = point_3d_cam[1] / point_3d_cam[2];
        Vector2::new(x, y)
    }

    /// Compute Jacobian of normalized projection w.r.t. 3D point in camera frame.
    /// For pinhole: [x/z, y/z], so ∂[x/z, y/z]/∂[x, y, z]
    fn jacobian_r_wrt_p_C(
        &self,
        point_3d_cam: Vector3<f64>,
    ) -> na::Matrix2x3<f64> {
        let x = point_3d_cam[0];
        let y = point_3d_cam[1];
        let z = point_3d_cam[2];

        // ∂(x/z)/∂x = 1/z, ∂(x/z)/∂y = 0, ∂(x/z)/∂z = -x/z²
        // ∂(y/z)/∂x = 0, ∂(y/z)/∂y = 1/z, ∂(y/z)/∂z = -y/z²
        let inv_z = 1.0 / z;
        let inv_z_sq = inv_z * inv_z;
        
        let mut jac = na::Matrix2x3::zeros();
        jac[(0, 0)] = inv_z;           // ∂(x/z)/∂x
        jac[(0, 1)] = 0.0;             // ∂(x/z)/∂y
        jac[(0, 2)] = -x * inv_z_sq;   // ∂(x/z)/∂z
        jac[(1, 0)] = 0.0;             // ∂(y/z)/∂x
        jac[(1, 1)] = inv_z;           // ∂(y/z)/∂y
        jac[(1, 2)] = -y * inv_z_sq;   // ∂(y/z)/∂z

        jac
    }
}

impl Factor for BundleAdjustmentFactor {
    #![allow(non_snake_case)]
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract 3D point in world frame
        let p_W = Vector3::new(params[0][0], params[0][1], params[0][2]);

        // Extract T_B_W (SE3 transform from W to B)
        let (R_B_W, t_B_W) = if let Some(T_B_W) = self.fixed_pose {
            assert_eq!(params.len(), 1, "BundleAdjustmentFactor with fixed pose requires 1 parameter vector");
            assert_eq!(params[0].len(), 3, "3D point must have 3 parameters");
            (
                T_B_W.fixed_view::<3, 3>(0, 0).into_owned(),
                T_B_W.fixed_view::<3, 1>(0, 3).into_owned(),
            )
        } else {
            assert_eq!(params.len(), 2, "BundleAdjustmentFactor requires 2 parameter vectors");
            assert_eq!(params[0].len(), 3, "3D point must have 3 parameters");
            assert_eq!(params[1].len(), 7, "System pose must have 7 parameters (tx, ty, tz, qw, qx, qy, qz)");
            let T_B_W = se3::SE3::from(params[1].clone());
            (
                T_B_W.rotation_so3().rotation_matrix().into(),
                T_B_W.translation().into(),
            )
        };

        // Pre-compute camera transform components (reused in jacobian)
        let R_C_B = self.T_C_B.fixed_view::<3, 3>(0, 0);
        let t_C_B = self.T_C_B.fixed_view::<3, 1>(0, 3);
        
        // Transform: p_W -> p_B -> p_C
        let p_B = R_B_W * p_W + t_B_W;
        let p_C = R_C_B * p_B + t_C_B;

        //println!("p_C: {:?}", p_C.to_owned().to_string());
        // Check cheirality of the 3D point
        // TODO fix this because it does not help
        if p_C.z <= 0.0 {
            // log::warn!("3D point is behind the camera, skipping optimization");
            let residuals = DVector::from_vec(vec![
                1e6, 1e6]);
            if self.fixed_pose.is_some() {
                // Only optimize 3D point
                let jac = DMatrix::zeros(2, 3);
                return (residuals, Some(jac));
            } else {
                let jac = DMatrix::zeros(2, 9);
                return (residuals, Some(jac));
            }
        }

        // Project and compute residuals
        let proj = self.project_normalized(p_C);
        let residuals = DVector::from_vec(vec![
            proj[0] - self.observation[0],
            proj[1] - self.observation[1],
        ]);

        let jacobian_matrix = if compute_jacobian {
            let jac_proj = self.jacobian_r_wrt_p_C(p_C); // 2x3
            
            // Pre-compute: jac_proj * R_C_B (reused for both translation and rotation jacobians)
            let jac_proj_R_C_B = jac_proj * R_C_B; // 2x3
            
            // ∂r/∂p_W = jac_proj * R_C_B * R_B_W
            let jac_r_wrt_p_W = jac_proj_R_C_B * &R_B_W; // 2x3
            
            if self.fixed_pose.is_some() {
                // Only optimize 3D point
                let mut jac = DMatrix::zeros(2, 3);
                jac.copy_from(&jac_r_wrt_p_W);
                Some(jac)
            } else {
                // TODO fix notation of AI-generated comments to match paper
                // Optimize both 3D point and pose: [∂r/∂p_W (2x3) | ∂r/∂T_B_W (2x6)]
                // where T_B_W SE3 tangent = [t; ω] (3 translation + 3 rotation)
                
                // Compute rotation jacobian: ∂r/∂ω = jac_proj * R_C_B * (-R_B_W * [p_W]×)
                let p_W_skew = skew_symmetric(&p_W);
                let jac_r_wrt_rot = jac_proj_R_C_B * (-&R_B_W * p_W_skew); // 2x3
                
                // Translation jacobian: ∂r/∂t = jac_proj * R_C_B * R_B_W (same as ∂r/∂p_W)
                // Concatenate: [∂r/∂p_W (2x3) | ∂r/∂t (2x3) | ∂r/∂ω (2x3)] = [2x3 | 2x6]
                let mut jac = DMatrix::zeros(2, 9);
                jac.view_mut((0, 0), (2, 3)).copy_from(&jac_r_wrt_p_W);  // ∂r/∂p_W
                jac.view_mut((0, 3), (2, 3)).copy_from(&jac_r_wrt_p_W);  // ∂r/∂t
                jac.view_mut((0, 6), (2, 3)).copy_from(&jac_r_wrt_rot);  // ∂r/∂ω
                Some(jac)
            }
        } else {
            None
        };
        let residuals_whitened = if self.sqrt_info.is_some() {
            DVector::from_vec((self.sqrt_info.unwrap() * residuals).as_slice().to_vec())
        } else {
            residuals
        };

        let jacobian_whitened = if let Some(sqrt_info) = self.sqrt_info {
            let sqrt_info_dmat = DMatrix::from_fn(2, 2, |i, j| sqrt_info[(i, j)]);
            jacobian_matrix.map(|jac| sqrt_info_dmat * jac)
        } else {
            jacobian_matrix
        };

        (residuals_whitened, jacobian_whitened)
    }

    fn get_dimension(&self) -> usize {
        2 // 2D residual (u, v)
    }
}




/// PnP factor
/// Observation: 2D point
/// Data: Transform from camera to body (T_C_B), 3D point p_W
/// Variables: System pose T_B_W 
/// Residual: 2D point - project(T_C_B * T_B_W * p_W)
#[derive(Debug, Clone)]
pub struct PnPFactor {
    pub observation: Vector2<f64>,
    pub T_C_B: Matrix4<f64>,
    pub p_W: Vector3<f64>,
}

impl PnPFactor {
    pub fn new(
        observation: Vector2<f64>,
        T_C_B: Matrix4<f64>,
        p_W: Vector3<f64>,
    ) -> Self {
        Self {
            observation,
            T_C_B,
            p_W
        }
    }

    /// Project a 3D point in camera frame to normalized coordinates (simple pinhole: x/z, y/z).
    fn project_normalized(
        &self,
        p_C: Vector3<f64>,
    ) -> Vector2<f64> {
        let x = p_C[0] / p_C[2];
        let y = p_C[1] / p_C[2];
        Vector2::new(x, y)
    }

    /// Compute Jacobian of normalized projection w.r.t. 3D point in camera frame.
    /// For pinhole: [x/z, y/z], so ∂[x/z, y/z]/∂[x, y, z]
    fn jacobian_r_wrt_p_C(
        &self,
        p_C: Vector3<f64>,
    ) -> na::Matrix2x3<f64> {
        let x = p_C[0];
        let y = p_C[1];
        let z = p_C[2];

        // ∂(x/z)/∂x = 1/z, ∂(x/z)/∂y = 0, ∂(x/z)/∂z = -x/z²
        // ∂(y/z)/∂x = 0, ∂(y/z)/∂y = 1/z, ∂(y/z)/∂z = -y/z²
        let inv_z = 1.0 / z;
        let inv_z_sq = inv_z * inv_z;
        
        let mut jac = na::Matrix2x3::zeros();
        jac[(0, 0)] = inv_z;           // ∂(x/z)/∂x
        jac[(0, 1)] = 0.0;             // ∂(x/z)/∂y
        jac[(0, 2)] = -x * inv_z_sq;   // ∂(x/z)/∂z
        jac[(1, 0)] = 0.0;             // ∂(y/z)/∂x
        jac[(1, 1)] = inv_z;           // ∂(y/z)/∂y
        jac[(1, 2)] = -y * inv_z_sq;   // ∂(y/z)/∂z

        jac
    }
}

impl Factor for PnPFactor {
    #![allow(non_snake_case)]
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert_eq!(params.len(), 1, "PnPFactor requires 1 parameter vector");
        assert_eq!(params[0].len(), 7, "System pose must have 7 parameters (tx, ty, tz, qw, qx, qy, qz)");
        let T_B_W = se3::SE3::from(params[0].clone());
        let R_B_W : na::Matrix3<f64> = T_B_W.rotation_so3().rotation_matrix().into();
        let t_B_W : na::Vector3<f64> = T_B_W.translation().into();
        
        // Pre-compute camera transform components (reused in jacobian)
        let R_C_B = self.T_C_B.fixed_view::<3, 3>(0, 0);
        let t_C_B = self.T_C_B.fixed_view::<3, 1>(0, 3);

        // Transform: p_W -> p_B -> p_C
        let p_B = R_B_W * self.p_W + t_B_W;
        let p_C = R_C_B * p_B + t_C_B;
        
        // Project and compute residuals
        let proj = self.project_normalized(p_C);
        let residuals = DVector::from_vec(vec![
            proj[0] - self.observation[0],
            proj[1] - self.observation[1],
        ]);

        let jacobian_matrix = if compute_jacobian {
            let jac_proj = self.jacobian_r_wrt_p_C(p_C); // 2x3
            
            // Pre-compute: jac_proj * R_C_B (reused for both translation and rotation jacobians)
            let jac_proj_R_C_B = jac_proj * R_C_B; // 2x3
            
            // ∂r/∂p_W = jac_proj * R_C_B * R_B_W
            let jac_r_wrt_p_W = jac_proj_R_C_B * &R_B_W; // 2x3
            
            
            // TODO fix notation of AI-generated comments to match paper
            // Optimize both 3D point and pose: [∂r/∂p_W (2x3) | ∂r/∂T_B_W (2x6)]
            // where T_B_W SE3 tangent = [t; ω] (3 translation + 3 rotation)
            
            // Compute rotation jacobian: ∂r/∂ω = jac_proj * R_C_B * (-R_B_W * [p_W]×)
            let p_W_skew = skew_symmetric(&self.p_W);
            let jac_r_wrt_rot = jac_proj_R_C_B * (-&R_B_W * p_W_skew); // 2x3
            
            // Translation jacobian: ∂r/∂t = jac_proj * R_C_B * R_B_W (same as ∂r/∂p_W)
            // Concatenate: [∂r/∂p_W (2x3) | ∂r/∂t (2x3) | ∂r/∂ω (2x3)] = [2x3 | 2x6]
            let mut jac = DMatrix::zeros(2, 6);
            jac.view_mut((0, 0), (2, 3)).copy_from(&jac_r_wrt_p_W);  // ∂r/∂t
            jac.view_mut((0, 3), (2, 3)).copy_from(&jac_r_wrt_rot);  // ∂r/∂ω
            Some(jac)
            
        } else {
            None
        };
        // log::error!("Residuals PnP factor: {:?}", residuals);
        // log::error!("Jacobian PnP factor: {:?}", jacobian_matrix);
        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        2 // 2D residual (u, v)
    }
}

fn set_block_3x3(J: &mut na::SMatrix<f64, 9, 30>, row: usize, col: usize, B: &na::SMatrix<f64, 3, 3>) {
    J.fixed_view_mut::<3,3>(row, col).copy_from(B);
}

fn set_block_3x3_dyn(J: &mut na::SMatrix<f64, 9, 30>, row: usize, col: usize, B: &na::SMatrix<f64,3,3>) {
    J.fixed_view_mut::<3,3>(row, col).copy_from(B);
}

fn right_jacobian_inv(phi: &na::Vector3<f64>) -> na::Matrix3<f64> {
    let a = phi.norm();
    if a < 1e-8 {
        // series: I + 0.5*phi^ + 1/12*(phi^)2 ...
        let ph = skew_symmetric(phi);
        return na::Matrix3::<f64>::identity() + 0.5 * ph + (1.0 / 12.0) * (ph * ph);
    }
    let ph = skew_symmetric(phi);
    let a2 = a * a;
    let cot = (a * 0.5).cos() / (a * 0.5).sin(); // cot(a/2)
    na::Matrix3::<f64>::identity()
        + 0.5 * ph
        + (1.0 / a2) * (1.0 - 0.5 * a * cot) * (ph * ph)
}

fn so3_log(R: &na::Matrix3<f64>) -> na::Vector3<f64> {
    let tr = R[(0,0)] + R[(1,1)] + R[(2,2)];
    let cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < 1e-8 {
        return na::Vector3::zeros();
    }
    let w_hat = (R - R.transpose()) * (0.5 * theta / theta.sin());
    na::Vector3::new(w_hat[(2,1)], w_hat[(0,2)], w_hat[(1,0)])
}

fn right_jacobian_so3(phi: &na::Vector3<f64>) -> na::Matrix3<f64> {
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


#[derive(Debug, Clone)]
pub struct ImuFactor {
    /// Preintegrated IMU measurement: (delta position, delta velocity, delta orientation)
    pub preint: PreInt,
    pub linearized_bias_a: na::Vector3<f64>,
    pub linearized_bias_g: na::Vector3<f64>,
}

impl ImuFactor {
    pub fn new(preint: PreInt, linearized_bias_a: na::Vector3<f64>, linearized_bias_g: na::Vector3<f64>) -> Self {
        Self {
            preint,
            linearized_bias_a,
            linearized_bias_g,
        }
    }
}

#[allow(non_snake_case)]
impl Factor for ImuFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert_eq!(params.len(), 4, "ImuFactor requires 4 parameter vectors");
        assert_eq!(params[0].len(), 7, "System pose must have 7 parameters (tx_i, ty_i, tz_i, qw_i, qx_i, qy_i, qz_i)");
        assert_eq!(params[1].len(), 9, "System vel + bias must have 9 parameters (vx_i, vy_i, vz_i, bax_i, bay_i, baz_i, bgx_i, bgy_i, bgz_i)");
        assert_eq!(params[2].len(), 7, "System pose must have 7 parameters (tx_j, ty_j, tz_j, qw_j, qx_j, qy_j, qz_j)");
        assert_eq!(params[3].len(), 9, "System vel + bias must have 9 parameters (vx_j, vy_j, vz_j, bax_j, bay_j, baz_j, bgx_j, bgy_j, bgz_j)");
        
        let T_B_W_i = se3::SE3::from(params[0].rows(0, 7).clone_owned());
        let T_B_W_j = se3::SE3::from(params[2].rows(0, 7).clone_owned());
        let v_i = params[1].rows(0, 3).clone_owned();
        let v_j = params[3].rows(0, 3).clone_owned();
        let ba_i = params[1].rows(3, 3).clone_owned();
        let ba_j = params[3].rows(3, 3).clone_owned();
        let bg_i = params[1].rows(6, 3).clone_owned();
        let bg_j = params[3].rows(6, 3).clone_owned();

        let idx_r = self.preint.idx_r;
        let idx_v = self.preint.idx_v;
        let idx_p = self.preint.idx_p;
        let idx_ba = self.preint.idx_ba;
        let idx_bg = self.preint.idx_bg;

        let db_a: na::Vector3<f64> = ba_i.clone() - self.linearized_bias_a; 
        let db_g: na::Vector3<f64> = bg_i.clone() - self.linearized_bias_g;

        let g_w = self.preint.gravity;
        
        let R_i = T_B_W_i.rotation_so3();
        let R_j = T_B_W_j.rotation_so3();
        let t_i = T_B_W_i.translation();
        let t_j = T_B_W_j.translation();

        // Compute the relative acceleration and position from i to j, corrected for gravity and bias
        let dt = self.preint.dt;
        let a_v = v_j.clone() - v_i.clone() + g_w.clone() * dt;
        let a_p = t_j - t_i - v_i.clone() * dt + 0.5 * g_w.clone() * dt * dt;

        let dR_dbg = self.preint.jacobian.fixed_view::<3,3>(idx_r, idx_bg).clone();
        let phi_b = dR_dbg * db_g;
        
        let delta_bg = dR_dbg * db_g.clone();
        let delta_bg_dyn: DVector<f64> = dvector![delta_bg.x, delta_bg.y, delta_bg.z]; 

        let dR_corr = self.preint.delta_R.compose(
            &so3::SO3Tangent::from(delta_bg_dyn).exp(None),
            None,
            None
        );

        let dv_dba = self.preint.jacobian.fixed_view::<3,3>(idx_v, idx_ba).clone();
        let dv_dbg = self.preint.jacobian.fixed_view::<3,3>(idx_v, idx_bg).clone();
        let dp_dba = self.preint.jacobian.fixed_view::<3,3>(idx_p, idx_ba).clone();
        let dp_dbg = self.preint.jacobian.fixed_view::<3,3>(idx_p, idx_bg).clone();

        let deleta_v_corr = self.preint.delta_v + dv_dbg * db_g.clone() + dv_dba.clone() * db_a.clone();

        let delta_p_corr = self.preint.delta_p + dp_dbg * db_g + dp_dba.clone() * db_a.clone();

        // Residual w.r.t. rotation
        let residual_dR = dR_corr.inverse(None)
            .compose(&R_i.inverse(None), None, None)
            .compose(&R_j, None, None).log(None);

        let Ri_inv = R_i.inverse(None).rotation_matrix();
        
        let residual_dv = Ri_inv * a_v - deleta_v_corr;
        let residual_dp = Ri_inv * a_p - delta_p_corr;
        let residual_bg = bg_j.clone() - bg_i.clone();
        let residual_ba = ba_j.clone() - ba_i.clone();
    
        // Convert SO3Tangent to DVector
        let residual_dR_vec: DVector<f64> = residual_dR.clone().into();
        
        // Concatenate all residuals into a single fixed-size Vector15
        let mut residuals_fixed = na::SVector::<f64, 15>::zeros();
        residuals_fixed.rows_mut(idx_r, 3).copy_from(&residual_dR_vec);
        residuals_fixed.rows_mut(idx_v, 3).copy_from(&residual_dv);
        residuals_fixed.rows_mut(idx_p, 3).copy_from(&residual_dp);
        residuals_fixed.rows_mut(idx_ba, 3).copy_from(&residual_ba);
        residuals_fixed.rows_mut(idx_bg, 3).copy_from(&residual_bg);
        let residuals_whitened = self.preint.whiten_residual_15(&(residuals_fixed));

        let residuals = DVector::from_vec(residuals_whitened.as_slice().to_vec());

        let jacobian_matrix = if compute_jacobian {
            let mut J = na::SMatrix::<f64, 15, 30>::zeros();

            // Convenience
            let RiT = R_i.rotation_matrix().transpose();
            let RjT = R_j.rotation_matrix().transpose();
            
            let r_dR: Vector3<f64> = Vector3::new(residual_dR.x(), residual_dR.y(), residual_dR.z());
            let Jr_inv = right_jacobian_inv(&r_dR);

            let d_rR_d_phi_i = -Jr_inv * (RjT * R_i.rotation_matrix());
            let d_rR_d_phi_j = Jr_inv;

            // Fill blocks
            J.fixed_view_mut::<3,3>(idx_r, idx_r).copy_from(&d_rR_d_phi_i); // phi_i
            J.fixed_view_mut::<3,3>(idx_r, idx_r + 15).copy_from(&d_rR_d_phi_j); // phi_j

            // Gyro bias block
            let J_br = right_jacobian_so3(&phi_b);
            let d_rR_d_bg = -Jr_inv * residual_dR.exp(None).rotation_matrix().transpose() * J_br * dR_dbg;
            J.fixed_view_mut::<3,3>(idx_r, idx_bg).copy_from(&(d_rR_d_bg));

            // === Velocity rows [3..6) ===
            let d_rv_d_phi_i = skew_symmetric(&(RiT * a_v)); // (Ri^T a_v)^wedge  [file:1]
            J.fixed_view_mut::<3,3>(idx_v, idx_r).copy_from(&d_rv_d_phi_i); // phi_i
            J.fixed_view_mut::<3,3>(idx_v, idx_v).copy_from(&(-RiT));       // v_i
            J.fixed_view_mut::<3,3>(idx_v, idx_v + 15).copy_from(&RiT);     // v_j
            J.fixed_view_mut::<3,3>(idx_v, idx_ba).copy_from(&(-dv_dba));   // ba_i
            J.fixed_view_mut::<3,3>(idx_v, idx_bg).copy_from(&(-dv_dbg));   // bg_i

            // === Translation rows [6..9) ===
            // let a_p = t_j - t_i - v_i.clone() * dt + 0.5 * g_w.clone() * dt * dt;
            let d_rp_d_phi_i = skew_symmetric(&(RiT * a_p)); // (Ri^T a_p)^wedge  [file:1]
            J.fixed_view_mut::<3,3>(idx_p, idx_r).copy_from(&d_rp_d_phi_i);          // phi_i
            J.fixed_view_mut::<3,3>(idx_p, idx_v).copy_from(&(-RiT * dt));          // v_i
            J.fixed_view_mut::<3,3>(idx_p, idx_p).copy_from(&(-na::Matrix3::identity()));  // p_i
            J.fixed_view_mut::<3,3>(idx_p, idx_p + 15).copy_from(&(RiT * R_j.rotation_matrix()));          // p_j
            J.fixed_view_mut::<3,3>(idx_p, idx_ba).copy_from(&(-dp_dba));           // ba_i
            J.fixed_view_mut::<3,3>(idx_p, idx_bg).copy_from(&(-dp_dbg));           // bg_i
            // === Bias rows [9..15) ===
            J.fixed_view_mut::<3,3>(idx_ba, idx_ba).copy_from(&(-na::Matrix3::identity()));   // ba_i
            J.fixed_view_mut::<3,3>(idx_ba, idx_ba + 15).copy_from(&na::Matrix3::identity());    // ba_j
            J.fixed_view_mut::<3,3>(idx_bg, idx_bg).copy_from(&(-na::Matrix3::identity()));  // bg_i
            J.fixed_view_mut::<3,3>(idx_bg, idx_bg + 15).copy_from(&na::Matrix3::identity());   // bg_j

            Some(J)
        } else {
            None
        };

        let jacobian_matrix = if let Some(jac) = jacobian_matrix {
            let jac_static = na::SMatrix::<f64, 15, 30>::from_iterator(jac.iter().cloned());
            let jacobian_matrix_whitened = self.preint.whiten_jacobian_15(&(jac_static));

            // Convert back to DMatrix
            if jacobian_matrix_whitened.max() > 1e8 || jacobian_matrix_whitened.min() < -1e8 {
                log::error!("Numerical unstable in preintegration whitening, large values in whitened jacobian: max {}, min {}", jacobian_matrix_whitened.max(), jacobian_matrix_whitened.min());
            }

            Some(DMatrix::from_iterator(15, 30, jacobian_matrix_whitened.iter().cloned()))
        } else {
            None
        };
        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        15 // 3 for delta position, 3 for delta velocity, 3 for delta orientation (quaternion), 6 for bias residuals (3 accel bias, 3 gyro bias)
    }
}