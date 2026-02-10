use nalgebra as na;
use na::{DVector, DMatrix, Vector3, Vector2, Matrix4, UnitQuaternion, Matrix3};
use apex_solver::factors::Factor;
use apex_solver::manifold::se3;

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
}

impl BundleAdjustmentFactor {
    pub fn new(
        observation: Vector2<f64>,
        T_C_B: Matrix4<f64>,
    ) -> Self {
        Self {
            observation,
            T_C_B,
            fixed_pose: None,
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

        (residuals, jacobian_matrix)
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

        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        2 // 2D residual (u, v)
    }
}



#[derive(Debug, Clone)]
pub struct IMUFactor {
    /// Preintegrated IMU measurement: (delta position, delta velocity, delta orientation)
    pub delta_p: Vector3<f64>,
    pub delta_v: Vector3<f64>,
    pub delta_q: UnitQuaternion<f64>,
}

impl Factor for IMUFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // TODO implement IMU factor linearization
        unimplemented!()
    }

    fn get_dimension(&self) -> usize {
        10 // 3 for delta position, 3 for delta velocity, 4 for delta orientation (quaternion)
    }
}