use std::collections::VecDeque;
use std::ops::AddAssign;
use apex_solver::optimizer::SolverResult;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::manifold::ManifoldType;
use apex_solver::core::problem::{Problem, VariableEnum};
use apex_solver::core::loss_functions::HuberLoss;
use faer::rand::rand_core::le;
use rerun::external::re_log_encoding::external::lz4_flex::frame;
use std::collections::HashMap;
use nalgebra as na;
use na::{DVector, UnitQuaternion};
use crate::datasets::CameraModelType;
use crate::imu::piecewise_integration::ImuPipeline;
use crate::optimization::factors::{BundleAdjustmentFactor, PnPFactor, ImuFactor};
use crate::optimization::observer::TerminalObserver;
use crate::estimator::Frame;
use crate::types::{Matrix3x3, Matrix4x4, Vector3};
use crate::feature_tracker::Feature;

/// Sliding window of keyframes for bundle adjustment optimization.
/// 
/// Maintains a fixed-size window of keyframes and manages the optimization
/// of poses and 3D points across these frames.
#[derive(Debug)]
pub struct SlidingWindow {
    /// Maximum number of keyframes in the sliding window.
    max_frames: usize,
    
    /// Current keyframes in the sliding window (ordered by insertion time, oldest first).
    keyframes: VecDeque<Frame>,

    /// Map points stored by feature ID: HashMap<feature_id, [x, y, z]>
    pub map_points: HashMap<usize, [f32; 3]>,

    /// Flag to indicate if the sliding window is using IMU with enough keyframes for optimization
    pub use_imu: bool,

}

impl SlidingWindow {
    #![allow(non_snake_case)]
    /// Create a new sliding window with the specified maximum number of frames.
    /// 
    /// # Arguments
    /// * `max_frames` - Maximum number of keyframes to keep in the window (default: 16)
    pub fn new(max_frames: usize) -> Self {
        Self {
            max_frames,
            keyframes: VecDeque::with_capacity(max_frames),
            map_points: HashMap::new(),
            use_imu: false,
        }
    }

    /// Create a new sliding window with the default size of 16 frames.
    pub fn default() -> Self {
        Self::new(8)
    }

    /// Add a keyframe to the sliding window.
    /// 
    /// If the window is full, the oldest frame (by frame_id) will be removed
    /// to make room for the new frame. Only keyframes should be added.
    /// 
    /// # Arguments
    /// * `frame` - The keyframe to add
    /// 
    /// # Returns
    /// `true` if the frame was added, `false` if it was rejected (e.g., not a keyframe)
    pub fn add_frame(&mut self, frame: Frame)  -> bool {
        // Only accept keyframes
        if !frame.is_keyframe {
            log::warn!(
                "[SlidingWindow] Attempted to add non-keyframe (frame_id: {})",
                frame.frame_id
            );
            return false;
        }

        // Remove oldest frame if window is full (FIFO - first in, first out)
        if self.keyframes.len() >= self.max_frames {
            if let Some(removed_frame) = self.keyframes.pop_front() {
                log::debug!(
                    "[SlidingWindow] Removed oldest frame (frame_id: {}) to make room for new keyframe",
                    removed_frame.frame_id
                );
            }
        }
        let frame_id = frame.frame_id;

        // Add frame to sliding window
        self.keyframes.push_back(frame);
        log::debug!(
            "[SlidingWindow] Added keyframe (frame_id: {}), window size: {}/{}",
            frame_id,
            self.keyframes.len(),
            self.max_frames
        );

        true
    }

    pub fn update_gravity(&mut self, gravity: Vector3) {
        for frame in self.keyframes.iter_mut() {
            if let Some(imu_preintegration) = &mut frame.imu_preintegration {
                imu_preintegration.gravity = gravity;
            }
        }
    }

    pub fn set_initial_rotation(&mut self, init_rotation: na::Matrix3<f64>) {
        self.keyframes[0].state.T_W_B.fixed_view_mut::<3, 3>(0, 0).copy_from(&init_rotation);
    }

    /// Get the current number of keyframes in the window.
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Check if the sliding window is empty.
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Check if the sliding window is full.
    pub fn is_full(&self) -> bool {
        self.keyframes.len() >= self.max_frames
    }

    /// Get a reference to a specific keyframe by index.
    pub fn get_frame(&self, index: usize) -> Option<&Frame> {
        self.keyframes.get(index)
    }

    pub fn get_keyframe_poses(&self) -> Vec<Matrix4x4> {
        // Return poses of body in world frame
        self.keyframes.iter().map(|f| f.state.T_W_B).collect()
    }

    pub fn get_keyframe_velocities(&self) -> Option<Vec<Vector3>> {
        if self.keyframes.is_empty() {
            None
        } else {
            Some(self.keyframes.iter().map(|f| f.state.velocity).collect())
        }
    }

    pub fn get_keyframe_accel_bias(&self) -> Option<Vec<Vector3>> {
        if self.keyframes.is_empty() {
            None
        } else {
            Some(self.keyframes.iter().map(|f| f.state.accel_bias).collect())
        }
    }

    pub fn get_keyframe_gyro_bias(&self) -> Option<Vec<Vector3>> {
        if self.keyframes.is_empty() {
            None
        } else {
            Some(self.keyframes.iter().map(|f| f.state.gyro_bias).collect())
        }
    }

    pub fn get_angular_velocities(&self) -> Option<Vec<Vector3>> {
        if self.keyframes.is_empty() {
            None
        } else {
            Some(self.keyframes.iter().map(|f| f.state.angular_velocity).collect())
        }
    }

    /// Clear all keyframes from the sliding window.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        log::debug!("[SlidingWindow] Cleared all keyframes");
    }

    fn build_solver_config(&self) -> LevenbergMarquardtConfig {
        LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseSchurComplement)
            .with_schur_variant(SchurVariant::Sparse)
            .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal)
            .with_max_iterations(20)
            .with_cost_tolerance(1e-6)
            .with_parameter_tolerance(1e-9)
            .with_jacobi_scaling(true)
            .with_damping(1e3)
            // .with_trust_region(0.2, 0.8, 0.1)
    }

    pub fn check_sliding_window_size_for_optimization(&self) -> Result<bool, std::io::Error> {
        if self.keyframes.is_empty() {
            log::warn!("[SlidingWindow] Cannot optimize: window is empty");
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Window is empty"));
        }

        if self.keyframes.len() < self.max_frames {
            log::warn!(
                "[SlidingWindow] Cannot optimize: need {} keyframes, have {}",
                self.max_frames,self.keyframes.len()
            );
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Need more keyframes"));
        }

        log::debug!(
            "[SlidingWindow] Starting bundle adjustment optimization with {} keyframes",
            self.keyframes.len()
        );
    
        return Ok(true);
    }

    pub fn optimize(&mut self) -> Result<bool, std::io::Error> {
        self.check_sliding_window_size_for_optimization()?;

        // Save current state before optimization for potential rollback
        let saved_keyframe_poses: Vec<Matrix4x4> = self.keyframes.iter()
            .map(|f| f.state.T_W_B)
            .collect();
        let saved_map_points = self.map_points.clone();

        // Initialize problem and solver
        let mut problem = Problem::new();
        let mut solver = LevenbergMarquardt::with_config(self.build_solver_config());
        let mut initial_values = HashMap::new();
        // solver.add_observer(TerminalObserver::new());
        
        // Initialize maps for tracking and counting observations
        let mut map_feature_to_landmark: HashMap<usize, String> = HashMap::new();
        let mut landmark_observation_count_left: HashMap<String, usize> = HashMap::new();
        let mut landmark_observation_count_right: HashMap<String, usize> = HashMap::new();

        // Fetch transforms between cameras and body
        let T_Cl_B = self.keyframes.front().unwrap().state.T_B_Cl.try_inverse().expect("T_B_Cl should be invertible");
        let T_Cr_B = self.keyframes.front().unwrap().state.T_B_Cr.try_inverse().expect("T_B_Cr should be invertible");

        // Count observations for each landmark across all frames, separately for left and right cameras
        for frame in self.keyframes.iter() {
            // Count left camera observations
            for feat in frame.left_features.iter() {
                let feature_id = feat.feature_id;
                
                // Get or create landmark variable name
                let lm_var = map_feature_to_landmark
                    .entry(feature_id)
                    .or_insert_with(|| format!("LM_{}", feature_id))
                    .clone();
                
                // Increment left camera observation count
                *landmark_observation_count_left.entry(lm_var.clone()).or_insert(0) += 1;
            }
            
            // Count right camera observations
            for feat in frame.right_features.iter() {
                let feature_id = feat.feature_id;
                
                // Get or create landmark variable name
                let lm_var = map_feature_to_landmark
                    .entry(feature_id)
                    .or_insert_with(|| format!("LM_{}", feature_id))
                    .clone();
                
                // Increment right camera observation count
                *landmark_observation_count_right.entry(lm_var.clone()).or_insert(0) += 1;
            }
        }

        let mut prev_kf_var = String::new();
        let mut prev_vb_var = String::new();


        let focal_length = if let Some(frame) = self.keyframes.front() {
            // Pattern match to get intrinsics directly from the camera model
            match &frame.left_cam {
                CameraModelType::OpenCV5(cam) => {
                    // For OpenCV5, intrinsics are in the DVector, typically [fx, fy, cx, cy, k1, k2, p1, p2, k3]
                    // Here we assume fx is the first element
                    cam.fx
                }
                CameraModelType::EUCM(cam) => {
                    cam.fx
                }
            }
        } else {
            500.0 // Default fallback value
        };
        let sqrt_info = if self.use_imu {
            na::Matrix2::identity() * focal_length / 1.5
        } else {
            na::Matrix2::identity()
        };

        // Add factors
        for (id_frame, frame) in self.keyframes.iter().enumerate() {
            // Add KF poses
            let kf_var = format!("KF_{}", id_frame);
            let T_B_W = frame.state.T_W_B.try_inverse().expect("T_W_B should be invertible");
            let t_B_W = T_B_W.fixed_view::<3, 1>(0, 3);
            let R_B_W = Matrix3x3::from(T_B_W.fixed_view::<3, 3>(0, 0));
            let q_B_W = UnitQuaternion::from_matrix(&R_B_W);
            let se3_data = DVector::from_vec(vec![
                t_B_W.x, t_B_W.y, t_B_W.z, q_B_W.w, q_B_W.i, q_B_W.j, q_B_W.k,
            ]);

            // println!("KF_{} initial pose: {:?}", frame.frame_id, se3_data);
            initial_values.insert(kf_var.clone(), (ManifoldType::SE3, se3_data.cast::<f64>()));

            // Add velocity and bias variables for each keyframe
            let v_bias = DVector::from_vec(vec![
                frame.state.velocity.x, frame.state.velocity.y, frame.state.velocity.z,
                frame.state.accel_bias.x, frame.state.accel_bias.y, frame.state.accel_bias.z,
                frame.state.gyro_bias.x, frame.state.gyro_bias.y, frame.state.gyro_bias.z,
            ]);
            
            // Process features from both cameras
            let camera_features = [
                (&frame.left_features, T_Cl_B),
                (&frame.right_features, T_Cr_B),
            ];

            let right_by_id: &HashMap<usize, &Feature> = &frame.right_features
                .iter()
                .map(|f| (f.feature_id, f))
                .collect();
            
            if id_frame > 0 && frame.imu_preintegration.is_some() && self.use_imu && frame.imu_preintegration.clone().unwrap().dt < 5.0 {
                panic!("IMU should be disabled!");
                let vb_var = format!("vb_{}", id_frame);
                let vb_var_prev = format!("vb_{}", id_frame - 1);
                let kf_var_prev = format!("KF_{}", id_frame - 1);
                
                initial_values.insert(vb_var.clone(), (ManifoldType::RN, v_bias.cast::<f64>()));

                let imu_factor = ImuFactor::new(
                    frame.imu_preintegration.as_ref().unwrap().clone(),
                    frame.imu_preintegration.as_ref().unwrap().linearized_ba.clone(),
                    frame.imu_preintegration.as_ref().unwrap().linearized_bg.clone(),
                );
                
                let var_names: Vec<&str> = vec![
                    &kf_var_prev, // Previous keyframe pose
                    &vb_var_prev, // Previous keyframe velocity and biases
                    &kf_var,     // Current keyframe pose
                    &vb_var,     // Current keyframe velocity and biases
                ];
                let huber_loss = HuberLoss::new(2.0).unwrap();
                problem.add_residual_block(&var_names, 
                    Box::new(imu_factor), Some(Box::new(huber_loss)));
            } else if id_frame > 0 && frame.imu_preintegration.is_some() && self.use_imu {
                log::warn!(
                    "[SlidingWindow] Skipping IMU factor for frame {} due to long preintegration dt = {:.3}s",
                    frame.frame_id,
                    frame.imu_preintegration.as_ref().unwrap().dt
                );
            };

            // for left_feat in frame.left_features.iter() {
            //     let Some(right_feat) = right_by_id.get(&left_feat.feature_id) else {
            //         continue; // Skip if no corresponding right feature
            //     };
            //     if self.map_points.contains_key(&left_feat.feature_id) {
            //         continue; // Skip if already in map points (will be added as factor later)
            //     }

            //     if let Some(p_w) =
            // }

            // TODO refactor the loop, quick and dirty check for left cam
            for (i, (features, T_C_B)) in camera_features.iter().enumerate() {
            // for left_observation in frame.left_features.iter() {
                for feat in features.iter() {
                    let feature_id = feat.feature_id;
                    let lm_var = map_feature_to_landmark
                        .get(&feature_id)
                        .cloned()
                        .unwrap_or_else(|| format!("LM_{}", feature_id));
                    
                    // Only process landmarks that are seen at least once in BOTH cameras (stereo constraint)
                    let count_left = landmark_observation_count_left.get(&lm_var).copied().unwrap_or(0);
                    let count_right = landmark_observation_count_right.get(&lm_var).copied().unwrap_or(0);
                    
                    if count_left > 0 && count_right > 0 {
                        // Create initial value for landmark if not already present
                        initial_values.entry(lm_var.clone()).or_insert_with(|| {
                            let data = if let Some(&last_pos) = self.map_points.get(&feature_id) {
                                DVector::from_vec(vec![
                                    last_pos[0] as f64,
                                    last_pos[1] as f64,
                                    last_pos[2] as f64,
                                ])
                            } else {
                                // Default initialization if not in map_points
                                let right_feat = right_by_id.get(&feature_id).expect("Feature should be observed in both cameras");
                                let p_B = self.triangulate_stereo(feat, &T_Cl_B, right_feat, &T_Cr_B).unwrap_or_else(|| {
                                    // Fallback: project feature at default depth
                                    let p_C = Vector3::new(feat.undistorted_coord[0] as f64, feat.undistorted_coord[1] as f64, 2.0);
                                    let T_B_C = T_C_B.try_inverse().expect("T_C_B should be invertible");
                                    let (R_B_C, t_B_C) = (
                                        T_B_C.fixed_view::<3, 3>(0, 0).into_owned(),
                                        T_B_C.fixed_view::<3, 1>(0, 3).into_owned(),
                                    );
                                    R_B_C * p_C + t_B_C
                                });
                                let (R_W_B, t_W_B) = (
                                        frame.state.T_W_B.fixed_view::<3, 3>(0, 0).into_owned(),
                                        frame.state.T_W_B.fixed_view::<3, 1>(0, 3).into_owned(),
                                    );
                                
                                let p_W = R_W_B * p_B  + t_W_B;
                                DVector::from_vec(vec![p_W.x, p_W.y, p_W.z])
                            };
                            (ManifoldType::RN, data)
                        });
                        
                        // Create camera projection factor
                        let mut factor = BundleAdjustmentFactor::new(
                            na::Vector2::new(feat.undistorted_coord[0], feat.undistorted_coord[1]).cast::<f64>(),
                            *T_C_B,
                            Some(sqrt_info),
                        );
                        
                        // Fix pose for first frame
                        if id_frame == 0 {
                            // Factor expects T_B_W (Body-from-World), but frame.state.T_W_B is World-from-Body
                            let T_B_W = frame.state.T_W_B.try_inverse().expect("T_W_B should be invertible");
                            factor = factor.with_fixed_pose(T_B_W);
                        }
                        
                        // Determine variable names based on frame
                        let var_names: Vec<&str> = if id_frame == 0 {
                            vec![&lm_var]
                        } else {
                            vec![&lm_var, &kf_var]
                        };
                        
                        // Add residual block with Huber loss
                        let huber_loss = HuberLoss::new(2.0).unwrap();
                        problem.add_residual_block(&var_names, Box::new(factor), Some(Box::new(huber_loss)));
                    }
                }
            }
        }

        let num_residuals = problem.num_residual_blocks();
        let num_variables = initial_values.len();
        
        log::debug!("Added SE3 and R3 variables, now {} variables total, {} residual blocks", num_variables, num_residuals);

        // Validate problem before optimization
        if num_residuals < 6 {
            log::warn!("[SlidingWindow] Too few residuals ({}), skipping optimization", num_residuals);
            return Ok(false);
        }
        
        // Check if we have enough constraints (roughly: need at least 2 residuals per variable for well-posed problem)
        if num_residuals < num_variables {
            log::warn!("[SlidingWindow] Underconstrained problem: {} residuals < {} variables, skipping optimization", 
                      num_residuals, num_variables);
            return Ok(false);
        }

        // Initialize variables in the problem
        problem.initialize_variables(&initial_values);

        // Try optimization with Schur complement first
        let opt_result = match solver.optimize(&problem, &initial_values) {
            Ok(result) => result,
            Err(e) => {
                // Check if it's a linear solve failure (singular matrix)
                let error_str = format!("{:?}", e);
                if error_str.contains("LinearSolveFailed") || error_str.contains("Singular matrix") {
                    log::warn!("[SlidingWindow] Schur complement failed with singular matrix, trying fallback solver (SparseCholesky)");
                    
                    // Create fallback solver with direct Cholesky
                    let mut fallback_solver = LevenbergMarquardt::with_config(
                        LevenbergMarquardtConfig::new()
                            .with_linear_solver_type(LinearSolverType::SparseCholesky)
                            .with_max_iterations(20)
                            .with_cost_tolerance(1e-6)
                            .with_parameter_tolerance(1e-9)
                            .with_jacobi_scaling(false)
                    );
                    
                    match fallback_solver.optimize(&problem, &initial_values) {
                        Ok(result) => {
                            log::debug!("[SlidingWindow] Fallback solver succeeded");
                            result
                        },
                        Err(e2) => {
                            log::error!("[SlidingWindow] Both Schur complement and fallback solver failed: {:?} - reverting to previous state", e2);
                            self.revert_to_saved_state(&saved_keyframe_poses, &saved_map_points);
                            return Ok(false);
                        }
                    }
                } else {
                    // Other optimization errors - revert to saved state
                    log::error!("[SlidingWindow] Optimization error: {:?} - reverting to previous state", e);
                    self.revert_to_saved_state(&saved_keyframe_poses, &saved_map_points);
                    return Ok(false);
                }
            }
        };
        
        // Check if optimization was successful based on status
        let is_successful = self.is_optimization_successful(&opt_result);
        
        if is_successful {
            // Process successful optimization result
            self.process_optimization_result(&opt_result);
            log::debug!(
                "[SlidingWindow] Optimization successful. Initial cost: {:.3}, final cost: {:.3}",
                opt_result.initial_cost,
                opt_result.final_cost
            );
            Ok(true)
        } else {
            // Optimization failed - revert to saved state
            log::warn!("[SlidingWindow] Optimization failed (status: {:?}) - reverting to previous state", opt_result.status);
            self.revert_to_saved_state(&saved_keyframe_poses, &saved_map_points);
            Ok(false)
        }
    }

    /// Check if optimization result indicates success
    fn is_optimization_successful(&self, opt_result: &SolverResult<HashMap<String, VariableEnum>>) -> bool {
        matches!(
            &opt_result.status,
            apex_solver::optimizer::OptimizationStatus::Converged
                | apex_solver::optimizer::OptimizationStatus::CostToleranceReached
                | apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached
                | apex_solver::optimizer::OptimizationStatus::GradientToleranceReached
                | apex_solver::optimizer::OptimizationStatus::TrustRegionRadiusTooSmall
                | apex_solver::optimizer::OptimizationStatus::MinCostThresholdReached
        )
    }

    /// Revert keyframe poses and map points to saved state
    fn revert_to_saved_state(
        &mut self,
        saved_keyframe_poses: &[Matrix4x4],
        saved_map_points: &HashMap<usize, [f32; 3]>,
    ) {
        // Restore keyframe poses
        for (i, frame) in self.keyframes.iter_mut().enumerate() {
            if i < saved_keyframe_poses.len() {
                frame.state.T_W_B = saved_keyframe_poses[i];
            }
        }
        
        // Restore map points
        self.map_points.clear();
        self.map_points.extend(saved_map_points.iter().map(|(k, v)| (*k, *v)));
        
        log::debug!("[SlidingWindow] Reverted {} keyframe poses and {} map points", 
                   saved_keyframe_poses.len(), saved_map_points.len());
    }

    fn process_optimization_result(&mut self, opt_result: &SolverResult<HashMap<String, VariableEnum>>) {
         // TODO: handle error properly
        
         // Determine convergence status accurately
        let (status, convergence_reason) = match &opt_result.status {
            apex_solver::optimizer::OptimizationStatus::Converged => {
                ("CONVERGED", "Converged".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::CostToleranceReached => {
                ("CONVERGED", "CostTolerance".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached => {
                ("CONVERGED", "ParameterTolerance".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::GradientToleranceReached => {
                ("CONVERGED", "GradientTolerance".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::TrustRegionRadiusTooSmall => {
                ("CONVERGED", "TrustRegionRadiusTooSmall".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::MinCostThresholdReached => {
                ("CONVERGED", "MinCostThresholdReached".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::MaxIterationsReached => {
                ("NOT_CONVERGED", "MaxIterations".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::Timeout => {
                ("NOT_CONVERGED", "Timeout".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::NumericalFailure => {
                ("NOT_CONVERGED", "NumericalFailure".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::IllConditionedJacobian => {
                ("NOT_CONVERGED", "IllConditionedJacobian".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::InvalidNumericalValues => {
                ("NOT_CONVERGED", "InvalidNumericalValues".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::UserTerminated => {
                ("NOT_CONVERGED", "UserTerminated".to_string())
            }
            apex_solver::optimizer::OptimizationStatus::Failed(msg) => {
                ("NOT_CONVERGED", format!("Failed:{}", msg))
            }
        };
        log::debug!("[SlidingWindow] Optimization status: {}, convergence_reason: {}", status, convergence_reason);

        // Update map_points and keyframe poses with optimized values
        self.map_points.clear();
        opt_result.parameters.iter().for_each(|(var_name, value)| {
            // Update map points
            if let Some(feature_id_str) = var_name.strip_prefix("LM_") {
                if let Ok(feature_id) = feature_id_str.parse::<usize>() {
                    let vec = value.to_vector();
                    // TODO check if points are estimated at obviously wrong locations (negative depths, etc.)
                    self.map_points.insert(feature_id, [vec[0] as f32, vec[1] as f32, vec[2] as f32]);
                }
            }
            // Update keyframe poses
            else if let Some(frame_id_str) = var_name.strip_prefix("KF_") {
                if let Ok(frame_id) = frame_id_str.parse::<i32>() {
                    let mat = apex_solver::manifold::se3::SE3::from(value.to_vector()).matrix();
                    // println!("KF_{} optimized pose: {:?}", frame_id, mat);
                    self.keyframes.get_mut(frame_id as usize).unwrap().state.T_W_B = mat.try_inverse().expect("T_W_B should be invertible");
                }
            }
            // Update velocity and bias if needed
             else if let Some(frame_id_str) = var_name.strip_prefix("vb_") {
                if let Ok(frame_id) = frame_id_str.parse::<i32>() {
                    let vec = value.to_vector();
                    let velocity = na::Vector3::new(vec[0], vec[1], vec[2]);
                    let accel_bias = na::Vector3::new(vec[3], vec[4], vec[5]);
                    let gyro_bias = na::Vector3::new(vec[6], vec[7], vec[8]);
                    let frame = self.keyframes.get_mut(frame_id as usize).unwrap();
                    frame.state.velocity = velocity;
                    frame.state.accel_bias = accel_bias;
                    frame.state.gyro_bias = gyro_bias;
                }
            }
        });

    }

    /// Track the motion of the system by solving a PnP-like problem
    /// The map points and existing keyframes are kept constant, only the new frame is optimized
    pub fn track_motion(&mut self, frame: &Frame) -> Result<Option<Matrix4x4>, std::io::Error> {

        // Create a new problem and solver
        let mut problem = Problem::new();
        let mut solver = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_linear_solver_type(LinearSolverType::SparseCholesky)
                .with_max_iterations(10)
                .with_cost_tolerance(1e-6)
                .with_parameter_tolerance(1e-9)
                .with_jacobi_scaling(false)
        );
        let mut initial_values = HashMap::new();
        // solver.add_observer(TerminalObserver::new());

        // Add variable for the new frame
        // Only the new frame is optimized and it's use_imu from the last keyframe
        let kf_var = format!("F");
        let T_B_W = self.keyframes.back().unwrap().state.T_W_B.try_inverse().expect("T_W_B should be invertible");
        let t_B_W = T_B_W.fixed_view::<3, 1>(0, 3);
        let R_B_W = Matrix3x3::from(T_B_W.fixed_view::<3, 3>(0, 0));
        let q_B_W = UnitQuaternion::from_matrix(&R_B_W);
        let se3_data = DVector::from_vec(vec![
            t_B_W.x, t_B_W.y, t_B_W.z, q_B_W.w, q_B_W.i, q_B_W.j, q_B_W.k,
        ]);
        initial_values.insert(kf_var.clone(), (ManifoldType::SE3, se3_data.cast::<f64>()));

        // Add factors: for both the left and right cameras, each point that was already in the map is used to optimize the new frame
        // Fetch transforms between cameras and body
        let T_Cl_B = self.keyframes.front().unwrap().state.T_B_Cl.try_inverse().expect("T_B_Cl should be invertible");
        let T_Cr_B = self.keyframes.front().unwrap().state.T_B_Cr.try_inverse().expect("T_B_Cr should be invertible");
        let camera_features = [
            (&frame.left_features, T_Cl_B),
            (&frame.right_features, T_Cr_B),
        ];
        for (features, T_C_B) in camera_features.iter() {
            for feat in features.iter() {
                let feature_id = feat.feature_id;

                let point = self.map_points.get(&feature_id);
                match point {
                    Some(point) => {
                        
                        // Create PnP factor
                        let factor = PnPFactor::new(
                            na::Vector2::new(feat.undistorted_coord[0], feat.undistorted_coord[1]).cast::<f64>(),
                            *T_C_B,
                            na::Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64)
                        );                        
                        // Add residual block with Huber loss
                        let huber_loss = HuberLoss::new(2.0).unwrap();
                        problem.add_residual_block(&[&kf_var], Box::new(factor), Some(Box::new(huber_loss)));
                    }
                    None => {
                        // log::debug!("[SlidingWindow] Motion tracking: point {} is not in the map", feature_id);
                    }
                }
            
            }
        }

        // Initialize variables in the problem
        problem.initialize_variables(&initial_values);

        // Optimize
        let opt_result = match solver.optimize(&problem, &initial_values) {
            Ok(result) => result,
            Err(e) => {
                // Optimization error
                log::error!("[SlidingWindow] Motion tracking optimizer error: {:?}", e);
                return Ok(None);
            }
        };

        // Check if optimization was successful
        let is_successful = self.is_optimization_successful(&opt_result);
        
        if is_successful {
            // Extract optimized pose T_B_W and convert to T_W_B
            if let Some(value) = opt_result.parameters.get(&kf_var) {
                let T_B_W_opt = apex_solver::manifold::se3::SE3::from(value.to_vector()).matrix();
                let T_W_B_opt = T_B_W_opt.try_inverse()
                    .expect("Optimized T_B_W should be invertible");
                log::debug!(
                    "[SlidingWindow] Motion tracking successful. Initial cost: {:.3}, final cost: {:.3}",
                    opt_result.initial_cost,
                    opt_result.final_cost
                );
                Ok(Some(T_W_B_opt))
            } else {
                log::warn!("[SlidingWindow] Motion tracking: optimized pose not found in result");
                Ok(None)
            }
        } else {
            log::warn!("[SlidingWindow] Motion tracking failed (status: {:?})", opt_result.status);
            Ok(None)
        }
    }

    pub fn solve_gyroscope_bias(&mut self, imu_preintegrator: &mut ImuPipeline) {
        let mut a = na::Matrix3::zeros();
        let mut b = na::Vector3::zeros();

        for (id, (frame_i, frame_j)) in self.keyframes.iter().zip(self.keyframes.iter().skip(1)).enumerate() {
            let q_j = frame_j.state.T_W_B.fixed_view::<3, 3>(0, 0);
            let q_i = frame_i.state.T_W_B.fixed_view::<3, 3>(0, 0);

            let q_ij = na::UnitQuaternion::from_matrix(&(q_i.transpose() * q_j));

            let tmp_a = if frame_j.imu_preintegration.is_some() && frame_i.imu_preintegration.is_some() {
                frame_j.imu_preintegration.as_ref().unwrap().jacobian.fixed_view::<3, 3>(0, 12).into_owned()
            } else {
                log::error!("[SlidingWindow] Cannot solve gyroscope bias: missing IMU preintegration for frames {} and {}", frame_i.frame_id, frame_j.frame_id);
                // na::Matrix3::identity() // TODO handle this case properly, maybe return Result or Option
                continue;
            };
            let tmp_b = 2.0 * (frame_j.imu_preintegration.as_ref().unwrap().delta_R.quaternion().inverse() * q_ij).vector();

            a += tmp_a.transpose() * tmp_a;
            b += tmp_a.transpose() * tmp_b;
        }
        let delta_bg = match a.cholesky() {
            Some(chol) => {
                let solved = chol.solve(&b);
                if solved.iter().all(|x| x.is_finite()) {
                    solved
                } else {
                    log::warn!("[SlidingWindow] Gyro bias solve produced non-finite values; keeping previous gyro biases");
                    return;
                }
            }
            None => {
                log::warn!("[SlidingWindow] Gyro bias solve failed (A not SPD); keeping previous gyro biases");
                return;
            }
        };

        let mut bgs = Vec::new();
        let mut bga = Vec::new();
        for (id, frame) in self.keyframes.iter().enumerate() {
            let new_gyro_bias  = frame.state.gyro_bias + delta_bg;
            bgs.push(new_gyro_bias);

            bga.push(frame.state.accel_bias);
        }

        // Collect data before mutating self.keyframes
        for (id, frame) in self.keyframes.iter_mut().enumerate() {
            let preint = imu_preintegrator.repropagate(
                &bga[id], 
                &bgs[id]);
            if let Some(preint) = preint {
                frame.imu_preintegration = Some(preint);
            }
        }
    }

    pub fn solve_linear_alignment(&self,) -> Result<(na::Vector3<f64>, f64), std::io::Error> {
        let n_frames = self.keyframes.len();
        let n_states = n_frames * 3 + 3 + 1;
        let mut a = na::DMatrix::<f64>::zeros(n_states, n_states);
        let mut b = na::DVector::<f64>::zeros(n_states);

        for (i, (frame_i, frame_j)) in self.keyframes.iter().zip(self.keyframes.iter().skip(1)).enumerate() {
            let mut tmp_a = na::DMatrix::<f64>::zeros(6, 10);
            let mut tmp_b = na::DVector::<f64>::zeros(6);

            let dt = frame_j.imu_preintegration.as_ref().unwrap().dt;

            let dp = &frame_j.imu_preintegration.as_ref().unwrap().delta_p;
            let dv = &frame_j.imu_preintegration.as_ref().unwrap().delta_v;

            let I = na::Matrix3::<f64>::identity();
            let ti = frame_i.state.T_W_B.fixed_view::<3, 1>(0, 3);
            let tj = frame_j.state.T_W_B.fixed_view::<3, 1>(0, 3);
            let Ri = frame_i.state.T_W_B.fixed_view::<3, 3>(0, 0);
            let Rj = frame_j.state.T_W_B.fixed_view::<3, 3>(0, 0);
            let RiT = Ri.transpose();

            tmp_a.view_mut((0, 0), (3, 3)).copy_from(&(-dt * I));
            tmp_a.view_mut((0, 6), (3, 3)).copy_from(&(RiT * dt * dt / 2.0 * I));
            tmp_a.view_mut((0, 9), (3, 1)).copy_from(&(RiT * (tj - ti) / 100.0));
            
            // tmp_b.rows_mut(0, 3).copy_from(&(dp + RiT * Rj * t_B_W - t_B_W));
            tmp_b.rows_mut(0, 3).copy_from(&(dp)); // Assume data is in the body frame

            tmp_a.view_mut((3, 0), (3, 3)).copy_from(&(-I));
            tmp_a.view_mut((3, 3), (3, 3)).copy_from(&(RiT * Rj));
            tmp_a.view_mut((3, 6), (3, 3)).copy_from(&(RiT * dt * I));

            tmp_b.rows_mut(3, 3).copy_from(dv);

            let cov_inv = na::Matrix6::<f64>::identity();

            let r_a = tmp_a.transpose() * cov_inv * &tmp_a;
            let r_b = tmp_a.transpose() * cov_inv * &tmp_b;

            a.view_mut((i * 3, i * 3), (6, 6)).add_assign(&r_a.view((0, 0), (6, 6)));
            b.view_mut((i * 3, 0), (6, 1)).add_assign(&r_b.fixed_rows::<6>(0));

            a.view_mut((n_states - 4, n_states - 4), (4, 4)).add_assign(&r_a.view((r_a.nrows() - 4, r_a.ncols() - 4), (4, 4)));
            b.rows_mut(n_states - 4, 4).add_assign(&r_b.fixed_rows::<4>(r_b.len() - 4));

            a.view_mut((i * 3, n_states - 4), (6, 4)).add_assign(&r_a.view((0, r_a.ncols() - 4), (6, 4)));
            a.view_mut((n_states - 4, i * 3), (4, 6)).add_assign(&r_a.view((r_a.nrows() - 4, 0), (4, 6)));
        }

        a *= 1000.0;
        b *= 1000.0;
        let mut solution = match a.cholesky() {
            Some(chol) => {
                let solved = chol.solve(&b);
                if solved.iter().all(|x| x.is_finite()) {
                    solved
                } else {
                    log::warn!("[Linear Alignment] Solve produced non-finite values");
                    return Err(std::io::Error::new(std::io::ErrorKind::Other, "Linear alignment solve produced non-finite values"));
                }
            }
            None => {
                log::warn!("[Linear Alignment] Solve failed (A not SPD)");
                return Err(std::io::Error::new(std::io::ErrorKind::Other, "Linear alignment solve failed (A not SPD)"));
            }
        };
        let s = solution[n_states - 1] / 100.0;
        log::info!("[Linear Alignment] Linear alignment scale factor: {:.6}", s);

        let g = solution.fixed_rows::<3>(n_states - 4).into_owned();
        log::debug!("[Linear Alignment] Result g: {:?}, {:?}", g.norm(), g.transpose());
        let g_prior = na::Vector3::<f64>::new(0.0, 0.0, 9.81007);
        if (g.norm() - g_prior.norm()).abs() > 0.5 || s < 0.7 || s > 1.3 {
            log::warn!("[Linear Alignment] Linear alignment result is invalid. g.norm(): {:.6}, g_prior.norm(): {:.6}, scale factor: {:.6}", g.norm(), g_prior.norm(), s);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Linear alignment result is invalid. Need more motion"));
        }

        let g_refined = self.refine_gravity(&g, g_prior);

        let s = solution[n_states - 1] / 100.0;
        let last_idx = solution.len() - 1;
        solution[last_idx] = s;
        log::info!("[Linear Alignment] Result g: {:?}, {:?}", g_refined.norm(), g_refined.transpose());

        if s < 0.0 {
            log::warn!("[Linear Alignment] Negative scale factor from linear alignment, setting to 1.0");
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Negative scale factor from linear alignment"));
        } else {
            Ok((g_refined, s))
        }
    }

    fn refine_gravity(&self, initial_g: &na::Vector3<f64>, g_prior: na::Vector3<f64>) -> na::Vector3<f64> {
        let n_frames = self.keyframes.len();
        let n_states = n_frames * 3 + 2 + 1;
        let mut a = na::DMatrix::<f64>::zeros(n_states, n_states);
        let mut b = na::DVector::<f64>::zeros(n_states);

        let mut g = initial_g.clone();

        for j in 0..4 {
            let lxly = self.tangent_basis(&g);

            for (i, (frame_i, frame_j)) in self.keyframes.iter().zip(self.keyframes.iter().skip(1)).enumerate() {
                let mut tmp_a = na::DMatrix::<f64>::zeros(6, 9);
                let mut tmp_b = na::DVector::<f64>::zeros(6);

                let dt = frame_j.imu_preintegration.as_ref().unwrap().dt;
                let dp = &frame_j.imu_preintegration.as_ref().unwrap().delta_p;
                let dv = &frame_j.imu_preintegration.as_ref().unwrap().delta_v;

                let I = na::Matrix3::<f64>::identity();
                let ti = frame_i.state.T_W_B.fixed_view::<3, 1>(0, 3);
                let tj = frame_j.state.T_W_B.fixed_view::<3, 1>(0, 3);
                let Ri = frame_i.state.T_W_B.fixed_view::<3, 3>(0, 0);
                let Rj = frame_j.state.T_W_B.fixed_view::<3, 3>(0, 0);
                let RiT = Ri.transpose();

                tmp_a.view_mut((0, 0), (3, 3)).copy_from(&(-dt * I));
                tmp_a.view_mut((0, 6), (3, 2)).copy_from(&(RiT * dt * dt / 2.0 * I * lxly));
                tmp_a.view_mut((0, 8), (3, 1)).copy_from(&(RiT * (tj - ti) / 100.0));
                
                tmp_b.rows_mut(0, 3).copy_from(&(dp - RiT * dt * dt / 2.0 * g));

                tmp_a.view_mut((3, 0), (3, 3)).copy_from(&(-I));
                tmp_a.view_mut((3, 3), (3, 3)).copy_from(&(RiT * Rj));
                tmp_a.view_mut((3, 6), (3, 2)).copy_from(&(RiT * dt * I * lxly));

                tmp_b.rows_mut(3, 3).copy_from(&(dv - RiT * dt * I * g));

                let cov_inv = na::Matrix6::<f64>::identity();

                let r_a = tmp_a.transpose() * cov_inv * &tmp_a;
                let r_b = tmp_a.transpose() * cov_inv * &tmp_b;

                a.view_mut((i * 3, i * 3), (6, 6)).add_assign(&r_a.view((0, 0), (6, 6)));
                b.view_mut((i * 3, 0), (6, 1)).add_assign(&r_b.fixed_rows::<6>(0));

                a.view_mut((n_states - 3, n_states - 3), (3, 3)).add_assign(&r_a.view((r_a.nrows() - 3, r_a.ncols() - 3), (3, 3)));
                b.rows_mut(n_states - 3, 3).axpy(1.0, &r_b.rows(r_b.len() - 3, 3), 1.0);

                a.view_mut((i * 3, n_states - 3), (6, 3)).add_assign(&r_a.view((0, r_a.ncols() - 3), (6, 3)));
                a.view_mut((n_states - 3, i * 3), (3, 6)).add_assign(&r_a.view((r_a.nrows() - 3, 0), (3, 6)));

            };
            a *= 1000.0;
            b *= 1000.0;
            let solution = match a.clone().cholesky() {
                Some(chol) => {
                    let solved = chol.solve(&b);
                    if solved.iter().all(|x| x.is_finite()) {
                        solved
                    } else {
                        log::warn!("[Gravity Refinement] Solve produced non-finite values at iteration {}; keeping previous gravity", j);
                        break;
                    }
                }
                None => {
                    log::warn!("[Gravity Refinement] Solve failed (A not SPD) at iteration {}; keeping previous gravity", j);
                    break;
                }
            };
            let dg = solution.fixed_rows::<2>(n_states - 3);
            g = (g + lxly * dg).normalize() * g_prior.norm();
        };

        g
    }

    fn tangent_basis(&self, g0: &na::Vector3<f64>) -> na::Matrix3x2<f64> {
        let a = na::Unit::new_normalize(*g0);  // Equivalent to g0.normalized()
        let mut tmp = na::Vector3::new(0.0, 0.0, 1.0);
        if a.as_ref() == &tmp {
            tmp = na::Vector3::new(1.0, 0.0, 0.0);
        }
        let b = na::Unit::new_normalize(tmp - a.as_ref() * (a.as_ref().dot(&tmp)));
        let c = a.cross(&b);
        let mut bc = na::Matrix3x2::<f64>::zeros();
        bc.column_mut(0).copy_from(b.as_ref());
        bc.column_mut(1).copy_from(&c);
        bc
    }

    fn triangulate_stereo(&self, left_feat: &Feature, T_Cl_B: &na::Matrix4<f64>, right_feat: &Feature, T_Cr_B: &na::Matrix4<f64>) -> Option<na::Vector3<f64>> {
        // Build DLT system A * X = 0 (4x4), solve via SVD
        // Each observation contributes 2 rows:  x*(P[2]) - P[0],  y*(P[2]) - P[1]
        let p_l = T_Cl_B.fixed_view::<3, 4>(0, 0);
        let p_r = T_Cr_B.fixed_view::<3, 4>(0, 0);

        let mut a = na::Matrix4::<f64>::zeros();
        a.row_mut(0).copy_from(&((left_feat.undistorted_coord[0] as f64) * p_l.row(2) - p_l.row(0)));
        a.row_mut(1).copy_from(&((left_feat.undistorted_coord[1] as f64) * p_l.row(2) - p_l.row(1)));
        a.row_mut(2).copy_from(&((right_feat.undistorted_coord[0] as f64) * p_r.row(2) - p_r.row(0)));
        a.row_mut(3).copy_from(&((right_feat.undistorted_coord[1] as f64) * p_r.row(2) - p_r.row(1)));

        let svd = na::SVD::new(a, false, true);
        let v = svd.v_t?.transpose();
        let x: na::Vector4<f64> = v.column(3).into();
        
        if x.w.abs() < 1e-10 { return None; }

        let sign = x.w.signum(); // ensure consistent sign
        let depth_l = sign * p_l.row(2).transpose().dot(&x);
        let depth_r = sign * p_r.row(2).transpose().dot(&x);

        if depth_l <= 0.0 || depth_r <= 0.0 { return None; }

        Some(na::Point3::from(x.xyz() / x.w).coords)
    }
}

