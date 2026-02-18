use crate::datasets::ImuData;
use crate::datasets::config::{Config, ImuConfig};
use crate::estimator::Frame;
use crate::feature_tracker::StereoPatchTracker;
use crate::imu;
use crate::imu::piecewise_integration::{self, ImuPiecewiseIntegration, PreInt};
use crate::types::{Matrix4x4, UnitQuaternion, Vector3};
use crate::viewers::Viewer;
use camera_intrinsic_model::GenericModel;
use image::{DynamicImage, GrayImage};
use anyhow::Result;
use nalgebra::{self as na, Vector6};
use std::time::Instant;
use crate::estimator::sliding_window::SlidingWindow;
use crate::datasets::CameraModelType;


/// Placeholder estimator implementation.
/// Currently mimics the control flow and logging structure of the C++ Estimator::process_frame,
/// but uses dummy values for tracking, optimization, and mapping.
pub struct Estimator<'a> {
    frame_id_counter: u64,
    frames_since_last_keyframe: u64,
    /// When true, emit detailed per-frame logs (equivalent to Config::m_enable_debug_output).
    enable_debug_output: bool,
    /// Full configuration loaded from YAML (used to derive intrinsics, etc.).
    pub(crate) config: Config,
    /// Patch-based stereo tracker reused across all frames.
    stereo_patch_tracker: StereoPatchTracker<6>,
    /// Sliding window of keyframes for bundle adjustment optimization.
    sliding_window: SlidingWindow,
    /// Optional viewer used for visualization; outlives the estimator.
    viewer: Option<&'a mut dyn Viewer>,
    /// Left camera model with intrinsics and distortion.
    left_cam: CameraModelType,
    /// Right camera model with intrinsics and distortion.
    right_cam: CameraModelType,
    // Transformation from body to left camera
    T_B_Cl: Matrix4x4,
    // Transformation from body to right camera
    T_B_Cr: Matrix4x4,
    // Full trajectory of keyframes
    trajectory: Vec<Matrix4x4>,
    // IMU piecewise integration module
    piecewise_integration: ImuPiecewiseIntegration,
    // IMU biases
    velocities: Vec<Vector3>,
    // IMU biases
    gyro_biases: Vec<Vector3>,
    accel_biases: Vec<Vector3>,
}

impl<'a> Estimator<'a> {
    #![allow(non_snake_case)]

    /// Create a new estimator configured with camera intrinsics and distortion
    /// loaded from the YAML configuration.
    ///
    /// The `viewer` reference must outlive the estimator.
    pub fn new(config: Config, viewer: Option<&'a mut dyn Viewer>) -> Self {
        Self::new_with_cameras(config, viewer, None, None, None)
    }

    /// Create a new estimator with optional camera models.
    /// If camera models are provided, they will be used; otherwise, they will be created from config.
    ///
    /// The `viewer` reference must outlive the estimator.
    pub fn new_with_cameras(
        config: Config,
        viewer: Option<&'a mut dyn Viewer>,
        left_cam: Option<CameraModelType>,
        right_cam: Option<CameraModelType>,
        imu_config: Option<ImuConfig>,
    ) -> Self {
        // let imu_config = imu_config.unwrap();
        let piecewise_integration = if imu_config.is_some() {
            ImuPiecewiseIntegration::from_config(imu_config.unwrap())
        } else {
            ImuPiecewiseIntegration::new()
        };
        // Use provided cameras or create from config
        let (left_cam, right_cam) = match (left_cam, right_cam) {
            (Some(l), Some(r)) => (l, r),
            _ => crate::datasets::create_camera_models_from_config(&config),
        };

        // Compute the transformation from left to right (T_C1_C0) as in compute_stereo.
        let T_B_Cl = na::Matrix4::from_row_slice(&config.camera.T_B_Cl);
        let T_B_Cr = na::Matrix4::from_row_slice(&config.camera.T_B_Cr);
        let keyframe_window_size = config.keyframe_management.keyframe_window_size as usize;
        let grid_size = config.feature_detection.grid_cols;
        let optical_flow_max_iterations = config.feature_detection.optical_flow_max_iterations;
        let optical_flow_convergence_threshold = config.feature_detection.optical_flow_convergence_threshold;
        
        Estimator {
            frame_id_counter: 0,
            frames_since_last_keyframe: 0,
            enable_debug_output: true,
            config,
            stereo_patch_tracker: StereoPatchTracker::<6>::new(
                grid_size,
                optical_flow_max_iterations,
                optical_flow_convergence_threshold,
            ),
            sliding_window: SlidingWindow::new(keyframe_window_size),
            viewer,
            left_cam,
            right_cam,
            T_B_Cl: T_B_Cl,
            T_B_Cr: T_B_Cr,
            trajectory: Vec::new(),
            piecewise_integration: piecewise_integration,
            velocities: Vec::new(),
            gyro_biases: Vec::new(),
            accel_biases: Vec::new(),
        }
    }

    /// Process a single stereo frame.
    pub fn process_frame(
        &mut self,
        left_image: &[u8],
        right_image: &[u8],
        timestamp_ns: i64,
        imu_data: Option<&[ImuData]>,
    ) -> Result<()> {
        let total_start_time = Instant::now();

        // New frame: update counters
        self.frame_id_counter += 1;
        self.frames_since_last_keyframe += 1;

        if self.enable_debug_output {
            log::debug!("============================== Frame {} ==============================", self.frame_id_counter);
        }

        // Timing placeholders
        let mut frame_creation_time_ms = 0.0f64;
        let mut patch_tracking_time_ms = 0.0f64;
        let mut motion_tracking_time_ms = 0.0f64;
        let mut optimization_time_ms = 0.0f64;

        // Frame creation
        let frame_creation_start = Instant::now();
        
        // Create GrayImage objects directly from input slices (no clone needed for tracking)
        let img_w = self.config.camera.image_width;
        let img_h = self.config.camera.image_height;
        
        let left_img = match GrayImage::from_raw(img_w, img_h, left_image.to_vec()) {
            Some(img) => img,
            None => {
                log::error!(
                    "[Estimator] Failed to construct GrayImage for left camera ({}x{}, len={})",
                    img_w,
                    img_h,
                    left_image.len()
                );
                return Ok(());
            }
        };
        let right_img = match GrayImage::from_raw(img_w, img_h, right_image.to_vec()) {
            Some(img) => img,
            None => {
                log::error!(
                    "[Estimator] Failed to construct GrayImage for right camera ({}x{}, len={})",
                    img_w,
                    img_h,
                    right_image.len()
                );
                return Ok(());
            }
        };

        /*
        // For debugging with TUM-VI: undistort the images with EUCM and save the result
        let eucm = match &self.left_cam {
            crate::datasets::CameraModelType::EUCM(cam) => cam,
            _ => panic!("left_cam is not an EUCM instance"),
        };
        let model1 = GenericModel::EUCM(*eucm);
        let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((1024, 1024)));
        let (xmap, ymap) = model1.init_undistort_map(&p, (1024, 1024), None);
        let img_l8 = DynamicImage::ImageLuma8(right_img.clone());
        let remaped = camera_intrinsic_model::remap(&img_l8, &xmap, &ymap);
        remaped.save("remaped0.png").unwrap();
        */
        log::info!(
            "[Estimator] Last velocity: {:?}, Last accel bias: {:?}, Last gyro bias: {:?}",
            self.velocities.last(),
            self.accel_biases.last(),
            self.gyro_biases.last()
        );

        // Create frame (images are not stored, only features will be added)
        let mut current_frame = Frame::from_stereo_images(
            timestamp_ns,
            self.frame_id_counter as i32,
            self.left_cam.clone(),
            self.right_cam.clone(),
            self.T_B_Cl,
            self.T_B_Cr,
            self.velocities.last().copied(),
            self.accel_biases.last().copied(),
            self.gyro_biases.last().copied(),
        );

        // Attach IMU measurements if available.
        if let Some(imu) = imu_data {
            current_frame.imu_from_last_frame = imu.to_vec();
        }

        frame_creation_time_ms = frame_creation_start.elapsed().as_secs_f64() * 1000.0;

        // Patch tracking
        let tracking_start = Instant::now();
        self.stereo_patch_tracker.process_frame(&left_img, &right_img, &mut current_frame);
        patch_tracking_time_ms = tracking_start.elapsed().as_secs_f64() * 1000.0;
        self.view_patch_tracking_results(&current_frame, &left_img, &right_img, img_w, img_h);

        // Motion tracking - only if the sliding window is full (has initialized keyframes)
        if self.sliding_window.is_full() { // DEBUG
            let motion_tracking_start = Instant::now();
            let motion_tracking_result = self.sliding_window.track_motion(&current_frame);
            match motion_tracking_result {
                Ok(Some(T_W_B)) => {
                    // Apply the optimized pose to the current frame
                    current_frame.state.T_W_B = T_W_B;
                    
                    // Check if translation and rotation since last keyframe is large enough to trigger a keyframe
                    let T_W_B_last_kf = self.sliding_window.get_keyframe_poses().last().unwrap().clone();
                    let T_rel = T_W_B * T_W_B_last_kf.try_inverse().unwrap();
                    let t_rel = T_rel.fixed_view::<3, 1>(0, 3).into_owned();
                    let R_rel = T_rel.fixed_view::<3, 3>(0, 0).into_owned();
                    let e_rel = Vector3::from([
                        UnitQuaternion::from_matrix(&R_rel).euler_angles().0,
                        UnitQuaternion::from_matrix(&R_rel).euler_angles().1,
                        UnitQuaternion::from_matrix(&R_rel).euler_angles().2]
                    );
                    log::debug!("[Estimator] Translation since last keyframe: {:.2?}, Euler angles since last keyframe: {:.2?}", t_rel, e_rel);

                    // Check if translation and rotation since last keyframe is large enough to trigger a keyframe
                    let translation_threshold = self.config.keyframe_management.translation_threshold;
                    let rotation_threshold = self.config.keyframe_management.rotation_threshold;

                    if t_rel.norm() > translation_threshold || e_rel.norm() > rotation_threshold {
                        log::debug!("[Estimator] Translation and rotation since last keyframe are large enough to trigger a keyframe");
                        current_frame.is_keyframe = true;
                    }
                    else {
                        current_frame.is_keyframe = false;
                    }
                    self.view_motion_tracking_results(&T_W_B);
                }
                Ok(None) => {
                    log::warn!("[Estimator] Motion tracking failed (optimization did not converge)");
                }
                Err(e) => {
                    log::error!("[Estimator] Motion tracking error: {:?}", e);
                }
            }
            motion_tracking_time_ms = motion_tracking_start.elapsed().as_secs_f64() * 1000.0;
        } else {
            log::debug!("[Estimator] Sliding window is not full, skipping motion tracking");
        }

        let mut imu_time_ms = 0.0f64;
        // View map points and keyframe poses
        // Bundle adjustment
        if current_frame.is_keyframe {
            let imu_start = Instant::now();
            let imu_preint = self.piecewise_integration.propagate(
                imu_data.unwrap_or(&[]),
                &self.accel_biases.last().unwrap_or(&Vector3::zeros()), // TODO: safeguard against empty history
                &self.gyro_biases.last().unwrap_or(&Vector3::zeros()),
            );
            
            current_frame.add_imu_preint(imu_preint);
            imu_time_ms = imu_start.elapsed().as_secs_f64() * 1000.0;

            let optimization_start = Instant::now();
            self.sliding_window.add_frame(current_frame);
            self.sliding_window.optimize(); // TODO handle error
            optimization_time_ms = optimization_start.elapsed().as_secs_f64() * 1000.0;
            self.view_optimization_results();
        }
        
        // Final timing summary
        let total_duration_ms = total_start_time.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "[Timing] frame_creation={:.3} ms, patch_tracking={:.3} ms, motion_tracking={:.3} ms, optimization={:.3} ms, imu={:.3} ms, total={:.3} ms",
            frame_creation_time_ms,
            patch_tracking_time_ms,
            motion_tracking_time_ms,
            optimization_time_ms,
            imu_time_ms,
            total_duration_ms
        );
    
        Ok(())
    }

    /// Helper: set the current frame index on the attached viewer, if any.
    pub fn set_viewer_frame(&mut self, frame_id: i64) {
        if let Some(v) = &mut self.viewer {
            v.set_frame(frame_id);
        }
    }

    /// Visualize tracking results: stereo images with tracked features.
    fn view_patch_tracking_results(&mut self, frame: &Frame, left_img: &GrayImage, right_img: &GrayImage, width: u32, height: u32) {
        if let Some(v) = &mut self.viewer {
            // Collect pixel coordinates into temporary vectors so we can pass slices.
            let left_points: Vec<(usize, [f32; 2])> =
                frame.left_features().iter().map(|f| (f.feature_id, f.pixel_coord)).collect();
            let right_points: Vec<(usize, [f32; 2])> =
                frame.right_features().iter().map(|f| (f.feature_id, f.pixel_coord)).collect();

            v.log_image_with_features_colored(
                left_img.as_raw(),
                width,
                height,
                &left_points,
                "stereo/left",
            );
            v.log_image_with_features_colored(
                right_img.as_raw(),
                width,
                height,
                &right_points,
                "stereo/right",
            );
        }
    }

    fn view_motion_tracking_results(&mut self, T_W_B: &Matrix4x4) {
        if let Some(v) = &mut self.viewer {
            let pose_path = format!("pose_current");
            v.log_pose(*T_W_B, pose_path.as_str());

            let width = self.config.camera.image_width;
            let height = self.config.camera.image_height;

            // Log left camera frustum at the pose location (left camera is at the pose)
            let left_focal_length = self.config.camera.left_intrinsics[0] as f32;
            let left_cam_path = format!("{}_left", pose_path);
            let T_W_Cl = T_W_B * self.T_B_Cl;
            v.log_pose(T_W_Cl, left_cam_path.as_str());
            v.log_camera_frustum(left_focal_length, width, height, left_cam_path.as_str(),0.4);
        }
    }

    /// Visualize optimization results: map points, keyframe poses, and camera frustums.
    fn view_optimization_results(&mut self) {
        if let Some(v) = &mut self.viewer {
            // Map points
            let colored_points: Vec<(usize, [f32; 3])> = self.sliding_window.map_points
                .iter()
                .map(|(&feature_id, &point)| (feature_id, point))
                .collect();
            v.log_points_colored(&colored_points, "map/points");

            // Keyframe poses with left and right camera frustrums
            let system_poses = self.sliding_window.get_keyframe_poses();
            for (pose_id, T_W_B) in system_poses.iter().enumerate() {
                // pose is T_W_B
                let pose_path = format!("pose_{}", pose_id);
                v.log_pose(*T_W_B, pose_path.as_str());

                let width = self.config.camera.image_width;
                let height = self.config.camera.image_height;

                // Log left camera frustum at the pose location (left camera is at the pose)
                let left_focal_length = self.config.camera.left_intrinsics[0] as f32;
                let left_cam_path = format!("{}_left", pose_path);
                let T_W_Cl = T_W_B * self.T_B_Cl;
                v.log_pose(T_W_Cl, left_cam_path.as_str());
                let size = 0.2; // if pose_id == system_poses.len() - 1 { 0.5 } else { 0.2 };
                v.log_camera_frustum(left_focal_length, width, height, left_cam_path.as_str(), size);
            
                
                // Log right camera pose and frustum for last pose
                // Removed for now as it made too much clutter
                /*
                if pose_id == system_poses.len() - 1 {
                    let right_cam_path = format!("{}_right", pose_path);
                    let right_focal_length = self.config.camera.right_intrinsics[0] as f32;
                    let T_W_Cr = T_W_B * self.T_B_Cr;
                    v.log_pose(T_W_Cr, right_cam_path.as_str());
                    v.log_camera_frustum(right_focal_length, width, height, right_cam_path.as_str(), size);
                }
                */
            }

            // History of keyframe poses
            let mat =  self.sliding_window.get_keyframe_poses().first().unwrap().clone();
            self.trajectory.push(mat);

            let vel = self.sliding_window.get_keyframe_velocities().first().unwrap().clone();
            self.velocities.push(vel);

            let ba = self.sliding_window.get_keyframe_accel_bias().first().unwrap().clone();
            self.accel_biases.push(ba);

            let bg = self.sliding_window.get_keyframe_gyro_bias().first().unwrap().clone();
            self.gyro_biases.push(bg);
            
            // Display trajectory as a continuous 3D path
            v.log_trajectory(&self.trajectory, "trajectory/path");
            // log::info!("[Estimator] System position: {:?}, {:?}, {:?}", mat[0][3], mat[1][3], mat[2][3]);
        }
    }
}
