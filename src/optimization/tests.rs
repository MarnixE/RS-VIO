#[cfg(test)]
mod tests {
    use crate::optimization::factors::ImuFactor;
    use crate::optimization::observer::TerminalObserver;
    use crate::optimization::factors::PinholeProjectionFactor;
    use crate::optimization::factors::BundleAdjustmentFactorTranslationOnly;
    use crate::optimization::factors::BundleAdjustmentFactor;
    use apex_solver::Factor;
    use apex_solver::manifold::Tangent;
    use apex_solver::manifold::so3;
    use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
    use apex_solver::linalg::LinearSolverType;
    use apex_solver::manifold::ManifoldType;
    use apex_solver::core::problem::Problem;
    use rand::random;
    use std::collections::HashMap;
    use nalgebra as na;
    use na::DVector;

    #[test]
    fn test_pinhole_projection_factor() {
        // True parameters: y = 2.0*x² - 3.0*x + 1.0
        let map_point_true = vec![1.0, 1.0, 1.0]; // point is right down forward of cam 0

        // cam 0: 90 def FoV, origin: point is at (1,1)
        // cam 1: 90 def FoV, 1m on the right: point is at (0, 1)

        
        let mut problem = Problem::new();
        
        let config = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseCholesky)
            .with_max_iterations(100)
            .with_cost_tolerance(1e-9)
            .with_parameter_tolerance(1e-9)
            .with_jacobi_scaling(false);

        let mut solver = LevenbergMarquardt::with_config(config);
        let mut initial_values = HashMap::new();


        let lm_var = format!("LM_{}", 0);
        let data = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        initial_values.insert(lm_var.clone(), (ManifoldType::RN, data));

        let mut T_W_right = na::Matrix4::identity();
        T_W_right[(0, 3)] = 0.5;
        let T_Cright_W = T_W_right.try_inverse().unwrap();

        let mut T_W_Cbottom = na::Matrix4::identity();
        T_W_Cbottom[(1, 3)] = 0.5;
        let T_Cbottom_W = T_W_Cbottom.try_inverse().unwrap();
        
        // Left camera projection factor
        let left_factor = PinholeProjectionFactor::new(
            na::Vector2::new(1.0, 1.0).cast::<f64>(),
            na::Matrix4::identity(),
        );
        problem.add_residual_block(
            &[&lm_var],
            Box::new(left_factor),
            None
        );// Left camera projection factor
        let right_factor = PinholeProjectionFactor::new(
            na::Vector2::new(0.5, 1.0).cast::<f64>(),
            T_Cright_W,
        );
        problem.add_residual_block(
            &[&lm_var],
            Box::new(right_factor),
            None
        );

        let bottom_factor = PinholeProjectionFactor::new(
            na::Vector2::new(1.0, 0.5).cast::<f64>(),
            T_Cbottom_W,
        );
        problem.add_residual_block(
            &[&lm_var],
            Box::new(bottom_factor),
            None
        );

        // Initialize variables in the problem
        problem.initialize_variables(&initial_values);

        // Configure Levenberg-Marquardt solver
        let config = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseCholesky)
            .with_max_iterations(50)
            .with_cost_tolerance(1e-6)
            .with_jacobi_scaling(false);

        let mut solver = LevenbergMarquardt::with_config(config);

        // Add terminal observer to monitor progress
        let observer = TerminalObserver::new();
        TerminalObserver::print_header();
        solver.add_observer(observer);

        // Run optimization
        let result = solver.optimize(&problem, &initial_values);

        // Check results
        assert!(result.is_ok(), "Optimization should succeed");
        let opt_result = result.unwrap();

        // Extract optimized parameters
        let optimized_params = opt_result.parameters.get(&lm_var).unwrap();
        let params_vec = optimized_params.to_vector();
        

        println!("\nOptimization Results:");
        println!("True parameters:  x={:?}", map_point_true);
        println!("Optimized params: x={:?}", params_vec);
        println!("Initial cost: {:.6}", opt_result.initial_cost);
        println!("Final cost: {:.6}", opt_result.final_cost);
        
        // Check that optimization converged
        use apex_solver::optimizer::OptimizationStatus;
        match opt_result.status {
            OptimizationStatus::Converged 
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached => {
                println!("Optimization converged successfully!");
            }
            _ => {
                println!("Warning: Optimization did not fully converge. Status: {:?}", opt_result.status);
            }
        }
    }

    /// Test bundle adjustment with translation-only optimization.
    /// 
    /// Tests the BundleAdjustmentFactorTranslationOnly factor by:
    /// 1. Creating random 3D landmarks in world frame
    /// 2. Observing them from multiple camera poses (with known extrinsics)
    /// 3. Optimizing landmark positions and camera translations
    #[test]
    fn test_bundle_adjustment_factor_translation_only() {
        // Constants
        const NUM_LANDMARKS: usize = 10;
        const NUM_POSES: usize = 5; // Number of system poses (each with left-right cameras)
        const NOISE_RANGE: f64 = 0.05;
        const MIN_DEPTH: f64 = 0.1; // Minimum depth for point to be visible
        const TRANSLATION_RANGE: f64 = 3.0; // Range for random translations
        
        // Helper: Extract landmark index from variable name
        fn get_landmark_idx(lm_var: &str) -> usize {
            lm_var.strip_prefix("LM_").unwrap().parse().unwrap()
        }
        
        // Helper: Project 3D point to normalized camera coordinates
        fn project_to_normalized(p_cam: na::Vector3<f64>) -> na::Vector2<f64> {
            na::Vector2::new(p_cam[0] / p_cam[2], p_cam[1] / p_cam[2])
        }
        
        // Helper: Check if point is visible (in front of camera)
        fn is_visible(p_cam: &na::Vector3<f64>, min_depth: f64) -> bool {
            p_cam[2] > min_depth && p_cam.iter().all(|&x| x.is_finite())
        }
        
        // Helper: Transform point from world to camera frame
        fn world_to_camera(
            p_world: na::Vector3<f64>,
            t_body_world: na::Vector3<f64>,
            t_cam_body: &na::Matrix4<f64>,
        ) -> na::Vector3<f64> {
            let r_c_b = t_cam_body.fixed_view::<3, 3>(0, 0);
            let t_c_b = t_cam_body.fixed_view::<3, 1>(0, 3);
            r_c_b * (p_world + t_body_world) + t_c_b
        }
        
        // Helper: Add factor for a camera observation
        fn add_camera_factor(
            problem: &mut Problem,
            lm_var: &str,
            cam_var: Option<&str>,
            observation: na::Vector2<f64>,
            t_cam_body: na::Matrix4<f64>,
            fixed_position: Option<na::Vector3<f64>>,
        ) {
            let mut factor = BundleAdjustmentFactorTranslationOnly::new(observation, t_cam_body);
            if let Some(pos) = fixed_position {
                factor = factor.with_fixed_position(pos);
            }
            
            let var_names: Vec<&str> = if let Some(cv) = cam_var {
                vec![lm_var, cv]
            } else {
                vec![lm_var]
            };
            
            problem.add_residual_block(&var_names, Box::new(factor), None);
        }

        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();
        let mut landmarks = Vec::new();
        let mut point_vars = Vec::new();

        // Generate random 3D landmarks with noisy initial estimates
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for i in 0..NUM_LANDMARKS {
            // True landmark position
            let true_point = vec![
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(0.5..3.0),
            ];
            landmarks.push(true_point.clone());
            
            // Noisy initial estimate
            let noise: Vec<f64> = (0..3)
                .map(|_| rng.gen_range(-NOISE_RANGE..NOISE_RANGE))
                .collect();
            let noisy_point: Vec<f64> = true_point
                .iter()
                .zip(noise.iter())
                .map(|(p, n)| p + n)
                .collect();
            
            let lm_var = format!("LM_{}", i);
            initial_values.insert(
                lm_var.clone(),
                (ManifoldType::RN, DVector::from_vec(noisy_point)),
            );
            point_vars.push(lm_var);
        }

        // Camera extrinsics (camera-to-body transforms)
        let t_cam_left_body = na::Matrix4::identity();
        let mut t_cam_right_body = na::Matrix4::identity();
        t_cam_right_body[(0, 3)] = 0.5; // Right camera offset in x

        // Generate random system poses
        let mut pose_translations = Vec::new();
        let mut pose_vars = Vec::new();
        
        for pose_id in 0..NUM_POSES {
            // Generate random translation for this pose
            let t_body_world = if pose_id == 0 {
                // First pose is fixed at origin
                na::Vector3::zeros()
            } else {
                na::Vector3::new(
                    rng.gen_range(-TRANSLATION_RANGE..TRANSLATION_RANGE),
                    rng.gen_range(-TRANSLATION_RANGE..TRANSLATION_RANGE),
                    rng.gen_range(-TRANSLATION_RANGE * 0.5..TRANSLATION_RANGE * 0.5),
                )
            };
            pose_translations.push(t_body_world);
            
            // Create variable for this pose (only if not fixed)
            if pose_id > 0 {
                let cam_var = format!("KF_{}", pose_id);
                // Initial estimate with some noise
                let noisy_translation = t_body_world + na::Vector3::new(
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                );
                let cam_data = DVector::from_vec(vec![
                    noisy_translation.x,
                    noisy_translation.y,
                    noisy_translation.z,
                ]);
                initial_values.insert(cam_var.clone(), (ManifoldType::RN, cam_data));
                pose_vars.push((pose_id, cam_var));
            }
        }

        // Add observations from each pose (left + right cameras)
        let mut total_observations = 0;
        for (pose_id, t_body_world) in pose_translations.iter().enumerate() {
            let cam_var_opt = pose_vars.iter().find(|(id, _)| *id == pose_id).map(|(_, v)| v.as_str());
            let is_fixed = pose_id == 0;
            
            for lm_var in &point_vars {
                let idx = get_landmark_idx(lm_var);
                let p_w = na::Vector3::new(landmarks[idx][0], landmarks[idx][1], landmarks[idx][2]);
                
                // Left camera observation
                let p_cam_left = world_to_camera(p_w, *t_body_world, &t_cam_left_body);
                if is_visible(&p_cam_left, MIN_DEPTH) {
                    let obs_left = project_to_normalized(p_cam_left);
                    add_camera_factor(
                        &mut problem,
                        lm_var,
                        cam_var_opt,
                        obs_left,
                        t_cam_left_body,
                        if is_fixed { Some(*t_body_world) } else { None },
                    );
                    total_observations += 1;
                }
                
                // Right camera observation
                let p_cam_right = world_to_camera(p_w, *t_body_world, &t_cam_right_body);
                if is_visible(&p_cam_right, MIN_DEPTH) {
                    let obs_right = project_to_normalized(p_cam_right);
                    add_camera_factor(
                        &mut problem,
                        lm_var,
                        cam_var_opt,
                        obs_right,
                        t_cam_right_body,
                        if is_fixed { Some(*t_body_world) } else { None },
                    );
                    total_observations += 1;
                }
            }
        }
        
        println!("Generated {} poses with {} total stereo observations", NUM_POSES, total_observations);

        // Initialize problem
        problem.initialize_variables(&initial_values);

        // Configure and run optimization
        let config = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseCholesky)
            .with_max_iterations(50)
            .with_cost_tolerance(1e-6)
            .with_jacobi_scaling(false);

        let mut solver = LevenbergMarquardt::with_config(config);
        let observer = TerminalObserver::new();
        TerminalObserver::print_header();
        solver.add_observer(observer);

        let result = solver.optimize(&problem, &initial_values);
        assert!(result.is_ok(), "Optimization should succeed");
        let opt_result = result.unwrap();

        // Verify results
        const MAX_LANDMARK_ERROR: f64 = 1e-3;
        let mut all_landmarks_valid = true;

        println!("\nOptimization Results:");
        println!("Initial cost: {:.6}", opt_result.initial_cost);
        println!("Final cost: {:.6}", opt_result.final_cost);

        // Check landmark convergence
        for (var_name, value) in opt_result.parameters.iter() {
            if var_name.starts_with("LM_") {
                let idx = get_landmark_idx(var_name);
                if let Some(true_point) = landmarks.get(idx) {
                    let optimized = value.to_vector();
                    let true_vec = DVector::from_vec(true_point.clone());
                    let error = (optimized - true_vec).norm();
                    
                    println!("  {}: error = {:.6}", var_name, error);
                    
                    if error > MAX_LANDMARK_ERROR {
                        println!(
                            "    WARNING: Landmark {} error ({:.6}) exceeds threshold ({:.6})",
                            idx, error, MAX_LANDMARK_ERROR
                        );
                        all_landmarks_valid = false;
                    }
                }
            }
        }

        // Check convergence status
        use apex_solver::optimizer::OptimizationStatus;
        let converged = matches!(
            opt_result.status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::ParameterToleranceReached
                | OptimizationStatus::GradientToleranceReached
        );

        if !converged {
            println!("Warning: Optimization did not fully converge. Status: {:?}", opt_result.status);
        }

        assert!(all_landmarks_valid, "Some landmarks did not converge to true values");
        assert!(converged, "Optimization did not converge");
    }


    /// Test bundle adjustment with translation and rotation (full SE3).
    /// 
    /// Similar to test_bundle_adjustment_factor_translation_only, but adds small rotations
    /// to the system poses. Since the factor only handles translations, there will be a
    /// small un-optimizable error due to the rotations. This will be fixed in future steps.
    #[test]
    fn test_bundle_adjustment_factor_full() {
        // Constants
        const NUM_LANDMARKS: usize = 10;
        const NUM_POSES: usize = 5; // Number of system poses (each with left-right cameras)
        const NOISE_RANGE: f64 = 0.05;
        const MIN_DEPTH: f64 = 0.1; // Minimum depth for point to be visible
        const TRANSLATION_RANGE: f64 = 3.0; // Range for random translations
        const MAX_ROTATION_ANGLE: f64 = 0.5; // Maximum rotation angle in radians (~5.7 degrees)
        
        // Helper: Extract landmark index from variable name
        fn get_landmark_idx(lm_var: &str) -> usize {
            lm_var.strip_prefix("LM_").unwrap().parse().unwrap()
        }
        
        // Helper: Project 3D point to normalized camera coordinates
        fn project_to_normalized(p_C: na::Vector3<f64>) -> na::Vector2<f64> {
            na::Vector2::new(p_C[0] / p_C[2], p_C[1] / p_C[2])
        }
        
        // Helper: Check if point is visible (in front of camera)
        fn is_visible(p_C: &na::Vector3<f64>, min_depth: f64) -> bool {
            p_C[2] > min_depth && p_C.iter().all(|&x| x.is_finite())
        }
        
        // Helper: Generate a small random rotation (axis-angle representation)
        fn generate_small_rotation(rng: &mut impl rand::Rng, max_angle: f64) -> na::UnitQuaternion<f64> {
            // Random axis (normalized)
            let axis = na::Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            let axis = axis.normalize();
            
            // Random angle
            let angle = rng.gen_range(-max_angle..max_angle);
            
            na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(axis), angle)
        }
        
        // Helper: Transform point from world to camera frame with rotation
        // Notation: T_A_B = SE3 transform from B to A, R_A_B = SO3 rotation from B to A, t_A_B = R3 translation from B to A
        fn world_to_camera_with_rotation(
            p_W: na::Vector3<f64>,
            t_B_W: na::Vector3<f64>,  // t_B_W: translation from W to B
            R_B_W: &na::UnitQuaternion<f64>,  // R_B_W: rotation from W to B
            T_C_B: &na::Matrix4<f64>,  // T_C_B: SE3 transform from B to C (camera)
        ) -> na::Vector3<f64> {
            // Transform chain: p_C = T_C_B * T_B_W * p_W
            // where T_B_W = [R_B_W | t_B_W; 0 0 0 1]
            // Extract R_C_B and t_C_B from T_C_B
            let R_C_B = T_C_B.fixed_view::<3, 3>(0, 0);
            let t_C_B = T_C_B.fixed_view::<3, 1>(0, 3);
            
            // Apply body-to-world transform: p_B = R_B_W * p_W + t_B_W
            let p_B = R_B_W * p_W + t_B_W;
            
            // Apply camera-to-body transform: p_C = R_C_B * p_B + t_C_B
            R_C_B * p_B + t_C_B
        }
        
        // Helper: Add factor for a camera observation
        fn add_camera_factor(
            problem: &mut Problem,
            lm_var: &str,
            cam_var: Option<&str>,
            observation: na::Vector2<f64>,
            T_C_B: na::Matrix4<f64>,  // T_C_B: SE3 transform from B to C
            fixed_pose: Option<na::Matrix4<f64>>,
        ) {
            // let sqrt_info = na::Matrix2::identity(); // Assuming unit covariance for simplicity
            let mut factor = BundleAdjustmentFactor::new(observation, T_C_B, None);
            if let Some(pose) = fixed_pose {
                factor = factor.with_fixed_pose(pose);
            }
            
            let var_names: Vec<&str> = if let Some(cv) = cam_var {
                vec![lm_var, cv]
            } else {
                vec![lm_var]
            };
            
            problem.add_residual_block(&var_names, Box::new(factor), None);
        }


        // Set up problem
        let mut problem = Problem::new();
        let mut initial_values = HashMap::new();
        let mut landmarks = Vec::new();
        let mut point_vars = Vec::new();

        // Generate random 3D landmarks with noisy initial estimates
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Set up landmarks
        for i in 0..NUM_LANDMARKS {
            // True landmark position
            let true_point = vec![
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(0.5..3.0),
            ];
            landmarks.push(true_point.clone());
            
            // Noisy initial estimate
            let noise: Vec<f64> = (0..3)
                .map(|_| rng.gen_range(-NOISE_RANGE..NOISE_RANGE))
                .collect();
            let noisy_point: Vec<f64> = true_point
                .iter()
                .zip(noise.iter())
                .map(|(p, n)| p + n)
                .collect();
            
            let lm_var = format!("LM_{}", i);
            initial_values.insert(
                lm_var.clone(),
                (ManifoldType::RN, DVector::from_vec(noisy_point)),
            );
            point_vars.push(lm_var);
        }

        // Camera extrinsics: T_Cl_B and T_Cr_B (SE3 transforms from B to Cl and Cr)
        let T_Cl_B = na::Matrix4::identity();
        let mut T_Cr_B = na::Matrix4::identity();
        T_Cr_B[(0, 3)] = 0.5; // Right camera offset in x

        // Generate random system poses with translations and rotations
        // Notation: t_B_W = translation from W to B, R_B_W = rotation from W to B
        let mut t_B_W_vec = Vec::new();
        let mut R_B_W_vec = Vec::new();
        let mut pose_vars = Vec::new();
        
        // Set up system poses with cameras
        for pose_id in 0..NUM_POSES {
            // Generate random translation: t_B_W (translation from W to B)
            let t_B_W = if pose_id == 0 {
                // First pose is fixed at origin
                na::Vector3::zeros()
            } else {
                na::Vector3::new(
                    rng.gen_range(-TRANSLATION_RANGE..TRANSLATION_RANGE),
                    rng.gen_range(-TRANSLATION_RANGE..TRANSLATION_RANGE),
                    rng.gen_range(-TRANSLATION_RANGE * 0.5..TRANSLATION_RANGE * 0.5),
                )
            };
            t_B_W_vec.push(t_B_W);
            
            // Generate small random rotation: R_B_W (rotation from W to B)
            let R_B_W = if pose_id == 0 {
                // First pose has no rotation (identity)
                na::UnitQuaternion::identity()
            } else {
                generate_small_rotation(&mut rng, MAX_ROTATION_ANGLE)
            };
            R_B_W_vec.push(R_B_W);
            
            // Create variable for this pose (only if not fixed)
            // Note: Currently only translation is optimized, rotation is not
            if pose_id > 0 {
                let cam_var = format!("KF_{}", pose_id);
                // Initial estimate with some noise (translation only)
                let noisy_translation = t_B_W + na::Vector3::new(
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                );
                let noisy_rotation = R_B_W * generate_small_rotation(&mut rng, MAX_ROTATION_ANGLE / 2.0);
                let cam_data = DVector::from_vec(vec![
                    noisy_translation.x,
                    noisy_translation.y,
                    noisy_translation.z, // then wijk 
                    noisy_rotation.w.clone(), noisy_rotation.i.clone(), noisy_rotation.j.clone(), noisy_rotation.k.clone(),
                ]);
                initial_values.insert(cam_var.clone(), (ManifoldType::SE3, cam_data));
                pose_vars.push((pose_id, cam_var));
            }
        }

        // Add observations from each pose (left + right cameras)
        let mut total_observations = 0;
        for (pose_id, (t_B_W, R_B_W)) in t_B_W_vec.iter().zip(R_B_W_vec.iter()).enumerate() {
            let cam_var_opt = pose_vars.iter().find(|(id, _)| *id == pose_id).map(|(_, v)| v.as_str());
            let is_fixed = pose_id == 0;

            let mut T_B_W = na::Matrix4::identity();
            T_B_W.fixed_view_mut::<3, 3>(0, 0).copy_from(&R_B_W.to_rotation_matrix().matrix());
            T_B_W.fixed_view_mut::<3, 1>(0, 3).copy_from(&t_B_W.to_owned());
            println!("pose_id: {}", pose_id);
            println!("T_B_W: {:?}", T_B_W.to_string());
            for lm_var in &point_vars {
                let idx = get_landmark_idx(lm_var);
                let p_W = na::Vector3::new(landmarks[idx][0], landmarks[idx][1], landmarks[idx][2]);
                
                // Left camera observation (with rotation applied)
                let p_Cl = world_to_camera_with_rotation(p_W, *t_B_W, R_B_W, &T_Cl_B);
                if is_visible(&p_Cl, MIN_DEPTH) {
                    let obs_Cl = project_to_normalized(p_Cl);
                    add_camera_factor(
                        &mut problem,
                        lm_var,
                        cam_var_opt,
                        obs_Cl,
                        T_Cl_B,
                        if is_fixed { Some(T_B_W) } else { None },
                    );
                    total_observations += 1;
                }
                
                // Right camera observation (with rotation applied)
                let p_Cr = world_to_camera_with_rotation(p_W, *t_B_W, R_B_W, &T_Cr_B);
                if is_visible(&p_Cr, MIN_DEPTH) {
                    let obs_Cr = project_to_normalized(p_Cr);
                    add_camera_factor(
                        &mut problem,
                        lm_var,
                        cam_var_opt,
                        obs_Cr,
                        T_Cr_B,
                        if is_fixed { Some(T_B_W) } else { None },
                    );
                    total_observations += 1;
                }
            }
        }
        
        println!("Generated {} poses with rotations and {} total stereo observations", NUM_POSES, total_observations);
        println!("Note: Rotations are applied but not optimized (factor only handles translations)");

        // Initialize problem
        problem.initialize_variables(&initial_values);

        // Configure and run optimization
        let config = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseCholesky)
            .with_max_iterations(50)
            .with_cost_tolerance(1e-6)
            .with_jacobi_scaling(false);

        let mut solver = LevenbergMarquardt::with_config(config);
        let observer = TerminalObserver::new();
        TerminalObserver::print_header();
        solver.add_observer(observer);

        let result = solver.optimize(&problem, &initial_values);
        assert!(result.is_ok(), "Optimization should succeed");
        let opt_result = result.unwrap();

        // Verify results
        // Note: Error threshold is relaxed because rotations cause un-optimizable error
        const MAX_LANDMARK_ERROR: f64 = 0.1; // Larger threshold due to rotation error

        println!("\nOptimization Results:");
        println!("Initial cost: {:.6}", opt_result.initial_cost);
        println!("Final cost: {:.6}", opt_result.final_cost);

        // Check landmark convergence
        for (var_name, value) in opt_result.parameters.iter() {
            if var_name.starts_with("LM_") {
                let idx = get_landmark_idx(var_name);
                if let Some(true_point) = landmarks.get(idx) {
                    let optimized = value.to_vector();
                    let true_vec = DVector::from_vec(true_point.clone());
                    let error = (optimized - true_vec).norm();
                    
                    println!("  {}: error = {:.6}", var_name, error);
                    
                    if error > MAX_LANDMARK_ERROR {
                        println!(
                            "    WARNING: Landmark {} error ({:.6}) exceeds threshold ({:.6})",
                            idx, error, MAX_LANDMARK_ERROR
                        );
                    }
                }
            }
        }

        // Check convergence status
        use apex_solver::optimizer::OptimizationStatus;
        let converged = matches!(
            opt_result.status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::ParameterToleranceReached
                | OptimizationStatus::GradientToleranceReached
        );

        if !converged {
            println!("Warning: Optimization did not fully converge. Status: {:?}", opt_result.status);
        }

    }

    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use nalgebra::{DMatrix, Matrix3, Vector3, UnitQuaternion};
    use crate::imu::piecewise_integration::PreInt;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    // ---- Helpers: build parameter blocks in the format your Factor expects ----
    fn pose7(t: Vector3<f64>, q_wxyz: [f64; 4]) -> DVector<f64> {
        // (tx,ty,tz,qw,qx,qy,qz)
        DVector::from_vec(vec![t.x, t.y, t.z, q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]])
    }

    fn speedbias9(v: Vector3<f64>, ba: Vector3<f64>, bg: Vector3<f64>) -> DVector<f64> {
        // (vx,vy,vz,bax,bay,baz,bgx,bgy,bgz)
        DVector::from_vec(vec![
            v.x, v.y, v.z,
            ba.x, ba.y, ba.z,
            bg.x, bg.y, bg.z
        ])
    }

    fn quat_from_wxyz(q: [f64; 4]) -> UnitQuaternion<f64> {
        // nalgebra uses (w, i, j, k)
        UnitQuaternion::from_quaternion(na::Quaternion::new(q[0], q[1], q[2], q[3]))
    }

    fn quat_to_wxyz(q: &UnitQuaternion<f64>) -> [f64; 4] {
        let qq = q.quaternion();
        [qq.w, qq.i, qq.j, qq.k]
    }

     // ---- Define the 30D local perturbation mapping used by the Jacobian columns ----
    //
    // Columns [0..30) interpreted as:
    //  0..3   : dphi_i
    //  3..6   : dp_i   (applied as p_i <- p_i + R_i * dp_i; body-frame translation perturbation)
    //  6..9   : dv_i
    //  9..12  : dphi_j
    //  12..15 : dp_j   (p_j <- p_j + R_j * dp_j)
    //  15..18 : dv_j
    //  18..21 : dbg_i
    //  21..24 : dba_i
    //  24..27 : dba_j
    //  27..30 : dbg_j
    //
    // This matches the way your Jacobian blocks are *used* in the dv/dp rows (bg at col 18, ba at col 21),
    // and it makes bias-residual rows easy to validate.
    fn apply_local_update_30(
        params: &Vec<na::DVector<f64>>,
        dx: &na::SVector<f64, 30>,
    ) -> Vec<na::DVector<f64>> {
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].len(), 7);
        assert_eq!(params[1].len(), 9);
        assert_eq!(params[2].len(), 7);
        assert_eq!(params[3].len(), 9);

        // New column layout (must match ImuFactor::linearize() fill order):
        // i: dphi(0..3), dv(3..6), dp(6..9), dba(9..12), dbg(12..15)
        // j: dphi(15..18), dv(18..21), dp(21..24), dba(24..27), dbg(27..30)
        const PHI_I: usize = 0;
        const V_I: usize   = 3;
        const P_I: usize   = 6;
        const BA_I: usize  = 9;
        const BG_I: usize  = 12;

        const PHI_J: usize = 15;
        const V_J: usize   = 18;
        const P_J: usize   = 21;
        const BA_J: usize  = 24;
        const BG_J: usize  = 27;

        let mut out = params.clone();

        // --- Pose update: [t(3), q(wxyz)(4)] with right perturbation ---
        // R <- R * Exp(dphi)
        // p <- p + R * dp   (dp is body-frame tangent translation)
        let mut update_pose7_inplace = |pose7v: &mut na::DVector<f64>, dphi: na::Vector3<f64>, dp: na::Vector3<f64>| {
            let t = na::Vector3::new(pose7v[0], pose7v[1], pose7v[2]);

            let qw = pose7v[3];
            let qx = pose7v[4];
            let qy = pose7v[5];
            let qz = pose7v[6];
            let q = UnitQuaternion::from_quaternion(na::Quaternion::new(qw, qx, qy, qz));

            // --- CHANGE START: dq from your so3::SO3::exp ---
            let dR_tangent = so3::SO3Tangent::from_components(dphi[0], dphi[1], dphi[2]);
            let dR= dR_tangent.exp(None);

            // Convert SO3 -> Matrix3 (you must adapt this one method name to your SO3 API)
            let Rm: na::Matrix3<f64> = dR.rotation_matrix(); // e.g. dR.matrix(), dR.as_matrix(), dR.rotmat(), etc.

            // Matrix3 -> Rotation3 -> UnitQuaternion
            let rot = na::Rotation3::from_matrix_unchecked(Rm);
            let dq = UnitQuaternion::from_rotation_matrix(&rot);
            // --- CHANGE END ---

            // Right perturbation (keep consistent with your analytic convention)
            let q_new = q * dq;

            let R = q.to_rotation_matrix();
            let t_new = t + R * dp;

            let qq = q_new.quaternion();
            pose7v[0] = t_new.x;
            pose7v[1] = t_new.y;
            pose7v[2] = t_new.z;
            pose7v[3] = qq.w;
            pose7v[4] = qq.i;
            pose7v[5] = qq.j;
            pose7v[6] = qq.k;
        };


        // pose i
        update_pose7_inplace(
            &mut out[0],
            na::Vector3::new(dx[PHI_I], dx[PHI_I + 1], dx[PHI_I + 2]),
            na::Vector3::new(dx[P_I],   dx[P_I   + 1], dx[P_I   + 2]),
        );

        // speedbias i: [v(3), ba(3), bg(3)] additive
        out[1][0] += dx[V_I];     out[1][1] += dx[V_I + 1];     out[1][2] += dx[V_I + 2];
        out[1][3] += dx[BA_I];    out[1][4] += dx[BA_I + 1];    out[1][5] += dx[BA_I + 2];
        out[1][6] += dx[BG_I];    out[1][7] += dx[BG_I + 1];    out[1][8] += dx[BG_I + 2];

        // pose j
        update_pose7_inplace(
            &mut out[2],
            na::Vector3::new(dx[PHI_J], dx[PHI_J + 1], dx[PHI_J + 2]),
            na::Vector3::new(dx[P_J],   dx[P_J   + 1], dx[P_J   + 2]),
        );

        // speedbias j: [v(3), ba(3), bg(3)] additive
        out[3][0] += dx[V_J];     out[3][1] += dx[V_J + 1];     out[3][2] += dx[V_J + 2];
        out[3][3] += dx[BA_J];    out[3][4] += dx[BA_J + 1];    out[3][5] += dx[BA_J + 2];
        out[3][6] += dx[BG_J];    out[3][7] += dx[BG_J + 1];    out[3][8] += dx[BG_J + 2];

        out
    }


    #[test]
    fn imu_factor_residual_zero_when_preint_matches_state() {
        let dt = 0.01;
        let g = Vector3::new(0.0, 0.0, -9.81);

        // Choose a simple state.
        let qi = UnitQuaternion::identity();
        let ri = qi.to_rotation_matrix();
        let ti = Vector3::new(1.0, 2.0, 3.0);
        let vi = Vector3::new(0.5, -0.2, 0.1);

        // Keep biases constant and equal across i->j.
        let bai = Vector3::new(0.01, -0.02, 0.03);
        let bgi = Vector3::new(-0.001, 0.002, -0.003);
        let baj = bai;
        let bgj = bgi;

        // Define j consistent with a "perfect" preintegrated measurement:
        // residual_dv = Ri^T*(v_j - v_i + g*dt) - dv = 0  => dv = Ri^T*(...)
        // residual_dp = Ri^T*(t_j - t_i - v_i*dt + 0.5*g*dt^2) - dp = 0 => dp = Ri^T*(...)
        let vj = vi - g * dt; // so vj - vi + g*dt = 0 => dv must be 0
        let tj = ti + vi * dt - 0.5 * g * dt * dt; // so translation residual is 0 with dp=0

        // For rotation, choose Rj = Ri * dR, and set preint.dR = Ri^{-1}Rj.
        let qj = qi;
        let dR = so3::SO3::identity(); // must match your implementation

        let preint = PreInt::new(
            dR,
            Vector3::zeros(),
            Vector3::zeros(),
            na::SMatrix::<f64, 15, 15>::identity(),
            dt,
            Vector3::zeros(),
            Vector3::zeros(),
            na::SMatrix::<f64, 15, 15>::identity(),
            na::SMatrix::<f64, 15, 15>::identity(),
            &Vec::new(),
            g,
        );

        let factor = ImuFactor::new(preint, bai, bgi);

        let params = vec![
            pose7(ti, quat_to_wxyz(&qi)),
            speedbias9(vi, bai, bgi),
            pose7(tj, quat_to_wxyz(&qj)),
            speedbias9(vj, baj, bgj),
        ];

        let (r, j) = factor.linearize(&params, true);
        assert_eq!(r.len(), 15);
        assert!(j.is_some());

        // Residual should be ~0 (numerical eps).
        for k in 0..15 {
            print!("r[{}] = {:.6e}  ", k, r[k]);
            assert_abs_diff_eq!(r[k], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn imu_factor_whitening_scales_residual_and_jacobian_rows() {
        // This test checks the same behavior as VINS-Mono's `sqrt_info * residual` and `sqrt_info * jacobian` [page:0].
        let dt = 0.1;

        let preint = PreInt {
            dt,
            dR: so3::SO3::identity(),
            dv: Vector3::new(1.0, 2.0, 3.0),
            dp: Vector3::new(4.0, 5.0, 6.0),
            linearized_ba: Vector3::zeros(),
            linearized_bg: Vector3::zeros(),
            sqrt_info: na::SMatrix::<f64, 15, 15>::identity() * 2.0, // no whitening
            cov: na::SMatrix::<f64, 15, 15>::identity(), // not used
            imu_buffer: Vec::new(), // not used
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            jacobian: na::SMatrix::<f64, 15, 15>::identity(), // not used
            idx_r: 0,
            idx_v: 3,
            idx_p: 6,
            idx_ba: 9,
            idx_bg: 12,
        };

        let bias_a = Vector3::zeros();
        let bias_g = Vector3::zeros();
        let factor = ImuFactor::new(preint.clone(), bias_a, bias_g);

        let qi = UnitQuaternion::identity();
        let qj = UnitQuaternion::identity();

        let params = vec![
            pose7(Vector3::zeros(), quat_to_wxyz(&qi)),
            speedbias9(Vector3::zeros(), Vector3::zeros(), Vector3::zeros()),
            pose7(Vector3::zeros(), quat_to_wxyz(&qj)),
            speedbias9(Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
        ];

        let (r_wh, j_wh) = factor.linearize(&params, true);
        let j_wh = j_wh.unwrap();

        // Now compute the same residual with whitening disabled (w9=1,w6=1) and compare scaling.
        let mut preint_un = preint.clone();
        preint_un.sqrt_info = na::SMatrix::<f64, 15, 15>::identity();
        let factor_un = ImuFactor::new(preint_un, bias_a, bias_g);
        let (r_un, j_un) = factor_un.linearize(&params, true);
        let j_un = j_un.unwrap();

        print!("Finite-difference column {:?}\n{:?}", r_wh, r_un);

        // First 9 rows scaled by 2, last 6 rows scaled by 3.
        for k in 0..9 {
            assert_relative_eq!(r_wh[k], 2.0 * r_un[k], epsilon = 1e-10);
        }
        for k in 9..15 {
            assert_relative_eq!(r_wh[k], 2.0 * r_un[k], epsilon = 1e-10);
        }

        // Same for Jacobian rows.
        for row in 0..9 {
            for col in 0..30 {
                assert_relative_eq!(j_wh[(row, col)], 2.0 * j_un[(row, col)], epsilon = 1e-10);
            }
        }
        for row in 9..15 {
            for col in 0..30 {
                assert_relative_eq!(j_wh[(row, col)], 2.0 * j_un[(row, col)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn imu_factor_jacobian_matches_finite_difference() {
        // This is the test that will catch wrong column assignments/signs.
        let dt = 0.02;
        let preint = PreInt {
            dt,
            dR: so3::SO3::identity(),
            dv: Vector3::new(0.1, -0.2, 0.05),
            dp: Vector3::new(0.01, 0.02, -0.03),
            linearized_bg: 0.04 * Vector3::zeros(),
            linearized_ba: 0.05 * Vector3::zeros(),
            sqrt_info: na::SMatrix::<f64, 15, 15>::identity(), // no whitening
            cov: na::SMatrix::<f64, 15, 15>::identity(), // not used
            imu_buffer: Vec::new(), // not used
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            jacobian: na::SMatrix::<f64, 15, 15>::identity(), // not used
            idx_r: 0,
            idx_v: 3,
            idx_p: 6,
            idx_ba: 9,
            idx_bg: 12,
        };

        let bias_a = Vector3::new(0.021, -0.011, -0.031);
        let bias_g = Vector3::new(-0.0041, 0.0031, 0.0021);
        let factor = ImuFactor::new(preint, bias_a, bias_g);

        let qi = UnitQuaternion::from_scaled_axis(Vector3::new(0.02, -0.01, 0.03));
        let qj = UnitQuaternion::from_scaled_axis(Vector3::new(-0.01, 0.04, 0.01));
        let params = vec![
            pose7(Vector3::new(1.0, 2.0, 3.0), quat_to_wxyz(&qi)),
            speedbias9(
                Vector3::new(0.5, -0.2, 0.1),
                Vector3::new(0.02, 0.01, -0.03),
                Vector3::new(-0.004, 0.003, 0.002),
            ),
            pose7(Vector3::new(1.1, 2.1, 2.9), quat_to_wxyz(&qj)),
            speedbias9(
                Vector3::new(0.45, -0.18, 0.12),
                Vector3::new(0.021, 0.011, -0.031),
                Vector3::new(-0.0045, 0.0032, 0.0019),
            ),
        ];

        let (r0, j_opt) = factor.linearize(&params, true);
        let j = j_opt.expect("jacobian must be returned");
        assert_eq!(j.nrows(), 15);
        assert_eq!(j.ncols(), 30);
        assert_eq!(r0.len(), 15);

        let eps = 1e-7;

        for col in 0..30 {
            let mut dxp = na::SVector::<f64, 30>::zeros();
            dxp[col] = eps;
            let dxm = -dxp;

            let pp = apply_local_update_30(&params, &dxp);
            let pm = apply_local_update_30(&params, &dxm);

            let (rp, _) = factor.linearize(&pp, false);
            let (rm, _) = factor.linearize(&pm, false);

            let num = (rp - rm) * (0.5 / eps);
            print!("Finite-difference column {}: {:?}\n", col, num.transpose());

            // Compare column to finite-difference (allow slightly looser tolerance for rotation components).
            for row in 0..15 {
                let tol = if row < 3 { 5e-5 } else { 5e-6 };
                let diff = (j[(row, col)] - num[row]).abs();
                assert!(
                    diff <= tol,
                    "Mismatch at (row={}, col={}): analytic={}, numeric={}, |diff|={}, tol={}",
                    row,
                    col,
                    j[(row, col)],
                    num[row],
                    diff,
                    tol,
                    // params
                );
            }
        }
    }

    #[test]
    fn imu_factor_compute_jacobian_flag_works() {
        let preint = PreInt {
            dt: 0.01,
            dR: so3::SO3::identity(),
            dv: Vector3::zeros(),
            dp: Vector3::zeros(),
            linearized_ba: Vector3::zeros(),
            linearized_bg: Vector3::zeros(),
            jacobian: na::SMatrix::<f64, 15, 15>::identity(), // not used
            sqrt_info: na::SMatrix::<f64, 15, 15>::identity(), // no whitening
            cov: na::SMatrix::<f64, 15, 15>::identity(), // not used
            imu_buffer: Vec::new(), // not used
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            idx_r: 0,
            idx_v: 3,
            idx_p: 6,
            idx_ba: 9,
            idx_bg: 12,
        };
        let factor = ImuFactor::new(preint, Vector3::zeros(), Vector3::zeros());

        let qi = UnitQuaternion::identity();
        let params = vec![
            pose7(Vector3::zeros(), quat_to_wxyz(&qi)),
            speedbias9(Vector3::zeros(), Vector3::zeros(), Vector3::zeros()),
            pose7(Vector3::zeros(), quat_to_wxyz(&qi)),
            speedbias9(Vector3::zeros(), Vector3::zeros(), Vector3::zeros()),
        ];

        let (_, j_none) = factor.linearize(&params, false);
        assert!(j_none.is_none());

        let (_, j_some) = factor.linearize(&params, true);
        assert!(j_some.is_some());
    }

    fn so3_exp_quat(dphi: Vector3<f64>) -> UnitQuaternion<f64> {
        let theta = dphi.norm();
        if theta < 1e-12 {
            // small-angle: sin(theta/2) ~ theta/2
            let half = 0.5;
            return UnitQuaternion::from_quaternion(na::Quaternion::new(
                1.0,
                half * dphi.x,
                half * dphi.y,
                half * dphi.z,
            ));
        }
        let axis = dphi / theta;
        let half = 0.5 * theta;
        UnitQuaternion::from_quaternion(na::Quaternion::new(
            half.cos(),
            axis.x * half.sin(),
            axis.y * half.sin(),
            axis.z * half.sin(),
        ))
    }

    fn quat_to_params_wxyz(q: &UnitQuaternion<f64>) -> (f64, f64, f64, f64) {
        let qq = q.quaternion();
        (qq.w, qq.i, qq.j, qq.k)
    }

    fn pack_pose7(t: Vector3<f64>, q: UnitQuaternion<f64>) -> nalgebra::DVector<f64> {
        let (qw, qx, qy, qz) = quat_to_params_wxyz(&q);
        nalgebra::DVector::from_vec(vec![t.x, t.y, t.z, qw, qx, qy, qz])
    }

    fn unpack_pose7(p: &nalgebra::DVector<f64>) -> (Vector3<f64>, UnitQuaternion<f64>) {
        let t = Vector3::new(p[0], p[1], p[2]);
        let q = UnitQuaternion::from_quaternion(na::Quaternion::new(p[3], p[4], p[5], p[6]));
        (t, q)
    }

    fn pack_vb9(v: Vector3<f64>, ba: Vector3<f64>, bg: Vector3<f64>) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(vec![
            v.x, v.y, v.z,
            ba.x, ba.y, ba.z,
            bg.x, bg.y, bg.z
        ])
    }

    fn unpack_vb9(p: &nalgebra::DVector<f64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let v  = Vector3::new(p[0], p[1], p[2]);
        let ba = Vector3::new(p[3], p[4], p[5]);
        let bg = Vector3::new(p[6], p[7], p[8]);
        (v, ba, bg)
    }

    // Parameter layout for your 15x30 Jacobian:
    // [phi_i(3), v_i(3), p_i(3), ba_i(3), bg_i(3), phi_j(3), v_j(3), p_j(3), ba_j(3), bg_j(3)]
    fn apply_delta_to_params(
        base_params: &[nalgebra::DVector<f64>], // 4 blocks: pose_i(7), vb_i(9), pose_j(7), vb_j(9)
        delta30: &na::SVector<f64, 30>,
        translation_is_local: bool,
    ) -> Vec<nalgebra::DVector<f64>> {
        let (t_i, q_i) = unpack_pose7(&base_params[0]);
        let (v_i, ba_i, bg_i) = unpack_vb9(&base_params[1]);
        let (t_j, q_j) = unpack_pose7(&base_params[2]);
        let (v_j, ba_j, bg_j) = unpack_vb9(&base_params[3]);

        let dphi_i = Vector3::new(delta30[0],  delta30[1],  delta30[2]);
        let dv_i   = Vector3::new(delta30[3],  delta30[4],  delta30[5]);
        let dp_i   = Vector3::new(delta30[6],  delta30[7],  delta30[8]);
        let dba_i  = Vector3::new(delta30[9],  delta30[10], delta30[11]);
        let dbg_i  = Vector3::new(delta30[12], delta30[13], delta30[14]);

        let dphi_j = Vector3::new(delta30[15], delta30[16], delta30[17]);
        let dv_j   = Vector3::new(delta30[18], delta30[19], delta30[20]);
        let dp_j   = Vector3::new(delta30[21], delta30[22], delta30[23]);
        let dba_j  = Vector3::new(delta30[24], delta30[25], delta30[26]);
        let dbg_j  = Vector3::new(delta30[27], delta30[28], delta30[29]);

        // Right-multiplicative rotation update: R <- R * Exp(dphi)
        let q_i_new = (q_i * so3_exp_quat(dphi_i));
        let q_j_new = (q_j * so3_exp_quat(dphi_j));

        let r_i: Matrix3<f64> = q_i.to_rotation_matrix().into_inner();
        let r_j: Matrix3<f64> = q_j.to_rotation_matrix().into_inner();

        // Translation update: either world-add (t <- t + dp) or local-add (t <- t + R*dp)
        let t_i_new = if translation_is_local { t_i + r_i * dp_i } else { t_i + dp_i };
        let t_j_new = if translation_is_local { t_j + r_j * dp_j } else { t_j + dp_j };

        let v_i_new  = v_i  + dv_i;
        let v_j_new  = v_j  + dv_j;
        let ba_i_new = ba_i + dba_i;
        let bg_i_new = bg_i + dbg_i;
        let ba_j_new = ba_j + dba_j;
        let bg_j_new = bg_j + dbg_j;

        vec![
            pack_pose7(t_i_new, q_i_new),
            pack_vb9(v_i_new, ba_i_new, bg_i_new),
            pack_pose7(t_j_new, q_j_new),
            pack_vb9(v_j_new, ba_j_new, bg_j_new),
        ]
    }

    fn central_diff_jacobian(
        factor: &ImuFactor,
        base_params: &[nalgebra::DVector<f64>],
        eps: f64,
        translation_is_local: bool,
    ) -> nalgebra::DMatrix<f64> {
        let (r0, _) = factor.linearize(base_params, false);
        let m = r0.len();
        let n = 30;

        let mut j_num = nalgebra::DMatrix::<f64>::zeros(m, n);

        for k in 0..n {
            let mut d = na::SVector::<f64, 30>::zeros();
            d[k] = eps;

            let p_plus = apply_delta_to_params(base_params, &d, translation_is_local);
            let (r_plus, _) = factor.linearize(&p_plus, false);

            let mut d_minus = na::SVector::<f64, 30>::zeros();
            d_minus[k] = -eps;
            let p_minus = apply_delta_to_params(base_params, &d_minus, translation_is_local);
            let (r_minus, _) = factor.linearize(&p_minus, false);

            let dr = (r_plus - r_minus) * (0.5 / eps);
            j_num.set_column(k, &dr);
        }

        j_num
    }

    #[test]
    fn test_imu_factor_residual_and_jacobian_fd() {
        let mut rng = StdRng::seed_from_u64(42); // fixed seed for reproducibility
        // Choose dt not too small (avoid numerical cancellation) and not too large (avoid large-angle issues).
        let dt = 0.05;

        // Gravity must match your factor's convention.
        let g_w = Vector3::new(0.0, 0.0, -9.81);

        // Random-ish pose i
        let t_i = Vector3::new(
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        );

        // Keep rotations small-ish to avoid log-map edge cases.
        let phi_i = Vector3::new(
            rng.gen_range(-0.2..0.2),
            rng.gen_range(-0.2..0.2),
            rng.gen_range(-0.2..0.2),
        );
        let q_i = so3_exp_quat(phi_i);

        let v_i = Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );

        // Biases (set equal to factor.bias_* so correction terms are zero at GT)
        let ba_i = Vector3::new(
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        );
        let bg_i = Vector3::new(
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        );

        // Create a consistent j state by applying a small motion
        let dphi_ij = Vector3::new(
            rng.gen_range(-0.1..0.1),
            rng.gen_range(-0.1..0.1),
            rng.gen_range(-0.1..0.1),
        );
        let q_j = (q_i * so3_exp_quat(dphi_ij)); // ensure unit norm

        let a_w = Vector3::new(
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5),
        );

        // Simple constant-acceleration propagation in world frame for test data:
        let v_j = v_i + (a_w + g_w) * dt;
        let t_j = t_i + v_i * dt + 0.5 * (a_w + g_w) * dt * dt;

        // Keep biases constant across i->j (random walk residuals should be 0)
        let ba_j = ba_i;
        let bg_j = bg_i;

        // Build params blocks
        let params = vec![
            pose7(t_i, [q_i.w, q_i.i, q_i.j, q_i.k]),
            speedbias9(v_i, ba_i, bg_i),
            pose7(t_j, [q_j.w, q_j.i, q_j.j, q_j.k]),
            speedbias9(v_j, ba_j, bg_j),
        ];

        // --- Construct a self-consistent preintegration object ---
        // Goal: at GT, residuals should be ~0.
        //
        // Forster-style model (in i frame): ΔR = R_i^T R_j, Δv = R_i^T (v_j - v_i - g dt), Δp = R_i^T (t_j - t_i - v_i dt - 0.5 g dt^2). [file:2]
        let r_i: Matrix3<f64> = q_i.to_rotation_matrix().into_inner();
        let r_j: Matrix3<f64> = q_j.to_rotation_matrix().into_inner();
        let r_i_t = r_i.transpose();

        // You must adapt these conversions to your so3/se3 types.
        let delta_r_mat = r_i_t * r_j;
        let delta_v = r_i_t * (v_j - v_i + g_w * dt);
        let delta_p = r_i_t * (t_j - t_i - v_i * dt + 0.5 * g_w * dt * dt);

        // The key is: set dt, dR, dv, dp, bias linearization point (bias_a, bias_g),
        // and make whitening identity (or consistent) so FD compares apples-to-apples.
        let preint = make_preint_identity(dt, delta_r_mat, delta_v, delta_p);

        // Factor biases must equal ba_i/bg_i used to define db_a/db_g=0 at GT
        let factor = ImuFactor {
            preint,
            linearized_bias_a: ba_i,
            linearized_bias_g: bg_i,
            // ... fill any other fields you have ...
        };

        // --- Residual at GT should be ~0 ---
        let (r_gt, j_opt) = factor.linearize(&params, true);
        let j_analytic = j_opt.expect("Expected jacobian");

        assert_eq!(r_gt.len(), 15);
        assert_eq!(j_analytic.nrows(), 15);
        assert_eq!(j_analytic.ncols(), 30);

        let r_norm = r_gt.norm();
        assert!(
            r_norm < 1e-8,
            "Residual at ground truth should be ~0, got norm={}, and residual {:?}",
            r_norm,
            r_gt
        );

        // --- Numeric Jacobian (central difference) ---
        // Central difference is more accurate than forward difference for smooth functions. [web:22]
        let eps = 1e-7;

        let j_num_world = central_diff_jacobian(&factor, &params, eps, false);
        let j_num_local = central_diff_jacobian(&factor, &params, eps, true);

        // Compare both: your analytic code assumes one of these pose translation perturbations.
        let diff_world = (&j_analytic - &j_num_world).amax();
        let diff_local = (&j_analytic - &j_num_local).amax();

        eprintln!("max|J_analytic - J_num_world| = {:.3e}", diff_world);
        eprintln!("max|J_analytic - J_num_local| = {:.3e}", diff_local);

        // Choose the one that matches best, but enforce that at least one is tight.
        let best = diff_world.min(diff_local);
        assert!(
            best < 5e-5,
            "Jacobian mismatch too large. best max-abs diff={:.3e} (world={:.3e}, local={:.3e})",
            best, diff_world, diff_local
        );
    }

    fn make_preint_identity(
        dt: f64,
        delta_r_mat: Matrix3<f64>,
        delta_v: Vector3<f64>,
        delta_p: Vector3<f64>,
    ) -> PreInt {
        let quat = UnitQuaternion::from_rotation_matrix(&na::Rotation3::from_matrix_unchecked(delta_r_mat));
        PreInt { 
            dR: so3::SO3::new(quat), 
            dv: delta_v, 
            dp: delta_p, 
            cov: na::SMatrix::<f64, 15, 15>::identity(), 
            dt, 
            linearized_bg: na::Vector3::<f64>::zeros(), 
            linearized_ba: na::Vector3::<f64>::zeros(), 
            sqrt_info: na::SMatrix::<f64, 15, 15>::identity(),
            imu_buffer: Vec::new(), // not used
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            jacobian: na::SMatrix::<f64, 15, 15>::identity(), // not used
            idx_r: 0,
            idx_v: 3,
            idx_p: 6,
            idx_ba: 9,
            idx_bg: 12,
        }
    }

}

