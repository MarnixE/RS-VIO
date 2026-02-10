use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};
use image::{ImageReader};
use crate::viewers::{Viewer, create_viewer};
use crate::estimator::Estimator;
use crate::datasets::{config::Config, ImageData, ImuData, FrameContext, PlayerConfig, PlayerResult};


pub struct FourSeasonsPlayer;

impl FourSeasonsPlayer {
    pub fn new() -> Self {
        FourSeasonsPlayer
    }

    pub fn run(&self, config: PlayerConfig) -> PlayerResult {
        let mut result = PlayerResult::default();

        // Load image timestamps
        let image_data = match Self::load_image_timestamps(&config.dataset_path) {
            Ok(data) => {
                if data.is_empty() {
                    result.error_message = "No images found in dataset".to_string();
                    return result;
                }
                data
            }
            Err(e) => {
                result.error_message = format!("Failed to load image timestamps: {}", e);
                return result;
            }
        };

        let start_frame_idx = 0;
        let end_frame_idx = image_data.len();

        // Initialize viewer
        let mut viewer: Option<Box<dyn Viewer>> = match create_viewer() {
            Ok(v) => {
                log::info!("[EurocPlayer] Viewer initialized successfully");
                Some(v)
            }
            Err(e) => {
                log::warn!("Failed to initialize viewer: {}", e);
                None
            }
        };

        // Load full YAML config
        let cfg = match Config::load(&config.config_path) {
            Ok(c) => c,
            Err(e) => {
                result.error_message = format!("Failed to load config '{}': {}", config.config_path, e);
                return result;
            }
        };

        // Create camera models from config
        let (left_cam, right_cam) = match Self::create_camera_models_from_config(&cfg) {
            Ok(cams) => cams,
            Err(e) => {
                result.error_message = format!("Failed to create camera models: {}", e);
                return result;
            }
        };

        // Give ownership of the configuration to the estimator and pass a
        // reference to the viewer (which outlives the estimator).
        let mut estimator = {
            let viewer_ref: Option<&mut dyn Viewer> =
                viewer.as_deref_mut().map(|v| v as &mut dyn Viewer);
            Estimator::new_with_cameras(cfg, viewer_ref, Some(left_cam), Some(right_cam), None)
        };
        Self::initialize_estimator(&mut estimator, &image_data);

        // Process frames
        let mut context = FrameContext::new(config.step_mode);

        context.current_idx = start_frame_idx;
        while context.current_idx < end_frame_idx {
            let should_process_frame = if context.auto_play {
                // Auto mode: process frame
                true
            } else {
                // Step mode: only process if advance_frame is set
                if context.advance_frame {
                    context.advance_frame = false;
                    true
                } else {
                    // In step mode with no advance request, just wait
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }
            };

            if should_process_frame {
                // Process single frame
                let frame_start = Instant::now();
                let _processing_time = match Self::process_single_frame(
                    &mut estimator,
                    &mut context,
                    &image_data,
                    &config.dataset_path,
                ) {
                    Ok(time) => time,
                    Err(e) => {
                        log::warn!("Error processing frame {}: {}", context.current_idx, e);
                        0.0
                    }
                };

                let frame_duration = frame_start.elapsed();
                let total_time_ms = frame_duration.as_secs_f64() * 1000.0;
                result.frame_processing_times.push(total_time_ms);

                context.current_idx += 1;
                context.processed_frames += 1;

                // Calculate sleep time based on actual frame intervals (only in auto mode)
                if context.auto_play && context.current_idx < end_frame_idx {
                    let current_timestamp = image_data[context.current_idx - 1].timestamp;
                    let next_timestamp = image_data[context.current_idx].timestamp;
                    let frame_interval_ms = (next_timestamp - current_timestamp) as f64 / 1e6; // nanoseconds to milliseconds

                    let sleep_time_ms = (frame_interval_ms - total_time_ms).max(0.0);
                    if sleep_time_ms > 0.0 {
                        thread::sleep(Duration::from_millis(sleep_time_ms as u64));
                    }
                }
            }
        }

        // Save results
        if config.enable_statistics {
            Self::save_trajectories(&estimator, &context, &config.dataset_path);
            Self::save_statistics(&result, &config.dataset_path);
        }

        // Calculate final statistics
        result.success = true;
        result.processed_frames = context.processed_frames;

        if !result.frame_processing_times.is_empty() {
            result.average_processing_time_ms = result.frame_processing_times.iter().sum::<f64>()
                / result.frame_processing_times.len() as f64;

            log::info!(
                "[4SeasonsPlayer] Average processing time: {:.2} ms ({:.1} fps)",
                result.average_processing_time_ms,
                1000.0 / result.average_processing_time_ms
            );
        }

        // Display final statistics summary
        if config.enable_console_statistics && result.success {
            log::info!("════════════════════════════════════════════════════════════════════");
            log::info!("                          STATISTICS                                ");
            log::info!("════════════════════════════════════════════════════════════════════");
            log::info!("");
            log::info!("                          TIMING ANALYSIS                           ");
            log::info!("════════════════════════════════════════════════════════════════════");
            log::info!(" Total Frames Processed: {}", result.processed_frames);
            log::info!(" Average Processing Time: {:.2}ms", result.average_processing_time_ms);
            let fps = 1000.0 / result.average_processing_time_ms;
            log::info!(" Average Frame Rate: {:.1}fps", fps);
            log::info!("════════════════════════════════════════════════════════════════════");
        }
        
        log::info!("[4SeasonsPlayer] Processing completed! Viewer remains open for inspection.");
        
        result
    }

    fn load_image_timestamps(dataset_path: &str) -> Result<Vec<ImageData>> {
        let data_file = Path::new(dataset_path).join("times.txt");
        let file = File::open(&data_file)
            .with_context(|| format!("Cannot open times.txt file: {}", data_file.display()))?;

        let reader = BufReader::new(file);
        let mut image_data = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            
            // Skip header and empty lines
            if line_num == 0 || line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }

            // Format: timestamp filename timestamp (space-separated)
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let timestamp_str = parts[0].trim();
                let filename = parts[0].trim().to_string() + ".png";
                if let Ok(timestamp) = timestamp_str.parse::<i64>() {
                    image_data.push(ImageData {
                        timestamp,
                        filename,
                    });
                }
            }
        }

        log::info!("[4SeasonsPlayer] Loaded {} image timestamps", image_data.len());
        Ok(image_data)
    }

    fn load_image(dataset_path: &str, filename: &str, cam_id: u32) -> Result<Vec<u8>> {
        let cam_folder = if cam_id == 0 { "cam0" } else { "cam1" };
        let full_path = Path::new(dataset_path)
            .join("undistorted_images")
            .join(cam_folder)
            .join(filename);

        if !full_path.exists() {
            anyhow::bail!("Cannot load image: {}", full_path.display());
        }

        // Load image using image crate
        let img = ImageReader::open(&full_path)
            .with_context(|| format!("Failed to open image: {}", full_path.display()))?
            .decode()
            .with_context(|| format!("Failed to decode image: {}", full_path.display()))?;

        // Convert to grayscale if needed (EuRoC images are typically grayscale)
        let gray_img = img.to_luma8();

        // Return raw pixel data as Vec<u8>
        let pixel_data = gray_img.as_raw().to_vec();
        
        Ok(pixel_data)
    }

    fn load_imu_data(
        _dataset_path: &str,
        _image_data: &[ImageData],
        _start_frame_idx: usize,
        _end_frame_idx: usize,
    ) -> Result<()> {
        // TODO: Implement IMU data loading
        log::info!("[4SeasonsPlayer] IMU data loading (placeholder)");
        Ok(())
    }

    /// Create camera models from config using the datasets module helper function
    fn create_camera_models_from_config(
        config: &Config,
    ) -> Result<(crate::datasets::CameraModelType, crate::datasets::CameraModelType)> {
        Ok(crate::datasets::create_camera_models_from_config(config))
    }

    fn initialize_estimator<'a>(_estimator: &mut Estimator<'a>, _image_data: &[ImageData]) {
        // TODO: Set initial pose if needed
        // For now, just a placeholder
        log::debug!("[4SeasonsPlayer] Estimator initialized");
    }

    fn process_single_frame<'a>(
        estimator: &mut Estimator<'a>,
        context: &mut FrameContext,
        image_data: &[ImageData],
        dataset_path: &str,
    ) -> Result<f64> {
        let frame_start = Instant::now();

        // Inform the estimator about the current frame index for visualization.
        estimator.set_viewer_frame(context.current_idx as i64);

        // Load stereo images
        let left_image = Self::load_image(dataset_path, &image_data[context.current_idx].filename, 0)?;
        let right_image = Self::load_image(dataset_path, &image_data[context.current_idx].filename, 1)?;

        if left_image.is_empty() {
            anyhow::bail!("Skipping frame {} due to empty image", context.current_idx);
        }
        
        // Get IMU data if VIO mode
        let imu_data = if estimator.config.enable_imu && context.processed_frames > 0 { // TODO when implementing IMU data loading
            Some(Self::get_imu_data_between_frames(
                context.previous_frame_timestamp,
                image_data[context.current_idx].timestamp,
            ))
        } else {
            None
        };

        // Process frame
        let imu_slice = imu_data.as_ref().map(|v| v.as_slice());
        estimator.process_frame(
            &left_image,
            &right_image,
            image_data[context.current_idx].timestamp,
            imu_slice,
        )?;

        // Update frame timestamp
        context.previous_frame_timestamp = image_data[context.current_idx].timestamp;

        let frame_duration = frame_start.elapsed();
        Ok(frame_duration.as_secs_f64() * 1000.0) // Return milliseconds
    }

    fn get_imu_data_between_frames(
        _previous_timestamp: i64,
        _current_timestamp: i64,
    ) -> Vec<ImuData> {
        // TODO: Implement IMU data retrieval between timestamps
        Vec::new()
    }

    fn save_trajectories(
        _estimator: &Estimator,
        _context: &FrameContext,
        _dataset_path: &str,
    ) {
        // TODO: Implement trajectory saving
        log::debug!("[4SeasonsPlayer] Saving trajectories (placeholder)");
    }

    fn save_statistics(result: &PlayerResult, dataset_path: &str) {
        let stats_file = Path::new(dataset_path).join(format!("statistics.txt"));

        if let Ok(mut file) = std::fs::File::create(&stats_file) {
            use std::io::Write;
            writeln!(file, "════════════════════════════════════════════════════════════════════").ok();
            writeln!(file, "                          STATISTICS                                ").ok();
            writeln!(file, "════════════════════════════════════════════════════════════════════").ok();
            writeln!(file, "").ok();

            // Timing statistics
            writeln!(file, "                          TIMING ANALYSIS                           ").ok();
            writeln!(file, "════════════════════════════════════════════════════════════════════").ok();
            writeln!(file, " Total Frames Processed: {}", result.processed_frames).ok();
            writeln!(file, " Average Processing Time: {:.2}ms", result.average_processing_time_ms).ok();
            let fps = 1000.0 / result.average_processing_time_ms;
            writeln!(file, " Average Frame Rate: {:.1}fps", fps).ok();
            writeln!(file, "════════════════════════════════════════════════════════════════════").ok();

            log::info!("[4SeasonsPlayer] Saved statistics to: {}", stats_file.display());
        }
    }
}
