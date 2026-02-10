use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};
use image::{ImageReader};
use crate::imu;
use crate::viewers::{Viewer, create_viewer};
use crate::estimator::Estimator;
use crate::datasets::{config::Config, ImageData, ImuData, FrameContext, PlayerConfig, PlayerResult, config::ImuConfig};


pub struct EurocPlayer;

impl EurocPlayer {
    pub fn new() -> Self {
        EurocPlayer
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

        let (imu_data, imu_config) = match Self::load_imu_data(
            &config.dataset_path,
            &image_data,
            0, // start_frame_idx
            image_data.len() - 1, // end_frame_idx
        ) {
            Ok(data) => data,
            Err(e) => {
                result.error_message = format!("Failed to load IMU data: {}", e);
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
            Estimator::new_with_cameras(cfg, viewer_ref, Some(left_cam), Some(right_cam))
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
                    thread::sleep(Duration::from_millis(30));
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
                    &imu_data,
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
                "[EurocPlayer] Average processing time: {:.2} ms ({:.1} fps)",
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

        log::info!("[EurocPlayer] Processing completed! Viewer remains open for inspection.");
        
        result
    }

    fn load_image_timestamps(dataset_path: &str) -> Result<Vec<ImageData>> {
        let data_file = Path::new(dataset_path).join("mav0/cam0/data.csv");
        let file = File::open(&data_file)
            .with_context(|| format!("Cannot open data.csv file: {}", data_file.display()))?;

        let reader = BufReader::new(file);
        let mut image_data = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            
            // Skip header and empty lines
            if line_num == 0 || line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let timestamp_str = parts[0].trim();
                let filename = parts[1].trim().to_string();

                if let Ok(timestamp) = timestamp_str.parse::<i64>() {
                    image_data.push(ImageData {
                        timestamp,
                        filename,
                    });
                }
            }
        }

        log::info!("[EurocPlayer] Loaded {} image timestamps", image_data.len());
        Ok(image_data)
    }

    fn load_image(dataset_path: &str, filename: &str, cam_id: u32) -> Result<Vec<u8>> {
        let cam_folder = if cam_id == 0 { "cam0" } else { "cam1" };
        let full_path = Path::new(dataset_path)
            .join("mav0")
            .join(cam_folder)
            .join("data")
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
        dataset_path: &str,
        image_data: &[ImageData],
        start_frame_idx: usize,
        end_frame_idx: usize,
    ) -> Result<(Vec<ImuData>, ImuConfig), std::io::Error> {
        // Validate frame indices
        if image_data.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "image_data is empty"));
        }
        if start_frame_idx > end_frame_idx {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Invalid frame range: start_frame_idx ({}) > end_frame_idx ({})",
                    start_frame_idx, end_frame_idx
                ),
            ));
        }
        if end_frame_idx >= image_data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "end_frame_idx ({}) exceeds image_data length ({})",
                    end_frame_idx, image_data.len()
                ),
            ));
        }

        // Extract temporal bounds from image timestamps
        let start_timestamp = image_data[start_frame_idx].timestamp;
        let end_timestamp = image_data[end_frame_idx].timestamp;

        // Construct path to IMU CSV file
        let imu_csv_path = Path::new(dataset_path).join("mav0").join("imu0").join("data.csv");
        let sensor_yaml_path = Path::new(dataset_path).join("mav0").join("imu0").join("sensor.yaml");

        let imu_config = match ImuConfig::load(sensor_yaml_path.to_str().unwrap()) {
            Ok(cfg) => cfg,
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to load IMU sensor config: {}", e),
                ))?;
            }
        };
        log::info!(
            "[EurocPlayer] Loading IMU data from {:?} with temporal filter [{}, {}]",
            imu_csv_path,
            start_timestamp,
            end_timestamp
        );

        // Open file with descriptive error
        let file = File::open(&imu_csv_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to open IMU CSV at {:?}: {}", imu_csv_path, e)))?;
        
        let reader = BufReader::new(file);
        
        // Pre-allocate with estimated capacity (200 Hz IMU × duration in seconds)
        // let duration_ns = (end_timestamp - start_timestamp).max(0) as u64;
        // let duration_s = duration_ns as f64 / 1e9;
        // let estimated_count = (duration_s * 200.0).ceil() as usize;
        // let mut imu_measurements = Vec::with_capacity(estimated_count);
        let mut imu_measurements = Vec::new(); // Start with empty and let it grow as needed
        
        // Stream and parse CSV line-by-line
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            
            // Skip header and empty lines
            if line_num == 0 || line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 7 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Line {} has {} columns, expected 7 (timestamp,gx,gy,gz,ax,ay,az)",
                        line_num + 1, parts.len()
                    )
                ));
            }
            
            // Parse timestamp first for early filtering
            let timestamp_str = parts[0].trim();
            if let Ok(timestamp) = timestamp_str.parse::<i64>() {
                // Apply temporal filter: [start_timestamp, end_timestamp]
                if timestamp < start_timestamp {
                    continue;  // Skip measurements before window
                }
                if timestamp > end_timestamp {
                    break;  // EuRoC CSV is sorted; no more relevant data
                }
                
                // Parse gyroscope and accelerometer data
                if let (Ok(gyro_x), Ok(gyro_y), Ok(gyro_z),
                        Ok(accel_x), Ok(accel_y), Ok(accel_z)) = (
                    parts[1].trim().parse::<f64>(),
                    parts[2].trim().parse::<f64>(),
                    parts[3].trim().parse::<f64>(),
                    parts[4].trim().parse::<f64>(),
                    parts[5].trim().parse::<f64>(),
                    parts[6].trim().parse::<f64>(),
                ) {
                    imu_measurements.push(ImuData {
                        timestamp,
                        gyro: [gyro_x, gyro_y, gyro_z],
                        accel: [accel_x, accel_y, accel_z],
                    });
                }
            }
        }
        
        // Return sorted vector (already sorted due to CSV ordering and sequential read)
        Ok((imu_measurements, imu_config))
        // // TODO: Implement IMU data loading
        // log::info!("[EurocPlayer] IMU data loading (placeholder)");
        // Ok(())
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
        log::debug!("[EurocPlayer] Estimator initialized");
    }

    fn process_single_frame<'a>(
        estimator: &mut Estimator<'a>,
        context: &mut FrameContext,
        image_data: &[ImageData],
        imu_data: &[ImuData],
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
        let imu_result = if context.processed_frames > 0 { // TODO when implementing IMU data loading
            Some(Self::get_imu_data_between_frames(
                imu_data,
                context.previous_frame_timestamp,
                image_data[context.current_idx].timestamp,
            ))
        } else {
            None
        };

        // Process frame
        let imu_slice = imu_result.as_ref().map(|v| v.as_slice());
        print!("IMU slice for frame {}: ", context.current_idx);
        if let Some(slice) = imu_slice {
            println!("{:?}", slice);
        } else {
            println!("None");
        }
        estimator.process_frame(
            &left_image,
            &right_image,
            image_data[context.current_idx].timestamp,
            imu_slice,
        )?;

        println!("Processed IMU measurements: {}", imu_slice.map_or(0, |v| v.len()));

        // Update frame timestamp
        context.previous_frame_timestamp = image_data[context.current_idx].timestamp;

        let frame_duration = frame_start.elapsed();
        Ok(frame_duration.as_secs_f64() * 1000.0) // Return milliseconds
    }

    fn get_imu_data_between_frames(
        imu_data: &[ImuData],
        previous_timestamp: i64,
        current_timestamp: i64,
    ) -> Vec<ImuData> {
        // Handle edge cases
        if imu_data.is_empty() || previous_timestamp >= current_timestamp {
            return Vec::new();
        }
        
        // Binary search for first measurement with timestamp > previous_timestamp
        let start_idx = imu_data.partition_point(|imu| imu.timestamp <= previous_timestamp);
        
        // If start_idx is beyond the array, no measurements exist after previous_timestamp
        if start_idx >= imu_data.len() {
            return Vec::new();
        }
        
        // Find first measurement with timestamp > current_timestamp
        let end_idx = imu_data[start_idx..].partition_point(|imu| imu.timestamp <= current_timestamp);
        let end_idx = start_idx + end_idx;
        
        // Clone the subset of measurements in the half-open interval (previous, current]
        // Note: We clone because the caller needs owned data for IMU pre-integration
        imu_data[start_idx..end_idx].iter().map(|imu| ImuData {
            timestamp: imu.timestamp,
            gyro: imu.gyro,
            accel: imu.accel,
        }).collect()
    }

    fn save_trajectories(
        _estimator: &Estimator,
        _context: &FrameContext,
        _dataset_path: &str,
    ) {
        // TODO: Implement trajectory saving
        log::debug!("[EurocPlayer] Saving trajectories (placeholder)");
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

            log::info!("[EurocPlayer] Saved statistics to: {}", stats_file.display());
        }
    }
}
