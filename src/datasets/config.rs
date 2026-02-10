use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub camera: CameraConfig,
    #[serde(rename = "keyframe_management")]
    pub keyframe_management: KeyframeManagementConfig,
    #[serde(rename = "feature_detection")]
    pub feature_detection: FeatureDetectionConfig,
    pub optimization: OptimizationConfig,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    #[serde(rename = "image_width")]
    pub image_width: u32,
    #[serde(rename = "image_height")]
    pub image_height: u32,
    #[serde(rename = "left_intrinsics")]
    pub left_intrinsics: Vec<f64>,
    #[serde(rename = "left_distortion")]
    pub left_distortion: Vec<f64>,
    #[serde(rename = "right_intrinsics")]
    pub right_intrinsics: Vec<f64>,
    #[serde(rename = "right_distortion")]
    pub right_distortion: Vec<f64>,
    #[serde(rename = "left_model")]
    pub left_model: Option<String>,
    #[serde(rename = "right_model")]
    pub right_model: Option<String>,
    #[serde(rename = "T_B_Cl")]
    pub T_B_Cl: Vec<f64>,
    #[serde(rename = "T_B_Cr")]
    pub T_B_Cr: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframeManagementConfig {
    #[serde(rename = "keyframe_window_size")]
    pub keyframe_window_size: u32,
    #[serde(rename = "translation_threshold")]
    pub translation_threshold: f64,
    #[serde(rename = "rotation_threshold")]
    pub rotation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDetectionConfig {
    #[serde(rename = "grid_size")]
    pub grid_cols: u32,
    #[serde(rename = "max_features_per_grid")]
    pub max_features_per_grid: u32,
    #[serde(rename = "optical_flow_max_iterations")]
    pub optical_flow_max_iterations: u32,   
    #[serde(rename = "optical_flow_convergence_threshold")]
    pub optical_flow_convergence_threshold: f64,
  
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    #[serde(rename = "bundle_adjustment_max_iterations")]
    pub bundle_adjustment_max_iterations: u32,
    #[serde(rename = "pnp_max_iterations")]
    pub pnp_max_iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuConfig {
    #[serde(rename = "sensor_type")]
    pub sensor_type: Option<String>,
    #[serde(rename = "rate_hz")]
    pub rate_hz: Option<f64>,
    #[serde(rename = "T_BS")]
    pub T_BS: Vec<f64>,
    #[serde(rename = "accel_noise_density")]
    pub accel_noise_density: f64,
    #[serde(rename = "gyro_noise_density")]
    pub gyro_noise_density: f64,
    #[serde(rename = "accel_random_walk")]
    pub accel_random_walk: f64,
    #[serde(rename = "gyro_random_walk")]
    pub gyro_random_walk: f64,
}

impl ImuConfig {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        // Strip YAML directive if present (e.g., %YAML:1.0)
        let content = if content.trim_start().starts_with("%YAML") {
            content
                .lines()
                .skip_while(|line| line.trim_start().starts_with("%"))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            content
        };
        let config: ImuConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}


impl Config {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        // Strip YAML directive if present (e.g., %YAML:1.0)
        let content = if content.trim_start().starts_with("%YAML") {
            content
                .lines()
                .skip_while(|line| line.trim_start().starts_with("%"))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            content
        };
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

}

