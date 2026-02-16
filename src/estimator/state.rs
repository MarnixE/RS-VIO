use nalgebra::Matrix3;
use rerun::external::glam::Vec3;

use crate::types::{Matrix4x4, Vector3, Matrix3x3};


#[derive(Debug, Clone)]
pub struct State {
    /// World-from-body pose as a 4x4 row-major matrix (T_w_b).
    pub T_W_B: Matrix4x4,

    pub T_B_Cl: Matrix4x4,
    pub T_B_Cr: Matrix4x4,

    /// Body-frame linear velocity in world coordinates.
    pub velocity: Vector3,

    /// Accelerometer bias.
    pub accel_bias: Vector3,

    /// Gyroscope bias.
    pub gyro_bias: Vector3,
}

impl State {

    pub fn new(T_B_Cl: Matrix4x4, T_B_Cr: Matrix4x4, 
        velocity: Option<Vector3>, accel_bias: Option<Vector3>, gyro_bias: Option<Vector3>) -> Self {
        Self {
            T_W_B: Matrix4x4::identity(),
            T_B_Cl: T_B_Cl,
            T_B_Cr: T_B_Cr,
            velocity: velocity.unwrap_or(Vector3::zeros()),
            accel_bias: accel_bias.unwrap_or(Vector3::zeros()),
            gyro_bias: gyro_bias.unwrap_or(Vector3::zeros()),
        }
    }

    /// Identity pose, zero velocity and zero biases.
    pub fn identity() -> Self {
        Self {
            T_W_B: Matrix4x4::identity(),
            T_B_Cl: Matrix4x4::identity(),
            T_B_Cr: Matrix4x4::identity(),
            velocity: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
        }
    }

}


