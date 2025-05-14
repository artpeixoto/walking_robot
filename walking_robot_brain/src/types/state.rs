use nalgebra::{Quaternion, Vector, Vector2, Vector3};

use crate::traits::{JsonExts, TryFromJson};

pub type G = f32;
pub type Reward = f32;
pub struct GameStateAndReward {
    pub game_state: GameState,
    pub reward: f32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct GameState {
    pub sensors_reading: SensorsReading,
    pub limbs_readings: BipedalLimbsReading,
}

pub enum GameUpdate {
    GameStarted,
    GameStep { state: GameState, reward: f32 },
}

#[derive(Clone, PartialEq, Debug)]
pub struct SensorsReading {
    pub target_pos      : Vector3<f32>,
    pub floor_distance  : f32,
    pub acc_reading     : AccelerometerReading,
    pub forces          : Vec<Force>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Force{
    pub pos     : Vector3<f32>,
    pub force   : Vector3<f32>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct BipedalLimbsReading {
    pub left: LimbReading,
    pub right: LimbReading,
}
#[derive(Clone, PartialEq, Debug)]
pub struct LimbReading {
	pub shoulder: LinkReading,
	pub thigh	: LinkReading,
	pub shin 	: LinkReading,
	pub foot	: TransformReading,
}
#[derive(Clone, PartialEq, Debug)]
pub struct LinkReading {
    pub motor		: MotorReading,
    pub transform	: TransformReading,
}


#[derive(Clone, PartialEq, Debug)]
pub struct TransformReading {
    pub linear_pos: Vector3<f32>,
    pub linear_speed: Vector3<f32>,
    pub linear_acc: Vector3<f32>,

    pub angular_pos: Quaternion<f32>,
    pub angular_speed: Vector3<f32>,
    pub angular_acc: Vector3<f32>,
}



#[derive(Clone, PartialEq, Debug)]
pub struct MotorReading {
    pub pos: f32,
    pub speed: f32,
    pub acc: f32,
    pub torque: f32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct AccelerometerReading{
	pub up				: Vector3<f32>,
	pub linear_speed	: Vector3<f32>,
	pub linear_acc		: Vector3<f32>,
	pub angular_speed	: Vector3<f32>,
	pub angular_acc		: Vector3<f32>,
}