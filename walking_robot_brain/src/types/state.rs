use anyhow::{anyhow, bail};
use json::{object::Object, JsonValue};
use nalgebra::{Vector2, Vector3};

use crate::traits::{JsonExts, TryFromJson};

pub type G = f32;
pub type Reward = f32;
pub struct GameStateAndReward{
	pub game_state	: GameState,
	pub reward		: f32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct GameState{
	pub sensors_reading: SensorsReading,
	pub limbs_readings : BipedalLimbsReading
}

pub enum GameUpdate{
	GameStarted,
	GameStep{
		state: GameState,
		reward: f32,
	}
}

#[derive(Clone, PartialEq, Debug)]
pub struct SensorsReading{
	pub target_direction: Vector2<f32>,
	pub up_orientation	: Vector3<f32>,
	pub target_distance: f32,
	pub floor_distance : f32,
	pub speed			: Vector3<f32>, 
	pub linear_acceleration: Vector3<f32>,
	pub angular_acceleration: Vector3<f32>,
}


#[derive(Clone, PartialEq, Debug)]
pub struct BipedalLimbsReading{
	pub left: LimbReading,
	pub right: LimbReading,	
}

#[derive(Clone, PartialEq, Debug)]
pub struct LimbJointsForces{
	pub shoulder: f32,
	pub thigh: f32,
	pub shin: f32,
}
#[derive(Clone, PartialEq, Debug)]
pub struct LimbJointsPositions{
	pub shoulder: f32,
	pub thigh: f32,
	pub shin: f32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct LimbReading{
	pub forces: LimbJointsForces,
	pub positions: LimbJointsPositions,
	pub is_foot_touching_floor: bool,
	pub force_applied_by_floor: Vector3<f32>
}
