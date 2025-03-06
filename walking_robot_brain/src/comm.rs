use std::{str::from_utf8};

use anyhow::{anyhow, bail};
use json::JsonValue;
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::{TcpListener, TcpStream}};
use tracing::{debug, warn};

use crate::{traits::{JsonExts, ToJson, TryFromJson}, types::{action::{GameAction, LimbActivation}, state::{BipedalLimbsReading, GameState, GameStateAndReward, GameUpdate, LimbJointsForces, LimbJointsPositions, LimbReading, SensorsReading}}};

pub struct SimulationConnector;
impl SimulationConnector{
	pub fn new() -> Self{
		Self
	}
	pub async fn connect(self) -> SimulationEndpoint {
		let stream = TcpListener::bind("127.0.0.1:8080").await.unwrap().accept().await.unwrap().0;
		SimulationEndpoint{stream}
	}
}
pub struct SimulationEndpoint{
	stream: TcpStream 
}

impl SimulationEndpoint{
	async fn send_msg(&mut self, bytes: &[u8]){
		self.stream.write_all(&bytes.len().to_be_bytes()).await.unwrap();
		self.stream.write_all(bytes).await.unwrap();
		self.stream.flush().await.unwrap();
	}
	async fn recv_msg(&mut self) -> Vec<u8> {
		let mut len_buf = [0_u8; size_of::<usize>()];
		self.stream.read_exact(&mut len_buf).await.unwrap();
		let len = usize::from_be_bytes(len_buf);
		debug!("message is {len} bytes long");

		let mut msg_buf = vec![0_u8; len];
		self.stream.read_exact(&mut msg_buf).await.unwrap();
		debug!("received: \"{}\"", from_utf8(&msg_buf).unwrap());
		msg_buf
	}
	pub async fn send_action(&mut self, action: &GameAction){
		let act_bytes = action.to_json().to_string().into_bytes();
		self.send_msg(&act_bytes).await;
	}

	pub async fn recv_sim_update(&mut self) -> GameUpdate{
		let msg = self.recv_msg().await;

		if &msg == b"GAME STARTED" {
			GameUpdate::GameStarted
		} else {
			let msg = json::parse(from_utf8(&msg).unwrap()).unwrap();
			let GameStateAndReward {game_state, reward} = msg.try_as().unwrap();
			GameUpdate::GameStep { state: game_state ,reward  }
		}
	}


}

impl ToJson for GameAction{
	fn to_json(&self) -> JsonValue {
		json::object!{
			LimbsActivation:{
				Left: (self.limbs_activation.left.to_json()),
				Right: (self.limbs_activation.right.to_json()),
			}
		}
	}
}
impl ToJson for LimbActivation{
	fn to_json(&self) -> JsonValue {
		json::object!{
			Shoulder: (self.shoulder_activation),
			Thigh	: (self.thigh_activation),
			Shin: (self.shin_activation),
		}
	}
}
impl TryFromJson for LimbReading{
		fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {
			let json = json.as_object().ok_or(anyhow!("aint no object"))?;
			let is_foot_touching_floor = json.get("IsFootTouchingFloor").unwrap().as_bool().unwrap();
			let force_applied_by_floor = json["ForceAppliedByFloor"].as_vector3().unwrap();
			let positions = {
				let json = json.get("JointPositions").unwrap().as_object().unwrap();

				let shoulder_pos = json["Shoulder"].as_f32().ok_or(anyhow!("no shoulder pos reading"))?;
				let thigh_pos = json["Thigh"].as_f32().ok_or(anyhow!("no thigh pos reading"))?;
				let shin_pos = json["Shin"].as_f32().ok_or(anyhow!("no shin pos reading"))?;
				LimbJointsPositions{
					shoulder: shoulder_pos,
					thigh: thigh_pos,
					shin: shin_pos
				}
			};
			let forces = {
				let json = json.get("JointForces").unwrap().as_object().unwrap();

				let shoulder_pos = json["Shoulder"].as_f32().ok_or(anyhow!("no shoulder pos reading"))?;
				let thigh_pos = json["Thigh"].as_f32().ok_or(anyhow!("no thigh pos reading"))?;
				let shin_pos = json["Shin"].as_f32().ok_or(anyhow!("no shin pos reading"))?;
				LimbJointsForces{
					shoulder: shoulder_pos,
					thigh: thigh_pos,
					shin: shin_pos
				}
			};
			Ok(LimbReading{
				forces,
				positions,
				is_foot_touching_floor,
				force_applied_by_floor
			})

		}
	}
impl TryFromJson for GameStateAndReward{
		fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {

			let JsonValue::Object(value) = json else {bail!("fuck")};
			let reward = value.get("Reward").ok_or(anyhow!("fuck"))?.as_f32().unwrap();
			let game_state = {
				let obj = value.get("State").unwrap().as_object().unwrap() ;
				let is_finished = obj.get("IsFinished").unwrap().as_bool().unwrap();
				let sensors_reading = {
					let obj = obj.get("HeadSensorsReading").unwrap().as_object().unwrap();

					let target_direction = obj.get("LocalTargetDir").unwrap().as_vector2().unwrap();
					let dist_from_target = obj.get("TargetDist").unwrap().as_f32().unwrap();
					let floor_distance= obj.get("FloorDist").unwrap().as_f32().unwrap();
					let up_orientation = obj["UpOrientation"].as_vector3().unwrap();
					let speed = obj["LocalSpeed"].as_vector3().unwrap();
					let linear_acceleration = obj["LocalLinearAcceleration"].as_vector3().unwrap();
					let angular_acceleration = obj["LocalAngularAcceleration"].as_vector3().unwrap();
					SensorsReading{
						floor_distance,
						target_distance: dist_from_target,
						speed,
						target_direction,
						up_orientation,
						linear_acceleration,
						angular_acceleration
					}
				};
				let limbs_readings = {
					let obj = obj["LimbsReading"].as_object().unwrap();
					let left_limb_reading = obj["Left"].try_as::<LimbReading>().unwrap();
					let right_limb_reading = obj["Right"].try_as::<LimbReading>().unwrap();
					BipedalLimbsReading{
						left: left_limb_reading,
						right: right_limb_reading,
					}
				};
				GameState{
					sensors_reading,
					limbs_readings
				}
			};
			Ok(GameStateAndReward{
				game_state,
				reward
			})
		}
	}