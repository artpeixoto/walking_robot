use std::str::from_utf8;

use anyhow::{anyhow, bail};
use itertools::Itertools;
use json::JsonValue;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};
use tracing::debug;

use crate::{
    traits::{JsonExts, ToJson, TryFromJson},
    types::{
        action::{GameAction, LimbActivation},
        state::{
            AccelerometerReading, BipedalLimbsReading, Force, GameState, GameStateAndReward, GameUpdate, LimbReading, LinkReading, MotorReading, SensorsReading, TransformReading
        },
    },
};

pub struct SimulationConnector;
impl SimulationConnector {
    pub fn new() -> Self {
        Self
    }
    pub async fn connect(self) -> SimulationEndpoint {
        let stream = TcpListener::bind("127.0.0.1:8080")
            .await
            .unwrap()
            .accept()
            .await
            .unwrap()
            .0;
        
        SimulationEndpoint { stream }
    }
}
pub struct SimulationEndpoint {
    stream: TcpStream,
}

impl SimulationEndpoint {
    async fn send_msg(&mut self, bytes: &[u8]) {
        self.stream
            .write_all(&bytes.len().to_be_bytes())
            .await
            .unwrap();
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
    pub async fn send_action(&mut self, action: &GameAction) {
        let act_bytes = action.to_json().to_string().into_bytes();
        self.send_msg(&act_bytes).await;
    }

    pub async fn recv_sim_update(&mut self) -> GameUpdate {
        let msg = self.recv_msg().await;

        if &msg == b"GAME STARTED" {
            GameUpdate::GameStarted
        } else {
            let msg = from_utf8(&msg).unwrap();
            debug!("msg is: {msg}");
            let msg = json::parse(msg).unwrap();
            let GameStateAndReward { game_state, reward } = msg.try_as().unwrap();
            GameUpdate::GameStep {
                state: game_state,
                reward,
            }
        }
    }
}

impl ToJson for GameAction {
    fn to_json(&self) -> JsonValue {
        json::object! {
            LimbsActivation:{
                Left: (self.limbs_activation.left.to_json()),
                Right: (self.limbs_activation.right.to_json()),
            }
        }
    }
}
impl ToJson for LimbActivation {
    fn to_json(&self) -> JsonValue {
        json::object! {
            Shoulder: (self.shoulder_activation),
            Thigh	: (self.thigh_activation),
            Shin: (self.shin_activation),
        }
    }
}

impl TryFromJson for GameStateAndReward {
    fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error> {
        let json = json.as_object().unwrap();
        Ok(GameStateAndReward { 
            game_state  : json["State"].try_as().unwrap(), 
            reward      : json["Reward"].as_f32().unwrap()
        })
    }
}
impl TryFromJson for GameState{
    fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {
        let json = json.as_object().unwrap();
        Ok(Self{
            limbs_readings: json["LimbsReading"].try_as::<BipedalLimbsReading>().unwrap(),
            sensors_reading:json["SensorsReading"].try_as().unwrap(),
        })
    }
}
impl TryFromJson for BipedalLimbsReading{
    fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {
        let json = json.as_object().unwrap();
        Ok(Self { 
            left    : json["Left"].try_as().unwrap(), 
            right   : json["Right"].try_as().unwrap(),
        })
    }
}

impl TryFromJson for SensorsReading{
    fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {
        let json = json.as_object().unwrap();
        Ok(Self { 
            target_pos      : json["TargetPos"].as_vector3().unwrap(), 
            floor_distance  : json["FloorDist"].as_f32().unwrap(),
            acc_reading     : json["AccelerometerReading"].try_as().unwrap(),
            forces          : {
                let JsonValue::Array(arr) = &json["Forces"] else {panic!()};

                arr.iter()
                    .map(|el| el.try_as::<Force>().unwrap())
                    .collect_vec()
            }
        }) 
    }
}

impl TryFromJson for LimbReading{
	fn try_from_json(json: &json::JsonValue) -> Result<Self, anyhow::Error>  {
		let json = json.as_object().unwrap();
		Ok(LimbReading{
			shoulder: json["ShoulderReading"]	.try_as().unwrap(),
			thigh	: json["ThighReading"]		.try_as().unwrap(),
			shin	: json["ShinReading"]		.try_as().unwrap(),
			foot	: json["FootReading"]		.try_as().unwrap(),
		})
	}
}
impl TryFromJson for Force{
    fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {
        let json = json.as_object().unwrap();
        Ok(Self{
            pos  : json["Position"].as_vector3().unwrap(),
            force: json["Force"].as_vector3().unwrap(),
        })
    }
}

impl TryFromJson for AccelerometerReading{
    fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error>  {
        let json = json.as_object().unwrap();
        Ok(AccelerometerReading{
            linear_speed    : json["LinearSpeed"].as_vector3().unwrap(),
            linear_acc      : json["LinearAcc"].as_vector3().unwrap(), 
            angular_speed   : json["AngularSpeed"].as_vector3().unwrap(),
            angular_acc     : json["AngularAcc"].as_vector3().unwrap(),
            up              : json["UpOrientation"].as_vector3().unwrap(), 
        })
    }
}

impl TryFromJson for TransformReading {
    fn try_from_json(json: &json::JsonValue) -> Result<Self, anyhow::Error> {
        let json = json.as_object().unwrap();
        Ok(Self {
            linear_pos		: json["LinearPos"].as_vector3().unwrap(),
            linear_speed	: json["LinearSpeed"].as_vector3().unwrap(),
            linear_acc		: json["LinearAcc"].as_vector3().unwrap(),

            angular_pos		: json["AngularPos"].as_quaternion().unwrap(), 
            angular_speed	: json["AngularSpeed"].as_vector3().unwrap(),
            angular_acc		: json["AngularAcc"].as_vector3().unwrap(),
        })
    }
}

impl TryFromJson for MotorReading {
    fn try_from_json(json: &json::JsonValue) -> Result<Self, anyhow::Error> {
        let json = json.as_object().unwrap();
        Ok(Self {
            acc: json["Acc"].as_f32().unwrap(),
            pos: json["Pos"].as_f32().unwrap(),
            speed: json["Speed"].as_f32().unwrap(),
            torque: json["Torque"].as_f32().unwrap(),
        })
    }
}

impl TryFromJson for LinkReading{
	fn try_from_json(json: &json::JsonValue) -> Result<Self, anyhow::Error>  {
		let json = json.as_object().unwrap();
		Ok(LinkReading { 
			motor		: json["MotorReading"].try_as::<MotorReading>().unwrap(), 
			transform	:  json["TransformReading"].try_as().unwrap(),
		})
	}
}
