use burn::{prelude::Backend, tensor::Tensor};

use crate::tensor_conversion::TensorConvertibleIterExts;

use super::{action::GameAction, state::{GameState, Reward}};

#[derive(Debug, Default)]
pub struct History {
	pub states : Vec<GameState>	,
	pub actions: Vec<GameAction>, 
	pub rewards: Vec<Reward>
}

impl History{
	pub fn to_tensor_history<B: Backend>(&self, dev: &<B as Backend>::Device ) -> TensorHistory<B>{
		TensorHistory{
			actions	: self.actions.iter().many_to_tensor(dev),
			states	: self.states.iter().many_to_tensor(dev),
			rewards	: self.rewards.iter().many_to_tensor(dev)
		}
	}
}

pub struct TensorHistory<B: Backend>{
	pub states 			: Tensor<B, 2>,
	pub actions			: Tensor<B, 2>,
	pub rewards			: Tensor<B, 2>
}

pub type HistoryStep 	= (GameState		, GameAction	 , Reward);