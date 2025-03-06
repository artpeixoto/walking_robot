use std::iter::once;

use burn::{nn::{Gelu, Linear, LinearConfig, Tanh}, prelude::*};
use itertools::Itertools;

use crate::{tensor_conversion::TensorConvertible, tools::UsedInTrait, types::{action::GameAction, state::{GameState, Reward}}};

#[derive( Config, )]
pub struct RsEstimatorConfig{
    pub state_layers_size 			: [usize;2],
    pub action_layers_size			: [usize;2],
    pub joint_layers_size				: [usize;3],
    pub logic 			: [usize;2],
	pub cut_through		: usize,
    pub end				: [usize;2],
}

impl RsEstimatorConfig{
	pub fn init<B:Backend>(self, dev: &<B as Backend>::Device) -> RsEstimator<B>{
		RsEstimator{
			action_linear_0: LinearConfig::new(GameAction::VALUES_COUNT, self.action_layers_size[0]).init(dev),
			action_act_0: Gelu,
			action_linear_1: LinearConfig::new(self.action_layers_size[0], self.action_layers_size[1]).init(dev),
			action_act_1: Gelu,

			state_linear_0: LinearConfig::new(GameState::VALUES_COUNT, self.state_layers_size[0]).init(dev),
			state_act_0: Gelu,
			state_linear_1: LinearConfig::new(self.state_layers_size[0], self.state_layers_size[1]).init(dev),
			state_act_1: Gelu,

			joint_linear_0: LinearConfig::new(self.state_layers_size[1] + self.action_layers_size[1], self.joint_layers_size[0]).init(dev),
			joint_act_0: Gelu,
			joint_linear_1: LinearConfig::new(self.joint_layers_size[0], self.joint_layers_size[1]).init(dev),
			joint_act_1: Gelu,
			joint_linear_2: LinearConfig::new(self.joint_layers_size[1], self.joint_layers_size[2]).init(dev),
			joint_act_2: Gelu,

			logic_linear_0: LinearConfig::new(self.joint_layers_size[2], self.logic[0]).init(dev),
			logic_act_0: Tanh,
			logic_linear_1: LinearConfig::new(self.logic[0], self.logic[1]).init(dev),
			logic_act_1: Tanh,

			cut_through_linear_0: LinearConfig::new(self.joint_layers_size[2], self.cut_through).init(dev),
			cut_through_act_0: Gelu,

			end_linear_0:LinearConfig::new(self.logic[1] + self.cut_through  , self.end[0]).init(dev), 
			end_act_0: Gelu,
			end_linear_1:LinearConfig::new(self.end[0], self.end[1]).init(dev), 
			end_act_1: Gelu,
			end_linear_2:LinearConfig::new(self.end[1], GameState::VALUES_COUNT + 1).init(dev), 
			end_act_2: Gelu,
		}
	}
}

#[derive(Debug, Module)]
pub struct RsEstimator<B: Backend>{
    state_linear_0: Linear<B>,
    state_act_0: Gelu,
    state_linear_1: Linear<B>,
    state_act_1: Gelu,

    action_linear_0: Linear<B>,
    action_act_0: Gelu,
    action_linear_1: Linear<B>,
    action_act_1: Gelu,

    joint_linear_0: Linear<B>,
    joint_act_0: Gelu,
    joint_linear_1: Linear<B>,
    joint_act_1: Gelu,
    joint_linear_2: Linear<B>,
    joint_act_2: Gelu,

    logic_linear_0: Linear<B>,
    logic_act_0: Tanh,
    logic_linear_1: Linear<B>,
    logic_act_1: Tanh,

    cut_through_linear_0: Linear<B>,
    cut_through_act_0: Gelu,

    end_linear_0: Linear<B>,
    end_act_0: Gelu, 
	end_linear_1: Linear<B>,
    end_act_1: Gelu,
	end_linear_2: Linear<B>,
    end_act_2: Gelu,
}

impl<B: Backend> RsEstimator<B>{
	pub fn estimate(&self, state: &GameState, action: &GameAction, dev: &<B as Backend>::Device ) -> (Reward, GameState){
		self.estimate_many(once((state, action)), dev).pop().unwrap()
	}
	pub fn estimate_many<'a>(&self, states_and_actions: impl Iterator<Item=(&'a GameState, &'a GameAction)>, dev: &<B as Backend>::Device) -> Vec<(Reward, GameState)>{
		let (states, actions) 
			: (Vec<_>, Vec<_>)
			= 	states_and_actions
				.map(|(state, action)| (
					state.to_tensor(dev).unsqueeze_dim(0),
					action.to_tensor(dev).unsqueeze_dim(0),
				))
				.unzip();
		let states_tensor 	= Tensor::cat(states, 0);
		let actions_tensor 	= Tensor::cat(actions, 0);
		let (rewards_tensor, next_states_tensor)  = self.forward(&states_tensor, &actions_tensor);

		rewards_tensor.iter_dim(0)
			.zip(next_states_tensor.iter_dim(0))
			.map(|(r, s)|{
				(r.into_data().into_vec().unwrap()[0], GameState::from_tensor(s.squeeze(0)))
			})
			.collect_vec()
	} 
	pub fn forward(&self, states_tensor: &Tensor<B, 2>, actions_tensor: &Tensor<B, 2> ) -> (Tensor<B, 1>, Tensor<B, 2>){
		let states_x = 
			self.state_linear_0.forward(states_tensor.clone())
			.used_in(|x| self.state_act_0.forward(x))
			.used_in(|x| self.state_linear_1.forward(x))
			.used_in(|x| self.state_act_1.forward(x));

		let actions_x = 
			self.action_linear_0.forward(actions_tensor.clone())
			.used_in(|x| self.action_act_0.forward(x))
			.used_in(|x| self.action_linear_1.forward(x))
			.used_in(|x| self.action_act_1.forward(x));

		let joint_x = Tensor::cat(vec![states_x, actions_x], 1);

		let joint_x = 			
			self.joint_linear_0.forward(joint_x)
			.used_in(|x| self.joint_act_0.forward(x))
			.used_in(|x| self.joint_linear_1.forward(x))
			.used_in(|x| self.joint_act_1.forward(x))
			.used_in(|x| self.joint_linear_2.forward(x))
			.used_in(|x| self.joint_act_2.forward(x));

		let logic_x = 
		 	joint_x.clone()
			.used_in(|x| self.logic_linear_0.forward(x))
			.used_in(|x| self.logic_act_0.forward(x))
			.used_in(|x| self.logic_linear_1.forward(x))
			.used_in(|x| self.logic_act_1.forward(x));

		let cut_through_x = 
			joint_x.clone()
			.used_in(|x| self.cut_through_linear_0.forward(x))
			.used_in(|x| self.cut_through_act_0.forward(x));

		let end_x = Tensor::cat(vec![logic_x, cut_through_x], 1);

		let output =  
			end_x
			.used_in(|x| self.end_linear_0.forward(x))
			.used_in(|x| self.end_act_0.forward(x))
			.used_in(|x| self.end_linear_1.forward(x))
			.used_in(|x| self.end_act_1.forward(x))
			.used_in(|x| self.end_linear_2.forward(x))
			.used_in(|x| self.end_act_2.forward(x));

		let r_output = output.clone().slice([None, Some((0, 1))]).squeeze(1);
		let s_output = output.clone().slice([None, Some((1, output.dims()[1] as i64))]);

		(r_output, s_output)
	}
}