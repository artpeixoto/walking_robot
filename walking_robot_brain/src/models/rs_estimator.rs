use std::iter::once;

use burn::{nn::{InstanceNorm, InstanceNormConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, Tanh}, prelude::*};
use itertools::Itertools;
use tracing::warn;

use crate::{modules::forward_module::ForwardModule,  tensor_conversion::TensorConvertible, tools::UsedInTrait, types::{action::GameAction, state::{GameState, Reward}}};

use super::builders::WINDOW_SIZE;

#[derive( Config, )]
pub struct RsEstimatorConfig{
    pub state_layers_size 		: [usize;1],
    pub action_layers_size		: [usize;1],
    pub joint_layers_size		: [usize;2],
    pub logic 					: [usize;1],
	pub cut_through				: [usize;1],
    pub end						: [usize;4],
}

impl RsEstimatorConfig{
	pub fn init<B:Backend>(self, dev: &<B as Backend>::Device) -> RsEstimator<B>{
		RsEstimator{
			action_linear_0: LinearConfig::new(GameAction::VALUES_COUNT * WINDOW_SIZE as usize, self.action_layers_size[0]).init(dev),
			action_act_0: LeakyReluConfig::new().init(),	

			state_linear_0: LinearConfig::new(GameState::VALUES_COUNT * WINDOW_SIZE as usize, self.state_layers_size[0]).init(dev),
			state_act_0: LeakyReluConfig::new().init(),


			joint_linear_0: LinearConfig::new(self.state_layers_size[0] + self.action_layers_size[0], self.joint_layers_size[0]).init(dev),
			joint_act_0: LeakyReluConfig::new().init(),
			joint_linear_1: LinearConfig::new(self.joint_layers_size[0], self.joint_layers_size[1]).init(dev),
			joint_act_1: LeakyReluConfig::new().init(),

			logic_linear_0: LinearConfig::new(self.joint_layers_size[1], self.logic[0]).init(dev),
			logic_act_0: Tanh,

			cut_through_linear_0: LinearConfig::new(self.joint_layers_size[1], self.cut_through[0]).init(dev),
			cut_through_act_0: LeakyReluConfig::new().init(),

			end_linear_0:LinearConfig::new(self.logic[0] + self.cut_through[0]  , self.end[0]).init(dev), 
			end_act_0: LeakyReluConfig::new().init(),
			end_linear_1:LinearConfig::new(self.end[0]  , self.end[1]).init(dev), 
			end_act_1: LeakyReluConfig::new().init(),
			end_linear_2:LinearConfig::new(self.end[1], self.end[2]).init(dev), 
			end_act_2: LeakyReluConfig::new().init(),
			end_linear_3:LinearConfig::new(self.end[2], self.end[3]).init(dev), 
			end_act_3: LeakyReluConfig::new().init(),
			end_linear_4:LinearConfig::new(self.end[3]  , GameState::VALUES_COUNT + 1).init(dev), 
		}
	}
}

#[derive(Debug, Module)]
pub struct RsEstimator<B: Backend>{
    state_linear_0	: Linear<B>,
    state_act_0		: LeakyRelu,

    action_linear_0	: Linear<B>,
    action_act_0	: LeakyRelu,

    joint_linear_0	: Linear<B>,
    joint_act_0		: LeakyRelu,
    joint_linear_1	: Linear<B>,
    joint_act_1		: LeakyRelu,

    logic_linear_0	: Linear<B>,
    logic_act_0		: Tanh,

    cut_through_linear_0: Linear<B>,
    cut_through_act_0: LeakyRelu,

    end_linear_0: Linear<B>,
    end_act_0: LeakyRelu, 
    end_linear_1: Linear<B>,
    end_act_1: LeakyRelu, 
    end_linear_2: Linear<B>,
    end_act_2: LeakyRelu, 
    end_linear_3: Linear<B>,
    end_act_3: LeakyRelu, 
    end_linear_4: Linear<B>,
}

impl<B: Backend> RsEstimator<B>{
	pub fn forward(&self, states_tensor: &Tensor<B, 2>, actions_tensor: &Tensor<B,2> ) -> (Tensor<B, 1>, Tensor<B, 2>) {
			let states_x = 
				states_tensor.clone()
				.used_in(|x| self.state_linear_0.forward(x))
				.used_in(|x| self.state_act_0.forward(x));

			let actions_x = 
				actions_tensor.clone()	
				// .used_in(|x| self.action_norm.forward(x))
				.used_in(|x| self.action_linear_0.forward(x))
				.used_in(|x| self.action_act_0.forward(x));

			let joint_x = Tensor::cat(vec![states_x, actions_x], 1);

			let joint_x= 			
				self.joint_linear_0.forward(joint_x)
				.used_in(|x| self.joint_act_0.forward(x))
				.used_in(|x| self.joint_linear_1.forward(x))
				.used_in(|x| self.joint_act_1.forward(x));

			let logic_x = 
				joint_x.clone()
				.used_in(|x| self.logic_linear_0.forward(x))
				.used_in(|x| self.logic_act_0.forward(x));

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
				.used_in(|x| self.end_act_2.forward(x))
				.used_in(|x| self.end_linear_3.forward(x))
				.used_in(|x| self.end_act_3.forward(x))
				.used_in(|x| self.end_linear_4.forward(x))
				
				;


			let r_output = output.clone().slice([None, Some((0, 1))]).squeeze(1);
			let s_output = output.clone().slice([None, Some((1, output.dims()[1] as i64))]);

			(r_output, s_output)
		// })
		// .unzip();
		// (Tensor::cat(r_output, 0), Tensor::cat(s_output, 0))
	}
}

impl<B: Backend> ForwardModule<B> for RsEstimator<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		let len = input.dims()[1];

		let states_tensor = input.clone().slice([None, Some((0_i64, GameState::VALUES_COUNT as i64 * WINDOW_SIZE))]);
		let actions_tensor = input.clone().slice([None, Some((GameState::VALUES_COUNT as i64 * WINDOW_SIZE , len as i64))]);
		
		let(rewards, next_states) = self.forward(&states_tensor, &actions_tensor);
		Tensor::cat(vec![rewards.unsqueeze_dim(1), next_states], 1)
	}
}
