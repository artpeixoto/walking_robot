use burn::{config::Config, module::Module, nn::{Gelu, InstanceNorm, InstanceNormConfig, Linear, LinearConfig, Tanh}, prelude::Backend, tensor::Tensor};
use itertools::Itertools;

use crate::{modules::forward_module::ForwardModule, tensor_conversion::TensorConvertible, tools::UsedInTrait, types::state::GameState};


#[derive( Config, )]
pub struct VEstimatorConfig{
    pub initial 			: [usize;2],

    pub logic 			: [usize;1],
	pub cut_through		: [usize;1],
    pub end				: [usize;1],
}

impl VEstimatorConfig{
	pub fn init<B:Backend>(self, dev: &<B as Backend>::Device) -> VEstimator<B>{
		VEstimator{

			linear_0: LinearConfig::new(GameState::VALUES_COUNT, self.initial[0]).init(dev),
			act_0: Gelu,
			linear_1: LinearConfig::new(self.initial[0], self.initial[1]).init(dev),
			act_1: Gelu,

			logic_linear_0: LinearConfig::new(self.initial[1], self.logic[0]).init(dev),
			logic_act_0: Tanh,

			cut_through_linear_0: LinearConfig::new(self.initial[1], self.cut_through[0]).init(dev),
			cut_through_act_0: Gelu,

			end_linear_0:LinearConfig::new(self.logic[0] + self.cut_through[0] , self.end[0]).init(dev), 
			end_act_0: Gelu,
			end_linear_1:LinearConfig::new(self.end[0], 1).init(dev), 
		}
	}
}

#[derive(Debug, Module)]
pub struct VEstimator<B: Backend>{
    linear_0: Linear<B>,
    act_0: Gelu,
    linear_1: Linear<B>,
    act_1: Gelu,

	logic_linear_0: Linear<B>,
    logic_act_0: Tanh,

    cut_through_linear_0: Linear<B>,
    cut_through_act_0: Gelu,

    end_linear_0: Linear<B>,
    end_act_0: Gelu, 
	end_linear_1: Linear<B>,
}

impl<B: Backend> VEstimator<B>{
	pub fn estimate(&self, state: &GameState, dev: &<B as Backend>::Device) -> f32{
		state.to_tensor::<B>(dev).unsqueeze_dim::<2>(0).into_data().as_slice::<f32>().unwrap()[0]
	}
	pub fn estimate_many<'a>(&self, states: impl Iterator<Item=&'a GameState>, dev: &<B as Backend>::Device) -> Vec<f32>{
		let states_tensor = states.map(|s| s.to_tensor::<B>(dev)).collect_vec().used_in(|s| Tensor::stack(s, 0));
		let values = self.forward(&states_tensor);
		values.into_data().to_vec().unwrap()
	}
	pub fn forward(&self, states_tensor: &Tensor<B, 2>,) -> Tensor<B, 1>{
		let states_x = 
		 	states_tensor.clone()
			.used_in(|x| self.linear_0.forward(x))
			.used_in(|x| self.act_0.forward(x))
			.used_in(|x| self.linear_1.forward(x))
			.used_in(|x| self.act_1.forward(x));


		let logic_x = 
		 	states_x.clone()
			.used_in(|x| self.logic_linear_0.forward(x))
			.used_in(|x| self.logic_act_0.forward(x));

		let cut_through_x = 
			states_x.clone()
			.used_in(|x| self.cut_through_linear_0.forward(x))
			.used_in(|x| self.cut_through_act_0.forward(x));


		let end_x = Tensor::cat(vec![logic_x, cut_through_x], 1);

		let output =  
			end_x
			.used_in(|x| self.end_linear_0.forward(x))
			.used_in(|x| self.end_act_0.forward(x))
			.used_in(|x| self.end_linear_1.forward(x))
			.squeeze(1);

		output
	}
}

impl<B: Backend> ForwardModule<B> for VEstimator<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(&input).unsqueeze_dim(1)
	}
}