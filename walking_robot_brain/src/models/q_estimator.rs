use std::iter;

use burn::{module::Module, nn::{LeakyRelu, LeakyReluConfig, Linear, LinearConfig, Tanh}, prelude::Backend, tensor::Tensor};
use itertools::Itertools;
use rand::rng;

use crate::{modules::{forward_module::ForwardModule, sequential::{LinearSequential, LinearSequentialConfig}}, tensor_conversion::TensorConvertible, types::{action::GameAction, policy::{nil_policy::NilPolicy, noisy_policy::NoisyPolicy, TensorPolicy}, state::GameState}};

use super::builders::WINDOW_SIZE;
pub struct QEstimatorConfig{
	pub initial		: Vec<usize>,
	pub logic		: Vec<usize>,
	pub cut_through : Vec<usize>,
	pub joint		: Vec<usize>,
}
impl QEstimatorConfig{
	pub fn init<B: Backend>(&self, dev: &<B as Backend>::Device) -> QEstimator<B>{
		let input_size = WINDOW_SIZE as usize * (GameState::VALUES_COUNT + GameAction::VALUES_COUNT);

		let initial = iter::once(input_size).chain(self.initial.iter().cloned()).collect_vec();
		let logic = iter::once(initial.last().unwrap().clone()).chain(self.logic.iter().cloned()).collect_vec();
		let cut_through = iter::once(initial.last().unwrap().clone()).chain(self.cut_through.iter().cloned()).collect_vec();
		let joint = 
			iter::once(cut_through.last().unwrap().clone() + logic.last().unwrap().clone())
			.chain(self.cut_through.iter().cloned()).collect_vec(); 

		let joint_to_output = joint.last().unwrap().clone();

		QEstimator{
			initial		: 
				LinearSequentialConfig{
					sizes:  initial,
					act: LeakyReluConfig::new().init()
				}
				.init(dev),
			logic		: 
				LinearSequentialConfig{
					sizes:  logic,
					act: Tanh,
				}
				.init(dev),

			cut_through	: 
				LinearSequentialConfig{
					sizes:  cut_through,
					act: LeakyReluConfig::new().init(),
				}
				.init(dev),
	
			joint		:
				LinearSequentialConfig{
					sizes:  joint,
					act: LeakyReluConfig::new().init(),
				}
				.init(dev),
	 
			output: LinearConfig::new(joint_to_output, 1).init(dev)
		}
	}
}
#[derive(
	Module, Debug
)]
pub struct QEstimator<B: Backend>{
	initial		: LinearSequential<B, LeakyRelu>,
	logic		: LinearSequential<B, Tanh>,
	cut_through : LinearSequential<B, LeakyRelu>,
	joint		: LinearSequential<B, LeakyRelu>,
	output		: Linear<B>
}

impl<B: Backend> ForwardModule<B> for QEstimator<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		let x = self.initial.forward(input);
		let logic  = self.logic.forward(x.clone());
		let cut_through = self.cut_through.forward(x);
		
		let x = Tensor::cat(vec![logic, cut_through], 1);
		let x = self.joint.forward(x);
		let x = self.output.forward(x); 

		x
	}
}
