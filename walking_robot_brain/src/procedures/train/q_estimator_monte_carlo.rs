use burn::{optim::Optimizer, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};

use crate::{loss::LossMod, models::{builders::WINDOW_SIZE, q_estimator::QEstimator}, tools::WindowsExt, types::history::{History, TensorHistory}};

use super::execute_training::{self, execute_training};

impl<B: AutodiffBackend> QEstimator<B>{
	pub fn train_monte_carlo(
		mut self,  
		history		: &TensorHistory<B>,
		alpha		: f32,
		lr			: f64,
		optim		: &mut impl Optimizer<Self, B>,
		loss_mod 	: &mut LossMod,
		dev  		: &<B as Backend>::Device, 
	) -> Self{
		let inputs =
			Tensor::cat(
				vec![
					history.states.clone().windows(WINDOW_SIZE),
					history.actions.clone().windows(WINDOW_SIZE)
				],
				1
			);

		let target_output = {
			let mut values = Vec::new();
			let mut value: Tensor<B, 2> = Tensor::zeros([1,1], dev);
			let mut tensor_rewards =  history.rewards.clone().iter_dim(0).collect::<Vec<_>>();
			tensor_rewards.reverse();
			for r in tensor_rewards{
				value = value.mul_scalar(alpha) + r;
				values.push(value.clone());
			}
			values.reverse();
			let count = values.len();
			Tensor::cat(values, 0).slice([(WINDOW_SIZE - 1, count as i64)])
		};

		execute_training(self, inputs, target_output, loss_mod, optim, lr)
	}
}