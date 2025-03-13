use burn::{nn::loss::{HuberLoss}, optim::Optimizer, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use tracing::info;

use crate::{loss::LossMod, models::{builders::WINDOW_SIZE, rs_estimator::RsEstimator}, procedures::train::execute_training::execute_training, tensor_conversion::TensorConvertibleIterExts, tools::WindowsExt, types::history::History};

impl<B: AutodiffBackend> RsEstimator<B>{
	pub fn train(
		mut self,  
		states 		: &Tensor<B,2>,
		actions		: &Tensor<B,2>,
		rewards		: &Tensor<B,2>,
		lr			: f64,
		optim		: &mut impl Optimizer<RsEstimator<B>, B>,
		loss_mod 	: &mut LossMod,
		dev  		: &<B as Backend>::Device, 
	) -> Self {
        info!("training rs_estimator ");
		let count =  states.dims()[0] as i64;
		let stacked_states = states.clone().slice([Some((0, count-1)),None]).windows(WINDOW_SIZE);
		let stacked_actions = actions.clone().slice([Some((0, count-1)),None]).windows(WINDOW_SIZE);

		let input_tensor = Tensor::cat(vec![stacked_states, stacked_actions], 1);

		let output_states = states.clone().slice([Some((WINDOW_SIZE , count)), None]);
		let rewards_tensor = rewards.clone().slice([Some((WINDOW_SIZE - 1, count-1)), None]);

		let target_output_tensor = Tensor::cat(vec![rewards_tensor, output_states], 1);

		self = execute_training(self, input_tensor, target_output_tensor, loss_mod, optim, lr);

		self
	}
}
