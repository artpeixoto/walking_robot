use std::iter;

use burn::{nn::loss::{HuberLoss, }, optim::Optimizer, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use tracing::info;

use crate::{models::v_estimator::VEstimator, procedures::train::execute_training::execute_training, tensor_conversion::TensorConvertible, types::history::History};

impl<B: AutodiffBackend> VEstimator<B>{

	pub fn monte_carlo_train(
		mut self,  
		history: &History,
		alpha: f32,
		lr: f64,
		optim: &mut impl Optimizer<VEstimator<B>, B>,
		loss_mod : &mut HuberLoss,
		dev  : &<B as Backend>::Device, 
	) -> Self {
        info!("training v_estimator with monte carlo");
		let count = history.states.len();
		let mut states_input_tensors: Vec<Tensor<B, 2>>  = Vec::new();
		let mut target_outputs: Vec<f32 > = Vec::new();
		let mut accumulated_g = 0.0f32;	

		for (state,  reward)  in iter::zip(history.states.iter(), history.rewards.iter()).rev(){
			states_input_tensors.push(state.to_tensor(dev).unsqueeze_dim(0));
			accumulated_g =  reward + accumulated_g * alpha ;
			target_outputs.push(accumulated_g);			
		}

		target_outputs.reverse();
		// let target_outputs_tensor = Tensor::<B, 1>::from_floats(target_outputs.as_slice(), dev).unsqueeze_dim(1);
		// let states_input_tensor = Tensor::cat(states_input_tensors, 0);

		// self = execute_training(self, states_input_tensor, target_outputs_tensor, loss_mod, optim, lr);

		self
	}
}
