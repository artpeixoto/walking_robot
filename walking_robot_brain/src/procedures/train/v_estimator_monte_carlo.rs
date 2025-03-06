use burn::{module::Module, nn::loss::MseLoss, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}, train::logger::LoggerBackend};
use itertools::Itertools;

use crate::{models::{rs_estimator::RsEstimator, v_estimator::VEstimator}, tensor_conversion::TensorConvertible, types::{action::GameAction, history::{History, HistoryStep}, state::{GameState, Reward}}};



impl<B: AutodiffBackend> VEstimator<B>{

	pub fn monte_carlo_train(
		self,  
		history: &History,
		alpha: f32,
		lr: f64,
		optim: &mut impl Optimizer<VEstimator<B>, B>,
		loss : &mut MseLoss,
		dev  : &<B as Backend>::Device, 
	) -> Self {
		let mut states_input_tensors: Vec<Tensor<B, 2>>  = Vec::new();
		let mut target_outputs: Vec<f32 > = Vec::new();
		let mut accumulated_g = 0.0f32;	

		for ((state, _action, reward))  in history.iter().rev(){
			states_input_tensors.push(state.to_tensor(dev).unsqueeze_dim(0));
			accumulated_g =  reward + accumulated_g * alpha ;
			target_outputs.push(accumulated_g);			
		}

		target_outputs.reverse();
		let target_outputs_tensor = Tensor::from_floats(target_outputs.as_slice(), dev);
		let states_input_tensor = Tensor::cat(states_input_tensors, 0);
		let pred_outputs = self.forward(&states_input_tensor);

		let loss = loss.forward_no_reduction(pred_outputs, target_outputs_tensor);
	    let grads = GradientsParams::from_grads(loss.backward(), &self);
		let new_self = optim.step(lr, self, grads);
		new_self
	}
}
