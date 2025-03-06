use burn::{module::Module, nn::loss::MseLoss, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}, train::logger::LoggerBackend};
use itertools::Itertools;

use crate::{models::{rs_estimator::RsEstimator, v_estimator::VEstimator}, tensor_conversion::TensorConvertible, types::{action::GameAction, history::{History, HistoryStep}, state::{GameState, Reward}}};



impl<B: AutodiffBackend> RsEstimator<B>{
	pub fn train(
		self,  
		history: &History,
		lr: f64,
		optim: &mut impl Optimizer<RsEstimator<B>, B>,
		loss : &mut MseLoss,
		dev  : &<B as Backend>::Device, 
	) -> Self {
		let mut states_input_tensors: Vec<Tensor<B, 2>>  = Vec::new();
		let mut actions_input_tensors: Vec<Tensor<B, 2>>   = Vec::new();
		let mut target_outputs_tensors: Vec<Tensor<B, 2>>   = Vec::new();

		
		for ((state, action, reward), (next_state, _next_action, _next_reward))  in history.iter().tuple_windows::<(&HistoryStep, &HistoryStep)>(){
			states_input_tensors.push(state.to_tensor(dev).unsqueeze());
			actions_input_tensors.push(action.to_tensor(dev).unsqueeze());

			target_outputs_tensors.push(
				Tensor::cat(
					vec![
						Tensor::<B, 1>::from_floats([*reward], dev), 
						next_state.to_tensor(dev)
					], 
					0
				)
				.unsqueeze()
			);

		}

		let states_input_tensor = Tensor::cat(states_input_tensors, 0);
		let actions_input_tensor = Tensor::cat(actions_input_tensors, 0);
		let target_output_tensor = Tensor::cat(target_outputs_tensors, 0);

		let (r_tensor, s_tensor) = self.forward(&states_input_tensor, &actions_input_tensor);
		let pred_output_tensor = Tensor::cat(vec![r_tensor.unsqueeze_dim(1), s_tensor], 1);

		let loss = loss.forward_no_reduction(pred_output_tensor, target_output_tensor);
	    let grads = GradientsParams::from_grads(loss.backward(), &self);
		let new_self = optim.step(lr, self, grads);
		new_self
	}
}
