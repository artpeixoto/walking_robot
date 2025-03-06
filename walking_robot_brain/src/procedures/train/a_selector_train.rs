use burn::{nn::loss::MseLoss, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use tracing::trace;

use crate::{models::{a_selector::ASelector, rs_estimator::RsEstimator, v_estimator::VEstimator}, state_action_tree::{ExpandTensorResult, RandomActionsStateExpander, StateActionNode, StateExpander}, tensor_conversion::TensorConvertible, types::history::History};

impl<B: AutodiffBackend> ASelector<B>{
	pub fn train(
		mut self,  
		history: &History,
		rs_estimator	: &RsEstimator<B>,
		expansion_fanout: usize,
		value_estimator	: &VEstimator<B>,

		optim: &mut impl Optimizer<ASelector<B>, B>,
		loss : &mut MseLoss,
	
		alpha: f32,
		lr: f64,
		dev  : &<B as Backend>::Device, 
	) -> Self{
		let mut expander = RandomActionsStateExpander::new(rs_estimator, &self, expansion_fanout, dev);
		let mut states_tensors = Vec::new();
		let mut best_actions_tensors = Vec::new();

		trace!("expanding history");

		for (state, action, reward) in history.iter(){
			let state_tensor = state.to_tensor(dev);

			let ExpandTensorResult { actions_tensor, rewards_tensor, next_states_tensor } = expander.expand_tensor(&state_tensor);

			let next_values_tensor = value_estimator.forward(&next_states_tensor);
			let values_tensor = next_values_tensor.mul_scalar(alpha) + rewards_tensor;
			
			let (_best_value, best_value_ix) = values_tensor.max_dim_with_indices(0);
			let best_action = actions_tensor.select(0, best_value_ix).squeeze::<1>(0).detach();
			states_tensors.push(state_tensor);
			best_actions_tensors.push(best_action);
		}
		trace!("building tensors");
		let states_tensor = Tensor::stack(states_tensors, 0);
		let best_actions_tensor = Tensor::stack(best_actions_tensors, 0);
		let best_actions_pred_tensor = self.forward(&states_tensor);

		trace!("learning");
		let loss = loss.forward_no_reduction(best_actions_pred_tensor, best_actions_tensor);
	    let grads = GradientsParams::from_grads(loss.backward(), &self);
		self = optim.step(lr, self, grads);
		self
	}
}