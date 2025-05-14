use std::collections::VecDeque;

use burn::prelude::{Backend, Tensor};
use rand::rng;

use crate::{models::{builders::WINDOW_SIZE, q_estimator::QEstimator}, modules::forward_module::ForwardModule, tensor_conversion::TensorConvertible, types::{action::GameAction, state::GameState}};

use super::{nil_policy::NilPolicy, noisy_policy::NoisyPolicy, MultiActionTensorPolicy, Policy, TensorPolicy};

pub struct QEstimatorPolicy<'a, B: Backend>{
	pub q_estimator		: &'a QEstimator<B>,
	pub policy			: NoisyPolicy<NilPolicy>,
	pub count			: usize,
	pub previous_states	: VecDeque<Tensor<B, 1>>,
	pub previous_actions: VecDeque<Tensor<B, 1>>,
	pub dev				: &'a <B as Backend>::Device, 
}

impl<'a, B: Backend> QEstimatorPolicy<'a, B>{
	pub fn new(q_estimator: &'a QEstimator<B>, count: usize, dev:&'a <B as Backend>::Device ) -> Self{
		Self{
			q_estimator,
			count,
			dev,
			previous_actions: VecDeque::new(),
			previous_states: VecDeque::new(),
			policy: NoisyPolicy::new(NilPolicy, 1.0, rng())
		}
	}
}

impl<'a, B: Backend> Policy for QEstimatorPolicy<'a, B>{
	fn select_action(&mut self, state: &GameState) -> GameAction {
		let state = state.to_tensor(self.dev);
		if self.previous_states.len() >= (WINDOW_SIZE as usize){
			self.previous_actions.pop_back();
			self.previous_states.pop_back();

			self.previous_states.push_front(state.clone());

			let actions: Tensor<B, 2> = self.policy.select_actions_tensor(state.unsqueeze(), self.count).squeeze(0);

			let base_tensor =  
				Tensor::cat(
					vec![ 
						Tensor::cat(self.previous_states.iter().cloned().collect(),0 ) ,
						Tensor::cat(self.previous_actions.iter().cloned().collect(),0)
					], 
					0
				)
				.unsqueeze()
				.repeat_dim(0, self.count);
			
			let q_est_input = Tensor::cat(vec![base_tensor, actions.clone()], 1);
			let qs 			= self.q_estimator.forward(q_est_input);
			let (_, best_act_ix) = qs.max_dim_with_indices(0);

			let action_tensor = actions.select(0, best_act_ix.squeeze(1)).squeeze(0);
			self.previous_actions.push_front(action_tensor.clone());
			let action = GameAction::from_tensor(action_tensor);
			return action;
		} else {
			let action = GameAction::default();
			let action_tensor = action.to_tensor(self.dev);
			self.previous_actions.push_front(action_tensor);
			self.previous_states.push_front(state);
			return action;
		}
	}
}