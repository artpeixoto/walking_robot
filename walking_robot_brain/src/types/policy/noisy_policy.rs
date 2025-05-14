use burn::prelude::{Backend, Tensor};
use itertools::Itertools;
use rand::{rngs::ThreadRng, Rng};

use crate::{tensor_conversion::TensorConvertible, tools::UsedInTrait, types::{action::GameAction, state::GameState}};

use super::{MultiActionTensorPolicy, Policy, TensorPolicy};

pub struct NoisyPolicy<P>{
	inner		: P,
	noise_amount: f32, 
	rng			: ThreadRng
}

impl<P> NoisyPolicy<P> {
	pub fn new(inner: P, noise_amount: f32, rng: ThreadRng) -> Self {
		Self { inner, noise_amount, rng }
	}
	fn add_noise_to_action(&mut self, action: GameAction) -> GameAction{
		let mut noise = || -> f32 {self.rng.random::<f32>() * 2.0 * self.noise_amount};
		action.iterate_values().map(|v| v + noise()).collect_vec().used_in(|v| GameAction::from_values(&v))
	}
}

impl<P: Policy> Policy for NoisyPolicy<P> {
	fn select_action(&mut self, state: &GameState) -> GameAction {
		let inner = self.inner.select_action(state);
		self.add_noise_to_action(inner)
	}
}



impl<B: Backend, P: TensorPolicy<B>> MultiActionTensorPolicy<B> for NoisyPolicy<P>{
	
	fn select_actions_tensor(&mut self, state_tensor: Tensor<B, 2>, actions_per_state: usize) -> Tensor<B, 3> {
		let dev = state_tensor.device();
		let state_count = state_tensor.dims()[0];
		let actions = self.inner.select_action_tensor(state_tensor).unsqueeze_dim(1).repeat_dim(1, actions_per_state);
		let noise = Tensor::random([state_count, actions_per_state, GameAction::VALUES_COUNT], burn::tensor::Distribution::Uniform(- self.noise_amount as f64, self.noise_amount as f64), &dev);
		let actions_tensor =  (actions + noise).clamp(-1.0, 1.0); //probably not the best idea...
		actions_tensor
	}
}
