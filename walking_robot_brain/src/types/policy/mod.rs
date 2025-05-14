
use burn::{prelude::Backend, tensor::Tensor};

use crate::tensor_conversion::TensorConvertible;

use super::{action::GameAction, state::GameState};

pub trait Policy{
	fn select_action(&mut self, state: &GameState) -> GameAction;
}

pub struct TensorFnPolicy<F>(pub F);

impl<B, F> 
	TensorPolicy<B> for TensorFnPolicy<F> 
where 
	B: Backend, 
	F: FnMut(Tensor<B,2> ) -> Tensor<B,2> 
{
	fn select_action_tensor(&mut self, states_tensor: Tensor<B,2>) -> Tensor<B,2> {
		(self.0)(states_tensor)
	}
}

pub struct FnPolicy<F>(pub F);

impl<F: FnMut(&GameState) -> GameAction> Policy for FnPolicy<F>{
	fn select_action(&mut self, state: &GameState) -> GameAction {
		(self.0)(state)
	}
}

pub trait HasDevice{
	type B: Backend;
	fn get_dev(&self) -> <Self::B as Backend>::Device;
}

pub trait TensorPolicy<B: Backend>{
	fn select_action_tensor(&mut self, states_tensor: Tensor<B,2>) -> Tensor<B,2>;
}


impl<B:Backend, P: TensorPolicy<B> + HasDevice<B = B>> Policy for P{
	fn select_action(&mut self, state: &GameState) -> GameAction {
		let dev = self.get_dev();
		let state_tensor = state.to_tensor(&dev).unsqueeze();
		let action_tensor = self.select_action_tensor(state_tensor);
		GameAction::from_tensor(action_tensor.squeeze(0))
	}
}

pub trait MultiActionTensorPolicy<B: Backend>{
	// fn select_actions(&mut self, state: &GameState, count: usize, dev: &<B as Backend>::Device) -> Vec<GameAction>{
	// 	let state_tensor = state.to_tensor(dev).unsqueeze();
	// 	let actions_tensor = self.select_actions_tensor(state_tensor, count).squeeze(0);
	// 	GameAction::many_from_tensor(actions_tensor)
	// }

	fn select_actions_tensor(&mut self, state_tensor: Tensor<B, 2>, count: usize) -> Tensor<B, 3>;
}


pub mod tree_policy;
pub mod noisy_policy;
pub mod nil_policy;
pub mod q_estimator_policy;


// let base_action_tensor = self.policy.forward(&game_state_tensor.clone().unsqueeze()).repeat_dim(0, self.actions_count);
// let action_variations_tensor = Tensor::random([self.actions_count, GameAction::VALUES_COUNT], burn::tensor::Distribution::Uniform(-0.6f64, 0.6f64), self.dev);