use burn::prelude::{Backend, Tensor};

use crate::{tensor_conversion::TensorConvertible, types::{action::GameAction, state::GameState}};

use super::{Policy, TensorPolicy};
pub struct NilPolicy;

impl Policy for NilPolicy{
	fn select_action(&mut self, _state: &GameState) -> GameAction {
		GameAction::default()
	}
}
impl<B: Backend> TensorPolicy<B> for NilPolicy{
	fn select_action_tensor(&mut self, states_tensor: Tensor<B,2>) -> Tensor<B,2> {
		Tensor::zeros([states_tensor.dims()[0], GameAction::VALUES_COUNT], &states_tensor.device())
	}
}