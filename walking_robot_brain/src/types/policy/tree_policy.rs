
use burn::{module::Module, prelude::Backend, tensor::Tensor};
use itertools::Itertools;

use crate::{
    procedures::sa_tree_expansion::TreeExpander, tools::UsedInTrait
};

use super::{HasDevice, MultiActionTensorPolicy, TensorPolicy};

pub struct TreeExpPolicy<'a, B: Backend, P: MultiActionTensorPolicy<B>> {
	tree_expander	: TreeExpander<'a, B, P>,
   	depth			: usize,
    breadth			: usize,
}

impl<'a, B: Backend, P: MultiActionTensorPolicy<B>> HasDevice for TreeExpPolicy<'a, B, P> {
	type B = B;
	fn get_dev(&self) -> <Self::B as Backend>::Device {
		self.tree_expander.get_dev()
	}
}


pub struct ExpandTensorResult<B: Backend> {
	pub actions_tensor		: Tensor<B, 2>,
    pub rewards_tensor		: Tensor<B, 1>,
    pub next_states_tensor	: Tensor<B, 2>,
}

impl<'a, B: Backend, P: MultiActionTensorPolicy<B>> TreeExpPolicy<'a, B, P> {
    pub fn new(
		tree_expander: TreeExpander<'a, B, P>,
        depth: usize,
        breadth: usize,
    ) -> Self {
        Self {
			tree_expander,
            depth,
            breadth,
        }
    }

}

impl<'a, B: Backend, P: MultiActionTensorPolicy<B>> TensorPolicy<B> for TreeExpPolicy<'a, B, P> {
    fn select_action_tensor(
        &mut self,
        states: Tensor<B, 2>,
    ) -> Tensor<B, 2>{
		let trees = self.tree_expander.expand_states_tensor(states, self.depth, self.breadth);

		// now we must pick the best current action for each child
		let best_actions = 
			trees.into_iter()
			.map(|(mut frontier, tree)|{
				let (best_id, _best_value) = frontier.take_best();
				let mut best = tree.get(best_id);
				while best.depth() > 1{ //follow down until the first leaf
					best = best.parent().unwrap()
				}
				best.node().unwrap().action.clone()
			})
			.collect::<Vec<_>>()
			.used_in(|ts| Tensor::stack(ts, 0));
		
		best_actions
    }
}
