use std::{collections::BTreeMap, iter};
use burn::{module::Module, prelude::Backend, tensor::{Int, Tensor}};
use fix_float::ff32;
use itertools::Itertools;
use crate::{models::{rs_estimator::RsEstimator, v_estimator::VEstimator}, tensor_conversion::TensorConvertible, tools::UsedInTrait, types::{action::GameAction, policy::{HasDevice, MultiActionTensorPolicy}, sa_tensor_tree::{Id, SaTensorTree}, state::GameState}};


pub struct TreeExpander<'a, B: Backend, P: MultiActionTensorPolicy<B>> {
    rs_estimator: &'a RsEstimator<B>,
    v_estimator	: &'a VEstimator<B>,
    policy		: &'a mut P,
	alpha		: f32
}

impl<'a, B: Backend, P: MultiActionTensorPolicy<B>> HasDevice for TreeExpander<'a, B, P> {
	type B = B;
	fn get_dev(&self) -> <Self::B as Backend>::Device {
		self.rs_estimator.devices()[0].clone()
	}
}



impl<'a, B: Backend, P: MultiActionTensorPolicy<B>> TreeExpander<'a, B, P> {
	pub fn new(rs_estimator: &'a RsEstimator<B>, v_estimator: &'a VEstimator<B>, policy: &'a mut P, alpha: f32) -> Self {
			Self { rs_estimator, v_estimator, policy, alpha }
		}
	
	pub fn expand_states_tensor(
		&mut self,
        states		: Tensor<B, 2>,
		depth		: usize,
		breadth		: usize,
 	) -> Vec<(NotOpened, SaTensorTree<B>)>{
		let dev = self.get_dev();

		let states_values = self.v_estimator.forward(&states).unsqueeze_dim(1).used_in(f32::many_from_tensor);
	
		let mut trees = 
		 	iter::zip(states.iter_dim(0), states_values.into_iter())
			.map(|(s, v)| {
				( NotOpened::new(v)
				, SaTensorTree::new(s.squeeze(0), self.alpha, &dev)
				)
			})
			.collect_vec();
		
		for _ in 0..depth{
			let (ids,  states, depths, acc_rewards) = {
				let len = trees.len();
				let mut ids 		 = Vec::with_capacity(len);
				let mut depths 		 = Vec::with_capacity(len);
				let mut states 	 	 = Vec::with_capacity(len);
				let mut acc_rewards  = Vec::with_capacity(len);
					trees
					.iter_mut()
					.enumerate()
					.for_each(|(ix, (frontier, tree))|{
						let (id, _value) = frontier.take_best();
						let node = tree.get(id);
						ids.push(id);
						depths.push(node.depth());
						states.push(node.state().clone());
						acc_rewards.push(node.acc_reward())
					});
					
				(ids,states, depths, acc_rewards )
			};
			let acc_rewards_tensor = Tensor::<B, 1>::from_data(acc_rewards.as_slice(), &dev);
			let depths_tensor = Tensor::<B, 1, Int>::from_data(depths.as_slice(), &dev);
			let states_count = states.len();
			let states_tensor = Tensor::stack(states, 0);

			let children_actions = self.policy.select_actions_tensor(states_tensor.clone(), breadth);

			// open all
			let (children_rewards, children_states, children_local_values) = {
				let states_tensor = states_tensor.unsqueeze_dim::<3>(1).repeat_dim(1, breadth).reshape([states_count*breadth, GameState::VALUES_COUNT]);
				let actions_tensor = children_actions.clone().reshape([states_count*breadth, GameAction::VALUES_COUNT]);

				let (children_rewards, children_states) = self.rs_estimator.forward(&states_tensor, &actions_tensor);

				let values_tensor= self.v_estimator.forward(&children_states).reshape([states_count, breadth]);
				let children_rewards = children_rewards.reshape([states_count, breadth]);
				let children_states = children_states.reshape([states_count, breadth, GameState::VALUES_COUNT]);

				(children_rewards, children_states, values_tensor)
			};

			// calculate all children values
			let (children_acc_rewards, children_values) = {
				let alpha_tensor = Tensor::<B, 1>::from_data([self.alpha].as_slice(), &dev).repeat_dim(0, states_count).unsqueeze_dim(1).repeat_dim(1, breadth );
				let reward_depths = depths_tensor	.unsqueeze_dim(1).repeat_dim(1, breadth);	

				let reward_alphas = alpha_tensor.clone().powf(reward_depths.float());
				let value_alphas = reward_alphas.clone() * alpha_tensor;

				let children_acc_rewards = children_rewards * reward_alphas ;

				(
					children_acc_rewards.clone(), 
					acc_rewards_tensor.unsqueeze_dim(1).repeat_dim(1, breadth) + children_acc_rewards + children_local_values * value_alphas)
			};

			let children_values = children_values.to_data().to_vec::<f32>().unwrap();

			let get_child_value = |parent_ix: usize, this_ix: usize|{
				children_values[parent_ix * breadth + this_ix]
			};

			trees.iter_mut().enumerate().for_each(|(parent_ix, (frontier, tree))|{
				let parent_id = ids[parent_ix];
				let parent_tensor_ix = Some((parent_ix as i64, parent_ix as i64 + 1));
				let children_states = children_states.clone().slice([parent_tensor_ix, None, None]).squeeze(0);
				let children_actions = children_actions.clone().slice([parent_tensor_ix, None, None]).squeeze(0);
				let children_acc_rewards =  children_acc_rewards.clone().slice([parent_tensor_ix, None]).squeeze(0);

				let children_ids = tree.add_children(parent_id, (children_actions, children_acc_rewards, children_states));

				for (ix, child_id) in children_ids.into_iter().enumerate(){
					frontier.insert(child_id, get_child_value(parent_ix, ix));
				}
			})
		}

		trees
	}
}

pub struct NotOpened(pub BTreeMap<ff32, Id>);
impl NotOpened{
	pub fn new(root_value: f32) -> Self{
		let mut res = Self(BTreeMap::default());
		res.insert(Id::Root, root_value);	
		res
	}
	pub fn insert(&mut self, item: Id, value: f32,  ) {
		self.0.insert(fix_float::ff32::try_from(value).unwrap_or_default(), item);
	}	
	pub fn take_best(&mut self) -> (Id, f32) {
		let (value, node_id) = self.0.pop_last().unwrap();
		(node_id, *value)
	}
}

