use std::{cmp::Reverse, collections::{BTreeMap, LinkedList}, ops::Not};

use burn::{nn::loss::MseLoss, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use fix_float::ff32;
use rand::{rng, seq::IndexedRandom, Rng};
use sorted_list::SortedList;
use tracing::trace;

use crate::{models::{a_selector::ASelector, rs_estimator::RsEstimator, v_estimator::VEstimator}, state_action_tree::{Id, RandomActionsStateExpander, StateActionNodeWalker, StateActionTree}, tensor_conversion::TensorConvertibleIterExts, types::history::History};


impl<B: AutodiffBackend> VEstimator<B>{
	pub fn td_train(
		mut self,  
		history: &History,
		rs_estimator	: &RsEstimator<B>,
		a_selector		: &ASelector<B>,
		alpha			: f32,
		samples			: usize,
		expansion_fanout: usize,
		expansion_count : usize,

		lr				: f64,
		optim: &mut impl Optimizer<VEstimator<B>, B>,

		loss : &mut MseLoss,
		dev  : &<B as Backend>::Device, 

	) -> Self {
		let mut states = Vec::new();
		let mut values = Vec::new();
		let mut rng = rng();
		struct NotOpened(BTreeMap<ff32, Id>);
		impl NotOpened{
			fn insert(&mut self, item: Id, value: f32,  ) {
				self.0.insert(ff32!(value), item);
			}	
			fn take_best(&mut self) -> Option<(Id, f32)>{
				let (value, node_id) = self.0.pop_last()?;
				Some((node_id, *value))
			}
		}

		for (game_state, _game_action, _reward) in (0..samples).into_iter().map(|_| history.choose(&mut rng).unwrap()){
			trace!("new game state!");
			let mut frontier = NotOpened(BTreeMap::new()); 
			let mut expander  = RandomActionsStateExpander::new(rs_estimator, a_selector, expansion_fanout, dev);
			let mut tree = StateActionTree::new(game_state.clone());
			let initial_best = self.estimate(game_state, dev);
			frontier.insert(Id::Root, initial_best);

			for _expansion_ix in 0..expansion_count{
				let Some((id, _)) = frontier.take_best() else {panic!()};

				let (depth, reward_sum) = {
					let current_best = tree.start_walking(id);
					let mut depth = 0;
					let mut reward_sum = 0.0;
					let mut walker = current_best;
					while walker.is_root().not(){
						let node_reward = walker.node().unwrap().reward;

						reward_sum = reward_sum * alpha + node_reward;

						depth += 1;
						walker = walker.parent().unwrap()
					}
					(depth, reward_sum)
				};

				tree.expand_once(id, &mut expander);

				let mut children_states = Vec::new();
				let mut children_ids    = Vec::new();
				let mut children_rewards = Vec::new();				

				for &child_id in tree.get_children(id).unwrap(){
					let child_node = tree.get(child_id).right().unwrap();

					children_ids.push(child_id);
					children_states.push(&child_node.state);
					children_rewards.push(child_node.reward);
				}

				let children_values = self.estimate_many(children_states.iter().map(|&c| c), dev);
				for i in 0..children_ids.len(){
					let child_value = children_values[i];
					let child_reward = children_rewards[i];
					let child_id = children_ids[i];

					let child_current_value = reward_sum + alpha.powi(depth+1) * child_reward + alpha.powi(depth+2) * child_value;

					frontier.insert(child_id, child_current_value);
				}
			}

			let (_, value) = frontier.take_best().unwrap();
			states.push(game_state);
			values.push(if value > initial_best {value} else {initial_best});		
		}

		let states_tensor = states.into_iter().many_to_tensor(dev);
		let values_tensor = Tensor::<B, 1>::from_floats(values.as_slice(), dev);
		let values_pred_tensor = self.forward(&states_tensor);

		let loss = loss.forward_no_reduction(values_pred_tensor, values_tensor, );
	    let grads = GradientsParams::from_grads(loss.backward(), &self);

		self = optim.step(lr, self, grads);
		self
	}
}