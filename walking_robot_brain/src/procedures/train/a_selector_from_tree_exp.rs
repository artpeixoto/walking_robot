use burn::{nn::loss::{Reduction}, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use rand::rng;
use tracing::info;

use crate::{loss::LossMod, models::{a_selector::ASelector, rs_estimator::RsEstimator, v_estimator::VEstimator}, procedures::sa_tree_expansion::TreeExpander, tensor_conversion::TensorConvertibleIterExts, tools::UsedInTrait, types::{history::History, policy::noisy_policy::NoisyPolicy}};


impl<B: AutodiffBackend> ASelector<B>{
	pub fn train_from_tree_exp(
		mut self,  
		history	: &History,
		rs_estimator: &RsEstimator<B>,
		v_estimator : &VEstimator<B>, 
		optim	: &mut impl Optimizer<ASelector<B>, B>,
		loss_mod 	: &mut LossMod,

		expansion_breadth: usize,
		expansion_depth  : usize, 

		actions_noise	 : f32,
		alpha	: f32,
		lr		: f64,
		dev  	: &<B as Backend>::Device, 
	) -> Self {
		info!("training a_selector from tree_expansion");
		let count = history.states.len();
		let states = history.states.iter().many_to_tensor(dev);


		let target_outputs = {
			let mut policy = NoisyPolicy::new(&self, actions_noise, rng());
			let mut expander = TreeExpander::new(rs_estimator, v_estimator, &mut policy, alpha);

			let trees = expander.expand_states_tensor(states.clone(), expansion_depth, expansion_breadth);

			trees
				.into_iter()
				.map(|(mut frontier, tree)|{
					let (best_id, _) = frontier.take_best();
					let mut node = tree.get(best_id);
					while node.depth() > 1 {
						node = node.parent().unwrap();
					}
					node.node().unwrap().action.clone()
				})
				.collect::<Vec<_>>()
				.used_in(|ts| Tensor::stack(ts, 0))
				.detach()
		};

		for _ in 0..count/10{
			let pred_outputs = self.forward(&states);

			let loss = loss_mod.forward_no_reduction(pred_outputs.clone(), target_outputs.clone());
			let red_loss = loss_mod.forward(pred_outputs, target_outputs.clone(), Reduction::Auto);
			// inform_loss(red_loss.clone());
			let grads = GradientsParams::from_grads(loss.backward(), &self);
			self = optim.step(lr, self, grads);
		}

		self	
	}
}

