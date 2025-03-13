
use burn::{nn::loss::{HuberLoss}, optim::Optimizer, prelude::Backend, tensor::backend::AutodiffBackend};
use rand::seq::IndexedRandom;
use tracing::info;

use crate::{loss::LossMod, models::{rs_estimator::RsEstimator, v_estimator::VEstimator}, procedures::{sa_tree_expansion::TreeExpander, train::execute_training::execute_training}, tensor_conversion::TensorConvertibleIterExts, types::{history::History, policy::MultiActionTensorPolicy}};


impl<B: AutodiffBackend> VEstimator<B>{
	pub fn td_train(
		mut self,  
		history				: &History,
		rs_estimator		: &RsEstimator<B>,
		policy				: &mut impl MultiActionTensorPolicy<B>,
		expansion_breadth	: usize,
		expansion_depth 	: usize,
		alpha 				: f32,
		lr					: f64,
		optim				: &mut impl Optimizer<VEstimator<B>, B>,
		loss_mod 			: &mut LossMod,
		dev  				: &<B as Backend>::Device, 
	) -> Self {
		info!("training v_estimator with td");
		let count = history.states.len();
		let states_tensor = history.states.iter().many_to_tensor(dev);
		let target_output ={
			let mut expander = TreeExpander::new(rs_estimator, &self, policy, alpha);
			let trees = expander.expand_states_tensor(states_tensor.clone(), expansion_depth, expansion_breadth);

			trees
				.into_iter()
				.map( |(mut frontier, _tree)| frontier.take_best().1)
				.collect::<Vec<_>>()
				.iter()
				.many_to_tensor(dev)
		}  ;
		self = execute_training(self, states_tensor, target_output, loss_mod, optim, lr);
		self
	}
}