use burn::{nn::loss::{ Reduction}, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::backend::AutodiffBackend};
use tracing::info;

use crate::{loss::LossMod, models::a_selector::ASelector, tensor_conversion::TensorConvertibleIterExts, types::history::History};


impl<B: AutodiffBackend> ASelector<B>{
	pub fn train_from_history(
		mut self,  
		history	: &History,

		optim	: &mut impl Optimizer<ASelector<B>, B>,
		loss_mod 	: &mut LossMod,
	
		lr		: f64,
		dev  	: &<B as Backend>::Device, 
	) -> Self {
		let count = history.states.len();
        info!("training a_selector from history");
		let states = history.states.iter().many_to_tensor(dev);
		let target_outputs = history.actions.iter().many_to_tensor(dev);		
		for _ in 0..count/10{
			let pred_outputs = self.forward(&states);

			let loss = loss_mod.forward_no_reduction(pred_outputs.clone(), target_outputs.clone());
			let red_loss = loss_mod.forward(pred_outputs, target_outputs.clone(), Reduction::Auto);
			// inform_loss( red_loss.clone());
			let grads = GradientsParams::from_grads(loss.backward(), &self);
			self = optim.step(lr, self, grads);
		}
		self
	}
}

