use std::iter;

use burn::{module::AutodiffModule, nn::loss::{HuberLoss, Reduction}, optim::{GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use rand::{rng, seq::IndexedRandom};
use tracing::{debug, info};

use crate::{ loss::LossMod, modules::forward_module::ForwardModule, tensor_conversion::TensorConvertible};

pub fn execute_training<B: Backend + AutodiffBackend, M: AutodiffModule<B> + ForwardModule<B>>(
    mut module		: M,
    input			: Tensor<B, 2>,
    target_output	: Tensor<B, 2>,
    loss_mod		: &mut LossMod,
    optim			: &mut impl Optimizer<M, B>,
	lr				: f64,
) -> M {
	debug!("input is: {input}", );
	debug!("target output is: {target_output}", );
	let pred_out = module.forward(input.clone());
	debug!("prediction output is: {pred_out}",);

	let loss = loss_mod.forward_no_reduction(pred_out.clone(), target_output.clone());	
	let red_loss = loss.clone().mean();

	info!("mean loss before training is {}", f32::from_tensor(red_loss));

	let grads = GradientsParams::from_grads(loss.backward(), &module);

	module = optim.step(lr, module, grads);

	let new_pred_out=  module.forward(input.clone());
	let new_loss  = loss_mod.forward(new_pred_out, target_output.clone(), Reduction::Mean);	
	info!("mean loss after training is {}", f32::from_tensor(new_loss));

	module
}
