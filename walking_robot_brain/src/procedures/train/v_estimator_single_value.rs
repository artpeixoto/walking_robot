use burn::{nn::loss::{HuberLoss}, optim::Optimizer, prelude::Backend, tensor::{backend::AutodiffBackend, Tensor}};
use tracing::info;

use crate::{loss::LossMod, models::v_estimator::VEstimator, procedures::train::execute_training::execute_training, tensor_conversion::TensorConvertibleIterExts, types::state::GameState};

impl<B: AutodiffBackend> VEstimator<B>{

	pub fn single_value_train<'a>(
		mut self,  
		samples	: impl Iterator<Item=&'a GameState>,
		value	: f32,
		lr		: f64,
		optim	: &mut impl Optimizer<VEstimator<B>, B>,
		loss_mod: &mut LossMod,
		dev  	: &<B as Backend>::Device, 
	) -> Self {
        info!("training v_estimator to reach {value}");
		let input = samples.many_to_tensor::<B>(dev);
		let target_output = Tensor::<B, 1>::from_data([value].as_slice(), dev).repeat_dim(0, input.dims()[0]).unsqueeze_dim(1);

		self = execute_training(self, input, target_output, loss_mod, optim, lr);

		self	
	}
}