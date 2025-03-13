use std::{fmt::{Debug}, iter};

use burn::{module::{Module, ModuleDisplay}, nn::{Linear, LinearConfig}, prelude::{Backend, Tensor}};


use super::forward_module::ForwardModule;

#[derive(Clone, Debug)]
pub struct LinearSequentialConfig<M>{
	pub sizes: Vec<usize>,
	pub act: M
}

impl<M> LinearSequentialConfig<M>{
	pub fn init<B: Backend>(&self, dev: &<B as Backend>::Device ) -> LinearSequential< B, M> 
	where 
		M: Module<B> + Clone,
	{
		let mut linears = Vec::new();
		let mut acts 	= Vec::new();
		for ix in 0..(self.sizes.len()-1 ){
			let in_size = *&self.sizes[ix];
			let out_size = *&self.sizes[ix + 1];

			linears.push(LinearConfig::new(in_size, out_size).init(dev));
			acts.push(self.act.clone());
		}

		LinearSequential{
			linears: linears,
			activations: acts
		}
	}
}

#[derive(Debug, Module)]
pub struct LinearSequential< B: Backend, M: Module<B> >{
	linears		: Vec<Linear<B>>,
	activations : Vec<M>,
}

impl<M, B> LinearSequential<B, M> 
where 
	M: Module<B> + Debug,
	B: Backend, 
{
}

impl<B, M> ForwardModule<B> for LinearSequential<B, M> 
where 
	M: Module<B> + ForwardModule<B> + Debug  + ModuleDisplay,
	B: Backend, 
{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		let mut current = input;
		for (linear, act) in iter::zip(&self.linears[0..], self.activations.iter()){
			current = act.forward(linear.forward(current));
		}
		current
	}
}

