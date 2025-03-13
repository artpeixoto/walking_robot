use std::{fmt::{self, Debug}, iter};
use burn::{
    backend::{wgpu::WgpuDevice, Wgpu}, module::{AutodiffModule, Module, ModuleDisplay}, nn::{gru::{Gru, GruConfig}, Gelu, Linear, LinearConfig, LstmConfig}, optim::Optimizer, prelude::Backend, tensor::{backend::AutodiffBackend, BasicOps, Int, Tensor, TensorKind}
};
use either::Either::{self, Left, Right};
use tracing::{info, warn};

use crate::{loss::LossMod, modules::{forward_module::ForwardModule, sequential::{LinearSequential, LinearSequentialConfig}}, procedures::train::execute_training::execute_training, tensor_conversion::TensorConvertible, tools::{UsedInTrait, WindowsExt}, types::{state::GameState, tensor_types::{GameActionsTensor, GameStatesTensor}}};

use super::builders::WINDOW_SIZE;

pub struct SaEncoderConfig{
	pub input_linear : Vec<usize>,
	pub recurrent	 : Vec<usize>,
	pub output_linear: Vec<usize>,
	pub final_output : usize,
}
impl SaEncoderConfig{
	pub fn init<B:Backend>(&self, dev: &<B as Backend>::Device) -> SaEncoder<B>{
		let input_linears = LinearSequentialConfig{sizes: self.input_linear.clone(), act: Gelu}.init(dev);			
		
		let mut recurrents = Vec::new();
		let mut prev_size = self.input_linear.last().unwrap().clone();
		for &size in self.recurrent.iter().chain(std::iter::once(&self.output_linear[0])){
			let gru = GruConfig::new(prev_size, size, true).init(dev);
			let gelu = Gelu;

			recurrents.push((gru, gelu));
			prev_size = size;
		}

		let output_linears = LinearSequentialConfig{sizes: self.output_linear.clone(), act: Gelu}.init(dev);			
		let final_output = LinearConfig::new(*self.output_linear.last().unwrap(), self.final_output).init(dev);
		SaEncoder { linears_0: input_linears, recurrents , linears_1: output_linears, final_output  }
	}
}



#[derive(Debug, Module)]
pub struct SaEncoder<B: Backend> 
{
    linears_0: LinearSequential<B, Gelu>,
	recurrents: Vec<( Gru<B>, Gelu )>	,
	linears_1: LinearSequential<B, Gelu>,
	final_output: Linear<B>
}

impl<B: Backend> SaEncoder<B>{}

impl<B: Backend> SaEncoder<B>
{
	fn forward_seq(
		&self, 
		state	: Tensor<B, 2>, 
		action	: Tensor<B, 2>, 
		rec_act	: Option<Vec<Tensor<B, 2>>>
	) -> (Tensor<B,2>, Vec<Tensor<B, 2>>) {

		let input = Tensor::cat(vec![state, action], 1);

		let mut x 				: Tensor<B, 2> 		= 
			self.linears_0.forward(input);

		let mut next_rec_acts 	: Vec<Tensor<B, 2>>	= 
			Vec::new();

		let rec_act = match rec_act{
			Some(rec_act) 	=> Left(rec_act.into_iter().map(|a| Some(a))),
			None 			=> Right(std::iter::repeat(None)),
		};

		for (rec_act, ( gru, gelu)) in iter::zip(rec_act.into_iter(), self.recurrents.iter()){
			let count = x.dims()[0];
			let next_act = 
				gru.forward(x.unsqueeze(), rec_act.map(|a| a.unsqueeze())).reshape([count as i32, -1]);

			let next_x = gelu.forward(next_act.clone());
			next_rec_acts.push(next_act);

			x = next_x;
		}
		let x = self.linears_1.forward(x);
		let x = self.final_output.forward(x);
		(x, next_rec_acts)
	}
}
pub struct SaDecoderConfig{
	pub linears: Vec<usize>,
}
impl SaDecoderConfig{
	pub fn init<B: Backend>(&self, dev: &<B as Backend>::Device) -> SaDecoder<B>{
		let mut linears_sizes = self.linears.clone();
		let output_size = linears_sizes.pop().unwrap(); 
		let &linears_last_size = linears_sizes.last().unwrap();
		let linears = LinearSequentialConfig{sizes: linears_sizes, act: Gelu}.init(dev);
		let output = LinearConfig::new(linears_last_size, output_size).init(dev);

		SaDecoder{
			linears,
			output
		}
	}
}


#[derive(Debug, Module)]
pub struct SaDecoder<B: Backend> 
{
	linears: LinearSequential<B, Gelu>,
	output: Linear<B>
}

impl<B: Backend> ForwardModule<B> for SaDecoder<B>
{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.output.forward(self.linears.forward(input))
	}
}



#[derive(Debug, Module)]
pub struct SaEnDec<B: Backend> 
{
	pub enc: SaEncoder<B>,
	pub dec: SaDecoder<B>
}

impl<B: Backend> ForwardModule<B> for SaEnDec<B>
{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		let sa = GameState::take_tensor_part(input, 1);
		let (state, action) = (sa.value, sa.rest);
		
		let (encoded, _) = self.enc.forward_seq(state, action, None);
		
		let count = encoded.dims()[0];
		warn!("{encoded}");
		let encoded = encoded.slice([(WINDOW_SIZE as usize - 1).. (count)]);

		let decoded = self.dec.forward(encoded);
		decoded
	}
}

impl<B: Backend> SaEnDec<B>
where 
	B: AutodiffBackend,
{
	pub fn train(
		self, 
		states: GameStatesTensor<B>,
		actions: GameActionsTensor<B>,
		loss_mod: &mut LossMod, 
		optim: &mut impl Optimizer<Self, B> , 
		lr: f64
	) -> Self{
		let stacked_states = states.clone().windows(WINDOW_SIZE);
		let stacked_actions = actions.clone().windows(WINDOW_SIZE);
		let stacked_input = Tensor::cat(vec![stacked_states, stacked_actions], 1);

		let input = Tensor::cat(vec![states, actions], 1);

		execute_training(self, input, stacked_input.clone(), loss_mod, optim, lr)
	}
}


