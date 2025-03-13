use burn::{module::Module, nn::{Gelu, HardSigmoid, LeakyRelu, Linear, Relu, Sigmoid, SwiGlu, Tanh}, prelude::Backend, tensor::Tensor};


pub trait ForwardModule<B: Backend>: Module<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2>;
}


impl<B: Backend> ForwardModule<B> for Linear<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		Linear::<B>::forward(&self, input)
	}
}

impl<B: Backend> ForwardModule<B> for Gelu{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}

impl<B: Backend> ForwardModule<B> for Relu{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}

impl<B: Backend> ForwardModule<B> for LeakyRelu{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}


impl<B: Backend> ForwardModule<B> for SwiGlu<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}


impl<B: Backend> ForwardModule<B> for Tanh{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}

impl<B: Backend> ForwardModule<B> for Sigmoid{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}

impl<B: Backend> ForwardModule<B> for HardSigmoid{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(input)
	}
}