use burn::{config::Config, module::Module, nn::{Gelu, InstanceNorm, InstanceNormConfig, Linear, LinearConfig, Tanh}, prelude::Backend, tensor::Tensor};

use crate::{modules::forward_module::ForwardModule, tensor_conversion::TensorConvertible, tools::UsedInTrait, types::{action::GameAction, policy::{HasDevice, TensorPolicy}, state::GameState}};

#[derive(Config)]
pub struct ASelectorConfig{
    pub linear_layers_size 	    : [usize;4],
    pub logic_layers_size	    : [usize;3],
    pub cut_through_layers_size : [usize;2],
    
    // pub recurrent_layers_size: [usize;3],
    pub end				: [usize;4],
}

impl ASelectorConfig{
   pub fn init<B: Backend>(&self, dev: &<B as Backend>::Device ) -> ASelector<B>{
    	let a = 
        ASelector{
			norm: InstanceNormConfig::new(GameState::VALUES_COUNT).init(dev),
			linear_0: LinearConfig::new(GameState::VALUES_COUNT, self.linear_layers_size[0]).init(dev),
			act_0: Gelu,
			linear_1: LinearConfig::new(self.linear_layers_size[0], self.linear_layers_size[1]).init(dev),
			act_1: Gelu,
			linear_2: LinearConfig::new(self.linear_layers_size[1], self.linear_layers_size[2]).init(dev),
			act_2: Gelu,
			linear_3: LinearConfig::new(self.linear_layers_size[2], self.linear_layers_size[3]).init(dev),
			act_3: Gelu,

			logic_linear_0: LinearConfig::new(self.linear_layers_size[3], self.logic_layers_size[0]).init(dev),
			logic_act_0: Tanh,
			logic_linear_1: LinearConfig::new(self.logic_layers_size[0], self.logic_layers_size[1]).init(dev),
			logic_act_1: Tanh,
			logic_linear_2: LinearConfig::new(self.logic_layers_size[1], self.logic_layers_size[2]).init(dev),
			logic_act_2: Tanh,


			cut_through_linear_0: LinearConfig::new(self.linear_layers_size[3], self.cut_through_layers_size[0]).init(dev),
			cut_through_act_0: Gelu,
			cut_through_linear_1: LinearConfig::new(self.cut_through_layers_size[0], self.cut_through_layers_size[1]).init(dev),
			cut_through_act_1: Gelu,

			end_linear_0:LinearConfig::new(self.logic_layers_size[2] + self.cut_through_layers_size[1],   self.end[0]).init(dev), 
			end_act_0: Gelu,
			end_linear_1:LinearConfig::new(self.end[0], self.end[1]).init(dev), 
			end_act_1: Gelu,
			end_linear_2:LinearConfig::new(self.end[1], self.end[2]).init(dev), 
			end_act_2: Gelu,
			end_linear_3:LinearConfig::new(self.end[2], self.end[3]).init(dev), 
			end_act_3: Gelu,
			end_linear_4:LinearConfig::new(self.end[3], GameAction::VALUES_COUNT).init(dev), 
			end_act_4: Tanh,
        };
        a.forward(&Tensor::zeros([1, GameState::VALUES_COUNT], dev));
        a
    }
}
#[derive(Module, Debug)]
pub struct ASelector<B: Backend>{
	
	norm: InstanceNorm<B>,

    linear_0: Linear<B>,
    act_0: Gelu,
    linear_1: Linear<B>,
    act_1: Gelu,
    linear_2: Linear<B>,
    act_2: Gelu,
    linear_3: Linear<B>,
    act_3: Gelu,


    logic_linear_0: Linear<B>,
    logic_act_0: Tanh,
    logic_linear_1: Linear<B>,
    logic_act_1: Tanh,
    logic_linear_2: Linear<B>,
    logic_act_2: Tanh,


    cut_through_linear_0: Linear<B>,
    cut_through_act_0: Gelu,
    cut_through_linear_1: Linear<B>,
    cut_through_act_1: Gelu,

    // rec_0: Lstm<B>,
    // rec_1: Lstm<B>,
    // rec_2: Lstm<B>,

    end_linear_0: Linear<B>,
	end_act_0: Gelu, 
	end_linear_1: Linear<B>,
    end_act_1: Gelu,
	end_linear_2: Linear<B>,
    end_act_2: Gelu,
	end_linear_3: Linear<B>,
    end_act_3: Gelu,
	end_linear_4: Linear<B>,
    end_act_4: Tanh,
}

impl<'a, B: Backend> HasDevice for &'a ASelector<B>{
	type B = B;

	fn get_dev(&self) -> <Self::B as Backend>::Device {
		self.devices()[0].clone()
	}
}

impl<'a, B: Backend> TensorPolicy<B> for &'a ASelector<B>{
	fn select_action_tensor(&mut self, states_tensor: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(&states_tensor)
	}

	// fn select_action(&mut self, state: &GameState) -> GameAction {
	// }
}


impl<B: Backend> ASelector<B>{
    pub fn forward(&self, states_tensor: &Tensor<B, 2>) -> Tensor<B, 2>{
        let linear_x = 
		 	states_tensor.clone()
			// .used_in(|x| self.norm.forward(x))
			.used_in(|x| self.linear_0.forward(x))
			.used_in(|x| self.act_0.forward(x))
			.used_in(|x| self.linear_1.forward(x))
			.used_in(|x| self.act_1.forward(x))
			.used_in(|x| self.linear_2.forward(x))
			.used_in(|x| self.act_2.forward(x))
			.used_in(|x| self.linear_3.forward(x))
			.used_in(|x| self.act_3.forward(x));

		let logic_x = 
		 	linear_x.clone()
			.used_in(|x| self.logic_linear_0.forward(x))
			.used_in(|x| self.logic_act_0.forward(x))
			.used_in(|x| self.logic_linear_1.forward(x))
			.used_in(|x| self.logic_act_1.forward(x))
			.used_in(|x| self.logic_linear_2.forward(x))
			.used_in(|x| self.logic_act_2.forward(x))
                        ;

		let cut_through_x = 
			linear_x.clone()
			.used_in(|x| self.cut_through_linear_0.forward(x))
			.used_in(|x| self.cut_through_act_0.forward(x))
			.used_in(|x| self.cut_through_linear_1.forward(x))
			.used_in(|x| self.cut_through_act_1.forward(x));


		let end_x = Tensor::cat(vec![logic_x, cut_through_x], 1);

		let output =  
			end_x
			.used_in(|x| self.end_linear_0.forward(x))
			.used_in(|x| self.end_act_0.forward(x))
			.used_in(|x| self.end_linear_1.forward(x))
			.used_in(|x| self.end_act_1.forward(x))
			.used_in(|x| self.end_linear_2.forward(x))
			.used_in(|x| self.end_act_2.forward(x))
			.used_in(|x| self.end_linear_3.forward(x))
			.used_in(|x| self.end_act_3.forward(x))
			.used_in(|x| self.end_linear_4.forward(x))
			.used_in(|x| self.end_act_4.forward(x))
			;
        output
    }
}


impl<B: Backend> ForwardModule<B> for ASelector<B>{
	fn forward(&self, input: Tensor<B,2>) -> Tensor<B,2> {
		self.forward(&input)
	}
}

