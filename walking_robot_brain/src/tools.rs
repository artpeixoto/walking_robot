use burn::{
    prelude::Backend,
    tensor::{BasicOps, Tensor, TensorKind},
};

pub trait UsedInTrait: Sized {
    fn used_in<O, F: FnOnce(Self) -> O>(self, f: F) -> O {
        f(self)
    }
}
impl<T: Sized> UsedInTrait for T {}

pub trait ToF32Ext {
    fn to_scalar_f32(self) -> f32;
}

pub trait WindowsExt

{
	fn windows(self, stack_size: i64) -> Self;
	
}

impl<B, K> WindowsExt for Tensor<B, 2, K>
where
	B: Backend,
	K: TensorKind<B> + BasicOps<B>,	
{
	fn windows(self, stack_size: i64) -> Self {
		let input_size = *&self.dims()[0] as i64;
		let mut layers = Vec::new();
		for ix in 0..stack_size {
			let layer = 
				self
				.clone()
				.slice([
					Some((ix, input_size - stack_size + ix + 1)), 
					None
				]);
			layers.push(layer);
		}
		Tensor::cat(layers, 1)
	}
}
#[cfg(test)]
mod test{
    use burn::{backend::{wgpu::WgpuDevice, Wgpu}, tensor::{Int, Tensor}};

    use crate::tools::WindowsExt;


	#[test]
	pub fn visualize_stack(){
		type B = Wgpu;
		let device = WgpuDevice::default();
		let input_tensor = Tensor::<B, 1, Int>::arange(0..10, &device).reshape([-1, 1]);

		println!("input tensor is: {input_tensor}");

		let stacked_tensor = input_tensor.windows(3);

		println!("stacked tensor is {stacked_tensor}");
	}
}