use burn::tensor::Tensor;

pub type GameActionsTensor<B> = Tensor<B,2>;
pub type GameStatesTensor<B>  = Tensor<B,2>;
pub type GameRewardTensors<B> = Tensor<B,1>;

pub type GameActionTensor<B> = Tensor<B,1>;
pub type GameStateTensor<B>  = Tensor<B,1>;
pub type GameRewardTensor<B> = Tensor<B,1>;