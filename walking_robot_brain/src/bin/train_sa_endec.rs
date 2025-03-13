use std::{iter, ops::Not, path::PathBuf, str::FromStr};

use burn::{backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, grad_clipping::GradientClippingConfig, module::Module, nn::loss::{HuberLossConfig, MseLoss}, optim::{AdamConfig, SgdConfig}, prelude::Backend, record::{DefaultFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder}};
use rand::seq::IndexedRandom;
use tracing::{info, warn};
use walking_robot_brain::{comm::SimulationConnector, loss::LossMod, models::{builders::{make_sa_endec, SA_DEC_MODEL_PATH, SA_ENC_MODEL_PATH}, rs_estimator::{RsEstimator, RsEstimatorConfig}}, types::{action::GameAction, policy::FnPolicy, state::GameState}};

fn main() {
    tokio
    ::runtime
    ::Builder
    ::new_current_thread()
    .enable_all()
    .build()
    .unwrap()
    .block_on(async_main())
}

async fn async_main(){
    pretty_env_logger::init_timed();
    warn!("yeah baby");
    type B = Autodiff<Wgpu<f32, i32>>;
	<B as Backend>::seed(420);
    let dev = WgpuDevice::DefaultDevice;
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    let mut sa_endec = make_sa_endec::<B>(&dev);
    let lr = 0.0001;

    let opt_config =
        // SgdConfig::new()
        // .with_gradient_clipping(Some(GradientClippingConfig::Value(1000.0)))
        // .with_momentum(Some(MomentumConfig::new().with_momentum(0.1)))
		AdamConfig::new()
        ;

    let mut optim = opt_config.clone().init();

    info!("waiting for connection, baby");
    let mut rng = rand::rng();
    let mut simulation = SimulationConnector::new().connect().await;

    loop{
        info!("Starting a new batch"); 

        let mut histories = Vec::new();
        for _i in 0..10{
            let policy = |_state: &GameState| {
                GameAction::random(&mut rng)
            };
            histories.push(simulation.run_episode(&mut FnPolicy(policy)).await.to_tensor_history(&dev));
        }

        for history in iter::from_fn(||histories.choose(&mut rng)).take(100){
            sa_endec = sa_endec.train(history.states.clone(), history.actions.clone(), &mut MseLoss::new(), &mut optim, lr);           
        }            

        info!("saving models...");
        sa_endec.enc.clone().save_file(SA_ENC_MODEL_PATH.as_path(), &recorder).unwrap();
        sa_endec.dec.clone().save_file(SA_DEC_MODEL_PATH.as_path(), &recorder).unwrap();
    }
}

