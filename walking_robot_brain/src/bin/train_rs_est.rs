use std::{
    iter, ops::Not, path::PathBuf, str::FromStr
};
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, grad_clipping::GradientClippingConfig, module::Module, nn::loss::{HuberLoss, HuberLossConfig, MseLoss}, optim::{momentum::MomentumConfig, AdamConfig, AdamWConfig, SgdConfig}, prelude::Backend, record::{DefaultFileRecorder, FullPrecisionSettings}};

use walking_robot_brain::{comm::SimulationConnector, models::builders::make_rs_estimator, types::policy::{noisy_policy::NoisyPolicy, FnPolicy}};
use walking_robot_brain::models::{a_selector::ASelectorConfig, rs_estimator::{RsEstimator, RsEstimatorConfig}, v_estimator::VEstimatorConfig};
use rand::{seq::IndexedRandom, Rng};
use tracing::{info, warn};
use walking_robot_brain::types::{action::GameAction, state::GameState};

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
    let recorder =  DefaultFileRecorder::<FullPrecisionSettings>::new();
    let base_path = PathBuf::from_str("./models/").unwrap();
    let start_from_beggining = base_path.exists().not();

    let rs_estimator_model_path = base_path.join("rs_estimator");
    let mut rs_estimator: RsEstimator<B> = {
        let mut model = make_rs_estimator(&dev);

        if !start_from_beggining{
            model = model.load_file(&rs_estimator_model_path, &recorder, &dev).unwrap();
        } 
        model
    };
    
    let rs_est_lr = 0.001;
    
    
    let opt_config =
        AdamWConfig::new()
        // .with_gradient_clipping(Some(GradientClippingConfig::Value(1000.0)))
        ;

    let mut rs_est_opt = opt_config.clone().init();

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

        for history in iter::from_fn(||histories.choose(&mut rng)).take(30){
            rs_estimator = rs_estimator.train(
                &history.states, 
                &history.actions, 
                &history.rewards, 
                rs_est_lr, 
                &mut rs_est_opt, 
                &mut MseLoss::new(),
                &dev
            );           
        }            

        info!("saving models...");
        rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
    }

}

