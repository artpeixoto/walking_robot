use std::{
    collections::VecDeque, ops::Not, path::{Path, PathBuf}, str::FromStr, time::Duration
};

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, config::Config, module::Module, nn::loss::MseLoss, optim::{AdaGrad, AdaGradConfig, Adam, AdamConfig, AdamW, AdamWConfig, GradientsParams, Optimizer, SgdConfig}, prelude::Backend, record::{DefaultFileRecorder, FileRecorder, FullPrecisionSettings}, tensor::{backend::AutodiffBackend, Distribution::Uniform, Tensor}};
use comm::{SimulationConnector, SimulationEndpoint};
use models::{a_selector::ASelectorConfig, rs_estimator::{RsEstimator, RsEstimatorConfig}, v_estimator::{VEstimator, VEstimatorConfig}};
use rand::{rngs::ThreadRng, Rng};
use tensor_conversion::TensorConvertible;
use tokio::{main, time::timeout};
use tools::UsedInTrait;
use tracing::{debug, info, trace, warn};
use types::{action::{GameAction, LimbActivation}, state::{GameState, GameUpdate, Reward}};

pub mod comm;
pub mod types;
pub mod traits;
pub mod tensor_conversion;
pub mod tools;
pub mod models;
pub mod procedures;
pub mod state_action_tree;

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
    let dev = WgpuDevice::DefaultDevice;
    let recorder =  DefaultFileRecorder::<FullPrecisionSettings>::new();

    let base_path = PathBuf::from_str("./models/").unwrap();
    let start_from_beggining = base_path.exists().not();

    let rs_estimator_model_path = base_path.join("rs_estimator");
    let mut rs_estimator: RsEstimator<B> = {
        let mut model = 
            RsEstimatorConfig{
                state_layers_size: [80, 80], 
                action_layers_size: [80, 80],
                joint_layers_size: [100,100,100],
                logic: [100,100],
                cut_through: 80,
                end: [120, 80] , 
                // [80; 3], 
                // [80; 2], 
                // 100, 
                // [80; 2] 
            }
            .init(&dev);

        if !start_from_beggining{
            model = model.load_file(&rs_estimator_model_path, &recorder, &dev).unwrap();
        } 
        model
    };
    let v_estimator_model_path = base_path.join("v_estimator");
    let mut v_estimator = {
        let mut model = 
            VEstimatorConfig::new(
                [80, 80, 80], 
                [80, 80], 
                [80, 80], 
                [80, 80] 
            )
            .init(&dev);

        if !start_from_beggining{
            model = model.load_file(&v_estimator_model_path, &recorder, &dev).unwrap();
        } 
        model
    };

    let a_selector_model_path = base_path.join("a_selector");
    let mut a_selector = {
        let mut model = 
            ASelectorConfig::new(
                [80, 80, 80, 80], 
                [80, 80, 80], 
                [80, 80], 
                [80, 80, 80, 80] 
            )
            .init(&dev);
        if !start_from_beggining{
            model = model.load_file(&a_selector_model_path, &recorder, &dev).unwrap();
        } 
        model
    };
    let rs_est_lr = 1e-3;
    let v_est_mc_lr = 1e-3;
    let v_est_td_lr = 1e-2;
    let a_sel_lr = 1e-3;

    let mut rs_est_opt = AdamConfig::new().init();
    let mut v_est_mc_opt = AdamConfig::new().init();
    let mut v_est_td_opt = AdamConfig::new().init(); 
    let mut a_sel_opt = AdamConfig::new().init();

    let alpha = 0.95;

    let mut application_loop = 0 ;

    info!("waiting for connection, baby");
    let mut rng = rand::rng();
    let mut simulation = SimulationConnector::new().connect().await;
    if !start_from_beggining{ 
        info!("No models were found. A first batch of learning from random actions will start"); 

        let mut histories = Vec::new();
        for _i in 0..100{
            let policy = |_state| {
                GameAction::random(&mut rng)
            };
            histories.push(simulation.run_episode(policy).await);
        }

        info!("starting learning batch..."); 
        for history in &histories{
            debug!("rs_estimator will start learning");
            rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut MseLoss::new(), &dev);           

            // debug!("v_estimator will start learning with monte carlo");
            // v_estimator = v_estimator.monte_carlo_train(&history, alpha, v_est_mc_lr, &mut v_est_mc_opt, &mut MseLoss::new(), &dev);

            // debug!("v_estimator will start learning with td");
            // v_estimator = v_estimator.td_train(&history, &rs_estimator, &a_selector, alpha, 20, 10, 2, v_est_td_lr, &mut v_est_td_opt, &mut MseLoss::new(), &dev);

            debug!("a_selector will start learning");
            a_selector = a_selector.train(&history, &rs_estimator, 1000, &v_estimator, &mut a_sel_opt, &mut MseLoss::new(), alpha, a_sel_lr, &dev);
        }            
        info!("saving models...");
        rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
        v_estimator.clone().save_file(v_estimator_model_path.clone(), &recorder).unwrap();
        a_selector.clone().save_file(a_selector_model_path.clone(), &recorder).unwrap();
    }

    '_MAIN_TRAIN_LOOP: loop {          
        let mut histories = VecDeque::new();
        info!("Collecting episodes...");
        for _ in 0..10{
            let policy = |state| {
                a_selector.select_action(&state, &dev)
            };
            let history = simulation.run_episode(policy).await;

            histories.push_front(history);     
        }

        for history in histories.into_iter(){
            info!("training rs_estimator ");
            rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut MseLoss::new(), &dev);

            info!("training v_estimator with monte carlo");
            v_estimator = v_estimator.monte_carlo_train(&history, alpha, v_est_mc_lr, &mut v_est_mc_opt, &mut MseLoss::new(), &dev);

            info!("training v_estimator with td");
            v_estimator = v_estimator.td_train(&history, &rs_estimator, &a_selector, alpha, 20, 10, 5, v_est_td_lr, &mut v_est_td_opt, &mut MseLoss::new(), &dev);

            info!("training a_selector");
            a_selector = a_selector.train(&history, &rs_estimator, 1000, &v_estimator, &mut a_sel_opt, &mut MseLoss::new(), alpha, a_sel_lr, &dev);
        }

        info!("saving models...");
        rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
        v_estimator.clone().save_file(v_estimator_model_path.clone(), &recorder).unwrap();
        a_selector.clone().save_file(a_selector_model_path.clone(), &recorder).unwrap();
        
        application_loop += 1;
    }
}

