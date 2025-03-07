use std::{
    collections::VecDeque, iter, ops::Not, path::{Path, PathBuf}, str::FromStr, time::Duration
};

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, config::Config, module::Module, nn::loss::MseLoss, optim::{AdaGrad, AdaGradConfig, Adam, AdamConfig, AdamW, AdamWConfig, GradientsParams, Optimizer, SgdConfig}, prelude::Backend, record::{DefaultFileRecorder, FileRecorder, FullPrecisionSettings}, tensor::{backend::AutodiffBackend, Distribution::Uniform, Tensor}};
use comm::{SimulationConnector, SimulationEndpoint};
use models::{a_selector::ASelectorConfig, rs_estimator::{RsEstimator, RsEstimatorConfig}, v_estimator::{VEstimator, VEstimatorConfig}};
use rand::{rngs::ThreadRng, seq::IndexedRandom, Rng};
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
                state_layers_size: [1000, 1000], 
                action_layers_size: [1000, 1000],
                joint_layers_size: [2000,2000,2000],
                logic: [2000,2000],
                cut_through: 2000,
                end: [3000, 1000] , 
                // [8000; 3], 
                // [8000; 2], 
                // 100, 
                // [8000; 2] 
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
                [1000, 1000, 1000], 
                [1000, 1000], 
                [1000, 1000], 
                [1000, 1000] 
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
                [1000, 1000, 1000, 1000], 
                [1000, 1000, 1000], 
                [1000, 1000],
                [1000, 1000, 1000, 1000] 
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


    info!("waiting for connection, baby");
    let mut rng = rand::rng();
    let mut simulation = SimulationConnector::new().connect().await;
    if start_from_beggining{ 
        info!("No models were found. A first batch of learning from random actions will start"); 

        let mut histories = Vec::new();
        for _i in 0..20{
            let policy = |_state| {
                GameAction::random(&mut rng)
            };
            histories.push(simulation.run_episode(policy).await);
        }
        info!("starting learning batch..."); 
        for history in iter::from_fn(||histories.choose(&mut rng)).take(1000){
            debug!("training rs_estimator ");
            rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut MseLoss::new(), &dev);           
        }            
        info!("saving models...");
        rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
        v_estimator.clone().save_file(v_estimator_model_path.clone(), &recorder).unwrap();
        a_selector.clone().save_file(a_selector_model_path.clone(), &recorder).unwrap();

        // const INITIAL_TRAINING_LOOP_COUNT:usize = 100;
        // info!("Starting training");
        // for initial_training_loop in 0..INITIAL_TRAINING_LOOP_COUNT{
        //     info!("Collecting episodes...");
        //     let mut histories = Vec::new();
        //     for _ in 0..10{
        //         let policy = |state| {
        //             if rng.random::<f32>() < ( initial_training_loop as f32 / INITIAL_TRAINING_LOOP_COUNT as f32){
        //                 a_selector.select_action(&state, &dev)
        //             } else {
        //                 GameAction::random(&mut rng)
        //             }
        //         };
        //         let history = simulation.run_episode(policy).await;

        //         histories.push(history);     
        //     }

        //     for history in histories.into_iter(){
        //         info!("training rs_estimator ");
        //         rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut MseLoss::new(), &dev);

        //         info!("training v_estimator with monte carlo");
        //         v_estimator = v_estimator.monte_carlo_train(&history, alpha, v_est_mc_lr, &mut v_est_mc_opt, &mut MseLoss::new(), &dev);

        //         info!("training v_estimator with td");
        //         v_estimator = v_estimator.td_train(&history, &rs_estimator, &a_selector, alpha, 20, 10, 5, v_est_td_lr, &mut v_est_td_opt, &mut MseLoss::new(), &dev);

        //         info!("training a_selector");
        //         a_selector = a_selector.train(&history, &rs_estimator, 100, &v_estimator, &mut a_sel_opt, &mut MseLoss::new(), alpha, a_sel_lr, &dev);
        //     }
        // } 
    } 

    '_MAIN_TRAIN_LOOP: loop {          
        let mut histories = Vec::new();
        info!("Collecting episodes...");

        for _ in 0..10{
            let history = simulation.run_episode({
                let mut steps = 0;
                let dev = &dev;
                let rng = &mut rng;
                let a_selector = &a_selector;

                move |s|{ 
                    let a = if steps > 10 { // 1 second
                        a_selector.select_action(&s, dev)
                    } else {
                        GameAction::random(rng)
                    };
                    
                    steps += 1;

                    a
                }
            })
            .await;
            histories.push(history)
        }

        for history in histories{
            info!("training rs_estimator ");
            rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut MseLoss::new(), &dev);

            info!("training v_estimator with monte carlo");
            v_estimator = v_estimator.monte_carlo_train(&history, alpha, v_est_mc_lr, &mut v_est_mc_opt, &mut MseLoss::new(), &dev);

            info!("training v_estimator with td");
            v_estimator = v_estimator.td_train(&history, &rs_estimator, &a_selector, alpha, 20, 10, 5, v_est_td_lr, &mut v_est_td_opt, &mut MseLoss::new(), &dev);

            info!("training a_selector");
            a_selector = a_selector.train(&history, &rs_estimator, 100, &v_estimator, &mut a_sel_opt, &mut MseLoss::new(), alpha, a_sel_lr, &dev);
        }
        info!("saving models...");
        rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
        v_estimator.clone().save_file(v_estimator_model_path.clone(), &recorder).unwrap();
        a_selector.clone().save_file(a_selector_model_path.clone(), &recorder).unwrap();
        
    }
}

