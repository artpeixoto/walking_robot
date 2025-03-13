// use std::{
//     iter, ops::Not, path::PathBuf, str::FromStr
// };
// use burn::{
//     backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, module::Module, nn::loss::LossMod, optim::{momentum::MomentumConfig, SgdConfig}, record::{DefaultFileRecorder, FullPrecisionSettings}};

// use walking_robot_brain::{comm::SimulationConnector, types::policy::{noisy_policy::NoisyPolicy, FnPolicy}};
// use walking_robot_brain::models::{a_selector::ASelectorConfig, rs_estimator::{RsEstimator, RsEstimatorConfig}, v_estimator::VEstimatorConfig};
// use rand::{seq::IndexedRandom, Rng};
// use tracing::{info, warn};
// use walking_robot_brain::types::{action::GameAction, state::GameState};

// fn main() {
//     tokio
//     ::runtime
//     ::Builder
//     ::new_current_thread()
//     .enable_all()
//     .build()
//     .unwrap()
//     .block_on(async_main())
// }

// async fn async_main(){
//     pretty_env_logger::init_timed();
//     warn!("yeah baby");
//     type B = Autodiff<Wgpu<f32, i32>>;
//     let dev = WgpuDevice::DefaultDevice;
//     let recorder =  DefaultFileRecorder::<FullPrecisionSettings>::new();
//     let base_path = PathBuf::from_str("./models/").unwrap();
//     let start_from_beggining = base_path.exists().not();

//     let rs_estimator_model_path = base_path.join("rs_estimator");
//     let mut rs_estimator: RsEstimator<B> = {
//         let mut model = 
//             RsEstimatorConfig{
//                 state_layers_size   : [ 100], 
//                 action_layers_size  : [ 100],
//                 joint_layers_size   : [ 100,100],
//                 logic               : [ 100],
//                 cut_through         : [ 100] ,
//                 end                 : [ 100] , 
//                 // [8000; 3], 
//                 // [8000; 2], 
//                 // 100, 
//                 // [8000; 2] 
//             }
//             .init(&dev);

//         if !start_from_beggining{
//             model = model.load_file(&rs_estimator_model_path, &recorder, &dev).unwrap();
//         } 
//         model
//     };
//     let v_estimator_model_path = base_path.join("v_estimator");
//     let mut v_estimator = {
//         let mut model = 
//             VEstimatorConfig::new(
//                 [ 100, 100], 
//                 [ 100], 
//                 [ 100], 
//                 [ 100] 
//             )
//             .init(&dev);

//         if !start_from_beggining{
//             model = model.load_file(&v_estimator_model_path, &recorder, &dev).unwrap();
//         } 
//         model
//     };

//     let a_selector_model_path = base_path.join("a_selector");
//     let mut a_selector = {
//         let mut model = 
//             ASelectorConfig::new(
//                 [100, 100, 100, 100], 
//                 [100, 100, 100], 
//                 [100, 100],
//                 [100, 100, 100, 100] 
//             )
//             .init(&dev);
//         if !start_from_beggining{
//             model = model.load_file(&a_selector_model_path, &recorder, &dev).unwrap();
//         } 
//         model
//     };
//     let rs_est_lr = 1e-5;
//     let v_est_mc_lr = 1e-4;
//     let v_est_td_lr = 1e-4;
//     let a_sel_lr = 1e-3;
//     let opt_config = 
//         SgdConfig::new()
//         // .with_gradient_clipping(Some(GradientClippingConfig::Value(3.1)))
//         .with_momentum(Some(MomentumConfig::new().with_momentum(0.1)))
//         ;

//     // let mut rs_est_opt = opt_config.clone().init();
//     let mut v_est_mc_opt = opt_config.clone().init();
//     let mut v_est_td_opt = opt_config.clone().init();
//     let mut a_sel_opt = opt_config.clone().init();

//     let alpha = 0.95;


//     info!("waiting for connection, baby");
//     let mut rng = rand::rng();
//     let mut simulation = SimulationConnector::new().connect().await;
//     if start_from_beggining{ 
//         info!("No models were found. The learning will start from the beggining"); 

//         info!("Training v_estimator and rs_estimator"); 
//         { // first we train the v_estimator to be very optimistic and teach the rs_estimator about the physics

//             for _ in 0..100{
//             let mut histories = Vec::new();
//                 for _i in 0..3{
//                     let policy = |_state: &GameState| {
//                         GameAction::random(&mut rng)
//                     };
//                     histories.push(simulation.run_episode(&mut FnPolicy(policy)).await);
//                 }

//                 for history in iter::from_fn(||histories.choose(&mut rng)).take(1000){
//                     // rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut LossMod::new(), &dev);           
//                 }            
//             }
//         }


//         // warn!("Training using sa_tree_expansion policy"); 
//         // for _ in 0..100 {   // now we must run the simulation picking the best actions for every step
//         //         let mut expander_policy = NoisyPolicy::new(TensorFnPolicy(|i: Tensor<B, 2>| Tensor::zeros([i.dims()[0], GameAction::VALUES_COUNT], &dev)), 1.0, rng.clone());

//         //         let mut policy = TreeExpPolicy::new(TreeExpander::new(&rs_estimator, &v_estimator, &mut expander_policy, alpha), 4, 100);

//         //         let history = simulation.run_episode(&mut policy).await;
//         //         rs_estimator = rs_estimator.train(&history, rs_est_lr, &mut rs_est_opt, &mut LossMod::new(), &dev);           

//         //         v_estimator = v_estimator.monte_carlo_train(&history,  alpha, v_est_mc_lr, &mut v_est_mc_opt, &mut LossMod::new(), &dev);

//         //         a_selector  = a_selector.train_from_history(&history, &mut a_sel_opt, &mut LossMod::new(), a_sel_lr, &dev);
//         // }


//         info!("saving models...");
//         rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
//         v_estimator.clone().save_file(v_estimator_model_path.clone(), &recorder).unwrap();
//         a_selector.clone().save_file(a_selector_model_path.clone(), &recorder).unwrap();


//         warn!("Initial training was finished"); 
//     }

//     '_MAIN_TRAIN_LOOP: loop {          
//         let mut histories = Vec::new();

//         for _ in 0..100{
//             let mut policy = NoisyPolicy::new(&a_selector, 0.15, rng.clone());
//             let history = simulation.run_episode(&mut policy).await;
//             histories.push(history)
//         }

//         for history in histories.iter(){
//             // rs_estimator = rs_estimator.train(history, rs_est_lr, &mut rs_est_opt, &mut LossMod::new(), &dev);

//             info!("training v_estimator with monte carlo");
//             v_estimator = v_estimator.monte_carlo_train(history, alpha, v_est_mc_lr, &mut v_est_mc_opt, &mut LossMod::new(), &dev);
//         }

//         for history in histories.iter(){
//             info!("training v_estimator with td");
//             v_estimator = v_estimator.td_train(history, &rs_estimator, &mut NoisyPolicy::new(&a_selector, 0.15, rng.clone()),  30, 5, alpha,v_est_td_lr, &mut v_est_td_opt, &mut LossMod::new(), &dev);

//             info!("training a_selector");
//             a_selector = a_selector.train_from_tree_exp(history, &rs_estimator, &v_estimator, &mut a_sel_opt,  &mut LossMod::new(), 30, 5, 0.25, alpha, a_sel_lr, &dev);
//         }
//         info!("saving models...");
//         rs_estimator.clone().save_file(rs_estimator_model_path.clone(), &recorder).unwrap();
//         v_estimator.clone().save_file(v_estimator_model_path.clone(), &recorder).unwrap();
//         a_selector.clone().save_file(a_selector_model_path.clone(), &recorder).unwrap();
        
//     }
// }
fn main(){}
