use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::{HuberLossConfig, MseLoss},
    optim::{AdamConfig, SgdConfig},
    prelude::Backend,
    record::{CompactRecorder, DefaultFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
};
use rand::seq::IndexedRandom;
use std::{iter, ops::Not, path::PathBuf, str::FromStr, sync::Mutex};
use tracing::{info, warn};
use walking_robot_brain::{
    comm::SimulationConnector,
    models::{
        builders::{make_q_estimator, Q_ESTIMATOR_MODEL_PATH},
        q_estimator::{self, QEstimator},
    },
    types::{history::TensorHistory, policy::q_estimator_policy::QEstimatorPolicy},
};

fn main() {
    tokio::runtime
        ::Builder
        ::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async_main())
}

async fn async_main() {
    pretty_env_logger::init_timed();

    warn!("yeah baby");

    type B = Autodiff<Wgpu<f32, i32>>;
    <B as Backend>::seed(420);
    let dev = WgpuDevice::DefaultDevice;
    let recorder = CompactRecorder::new();
    let mut training_q_estimator = make_q_estimator::<B>(&dev);
    let mut running_q_estimator = training_q_estimator.clone();

    let lr = 0.0001;
    let opt_config = AdamConfig::new();
    let mut optim = opt_config.clone().init();
    let mut loss_mod = MseLoss::new();
    let alpha = 0.99;

    info!("waiting for connection, baby");

    let mut rng = rand::rng();
    let mut simulation = SimulationConnector::new().connect().await;

    loop {
        for _ in 0..10 {
            let mut policy = QEstimatorPolicy::new(&running_q_estimator, 100, &dev);
            let mut histories = Vec::new();
            for _ in 0..4 {
                histories.push(
                    simulation
                    .run_episode(&mut policy)
                    .await
                );
            }
            let histories_len = histories.len();
            for history in iter::from_fn(|| histories.choose(&mut rng)).take(histories_len * 8) {
                training_q_estimator =
                    training_q_estimator.train_monte_carlo(
                        &history.to_tensor_history(&dev), 
                        alpha, 
                        lr, 
                        &mut optim, 
                        &mut loss_mod, 
                        &dev
                    );
            }
        }
        info!("saving estimator");
        training_q_estimator.clone().save_file(Q_ESTIMATOR_MODEL_PATH.as_path(), &recorder).unwrap();

        info!("Setting the running_q_estimator to be like the training");
        running_q_estimator = training_q_estimator.clone().no_grad();
    }
}
