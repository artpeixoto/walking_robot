use std::{
    path::PathBuf, str::FromStr
};
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, module::Module, record::{DefaultFileRecorder, FullPrecisionSettings}};

use walking_robot_brain::comm::SimulationConnector;
use walking_robot_brain::models::a_selector::ASelectorConfig;
use tracing::{info, warn};

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
    warn!("testing, baby");
    type B = Autodiff<Wgpu<f32, i32>>;
    let dev = WgpuDevice::DefaultDevice;
    let recorder =  DefaultFileRecorder::<FullPrecisionSettings>::new();

    let base_path = PathBuf::from_str("./models/").unwrap();

    let a_selector_model_path = base_path.join("a_selector");
    let a_selector = {
        let mut model = 
            ASelectorConfig::new(
                [2000, 2000, 2000, 2000], 
                [2000, 2000, 2000], 
                [2000, 2000],
                [2000, 2000, 2000, 2000] 
            )
            .init::<B>(&dev);
        model = model.load_file(&a_selector_model_path, &recorder, &dev).unwrap();
        model
    };

    info!("waiting for connection, baby");
    let rng = rand::rng();
    let mut simulation = SimulationConnector::new().connect().await;
    '_MAIN_LOOP: loop {
        simulation.run_episode(&mut &a_selector).await;
    }
}

