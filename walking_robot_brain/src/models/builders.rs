use std::{
    ops::Deref,
    path::PathBuf,
    str::FromStr,
    sync::{LazyLock, Mutex},
};

use burn::{
    module::Module,
    nn::Gelu,
    prelude::Backend,
    record::{FullPrecisionSettings, PrettyJsonFileRecorder},
};
use rand::TryRngCore;

use crate::{
    modules::sequential::LinearSequentialConfig, tensor_conversion::TensorConvertible,
    types::{action::GameAction, state::GameState},
};

use super::{
    a_selector::{self, ASelector, ASelectorConfig},
    rs_estimator::{RsEstimator, RsEstimatorConfig},
    sa_endec::{SaDecoderConfig, SaEnDec, SaEncoder, SaEncoderConfig},
    v_estimator::{VEstimator, VEstimatorConfig},
};
pub const MODELS_PATH: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::from_str("models/").unwrap());
pub const A_SELECTOR_MODEL_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| MODELS_PATH.join("a_selector.json"));
pub const RS_ESTIMATOR_MODEL_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| MODELS_PATH.join("rs_estimator.json"));
pub const SA_DEC_MODEL_PATH: LazyLock<PathBuf> = LazyLock::new(|| MODELS_PATH.join("sa_dec.json"));
pub const SA_ENC_MODEL_PATH: LazyLock<PathBuf> = LazyLock::new(|| MODELS_PATH.join("sa_enc.json"));
pub const V_ESTIMATOR_MODEL_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| MODELS_PATH.join("v_estimator.json"));

pub static MODELS_RECORDER: LazyLock<Mutex<PrettyJsonFileRecorder<FullPrecisionSettings>>> =
    LazyLock::new(|| Mutex::new(PrettyJsonFileRecorder::new()));
pub const WINDOW_SIZE: i64 = 50;
pub const ENC_STATE_SIZE: usize = 128;
pub fn make_a_selector<B: Backend>(dev: &<B as Backend>::Device) -> ASelector<B> {
    let mut model = ASelectorConfig {
        linear_layers_size: [512, 1024, 2048, 1024],
        logic_layers_size: [256, 512, 512],
        cut_through_layers_size: [1024, 1024],
        end: [1024; 4],
    }
    .init(dev);
    if A_SELECTOR_MODEL_PATH.exists() {
        model = model
            .clone()
            .load_file(
                &*A_SELECTOR_MODEL_PATH,
                MODELS_RECORDER.lock().unwrap().deref(),
                dev,
            )
            .unwrap();
    }
    model
}

pub fn make_rs_estimator<B: Backend>(dev: &<B as Backend>::Device) -> RsEstimator<B> {
    let mut model = RsEstimatorConfig {
        state_layers_size: [1024*4],
        action_layers_size: [128],
        joint_layers_size: [1024*4, 2048],
        logic: [1024],
        cut_through: [1024],
        end: [2048, 1024, 512, 256],
    }
    .init(dev);

    if RS_ESTIMATOR_MODEL_PATH.exists() {
        model = model
            .clone()
            .load_file(
                &*RS_ESTIMATOR_MODEL_PATH,
                MODELS_RECORDER.lock().unwrap().deref(),
                dev,
            )
            .unwrap();
    }
    model
}

pub fn make_v_estimator<B: Backend>(dev: &<B as Backend>::Device) -> VEstimator<B> {
    let mut model = VEstimatorConfig {
        initial: [512, 1024],
        logic: [512],
        cut_through: [1024],
        end: [2048],
    }
    .init(dev);

    if V_ESTIMATOR_MODEL_PATH.exists() {
        model = model
            .clone()
            .load_file(
                &*V_ESTIMATOR_MODEL_PATH,
                MODELS_RECORDER.lock().unwrap().deref(),
                dev,
            )
            .unwrap();
    }
    model
}

pub fn make_sa_endec<B: Backend>(dev: &<B as Backend>::Device) -> SaEnDec<B> {
    let encoder = {
        let mut model = 
            SaEncoderConfig{
                input_linear: vec![ 
                    GameState::VALUES_COUNT + GameAction::VALUES_COUNT ,
                    256,
                ],
                recurrent: vec![
                    512, 
                    4096,
                    2048 * 4,
                    4096,
                    1024,
                ],
                output_linear: vec![
                    512
                ],
                final_output: ENC_STATE_SIZE as usize,
            }
            .init(dev); 
        if SA_ENC_MODEL_PATH.exists(){
            model = model.load_file(SA_ENC_MODEL_PATH.as_path(), MODELS_RECORDER.lock().unwrap().deref(), dev).unwrap();
        }
        model
    };

    let decoder = {
        let mut model = 
            SaDecoderConfig{
                linears: vec![
                    ENC_STATE_SIZE,
                    1024,
                    2048,
                    4096, 
                    4096, 
                    2048,
                    5 * (GameState::VALUES_COUNT + GameAction::VALUES_COUNT)
                ]
            }
            .init(dev);

        if SA_DEC_MODEL_PATH.exists(){
            model = model.load_file(SA_DEC_MODEL_PATH.as_path(),  MODELS_RECORDER.lock().unwrap().deref(), dev).unwrap();
        }
        model
    };
    SaEnDec { enc: encoder, dec: decoder }
}
