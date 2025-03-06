use crate::traits::{ToJson, TryFromJson};
use rand::{distr::uniform::{UniformFloat, UniformSampler}, Rng};

#[derive(Clone, PartialEq, Debug, Default)]
pub struct GameAction {
    pub limbs_activation: BipedalLimbsActivation,
}

impl GameAction {
    pub fn random(rng: &mut impl Rng) -> Self {
        let uniform_sampler = UniformFloat::<f32>::new(-1.0, 1.0).unwrap();
        let mut sample = || uniform_sampler.sample(rng);
        GameAction {
            limbs_activation: BipedalLimbsActivation {
                left: LimbActivation {
                    shoulder_activation: sample(),
                    thigh_activation: sample(),
                    shin_activation: sample(),
                },
                right: LimbActivation {
                    shoulder_activation: sample(),
                    thigh_activation: sample(),
                    shin_activation: sample(),
                },
            },
        }
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct BipedalLimbsActivation {
    pub left: LimbActivation,
    pub right: LimbActivation,
}
impl BipedalLimbsActivation {
    pub fn new_full_on() -> Self {
        let full_on = LimbActivation {
            shin_activation: 1.0,
            shoulder_activation: 1.0,
            thigh_activation: 1.0,
        };
        Self {
            left: full_on.clone(),
            right: full_on.clone(),
        }
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct LimbActivation {
    pub shoulder_activation: f32,
    pub thigh_activation: f32,
    pub shin_activation: f32,
}
