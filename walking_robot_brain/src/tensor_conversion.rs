use std::{array, iter, mem::take, sync::LazyLock};

use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::{HuberLoss, HuberLossConfig, MseLoss},
        Gelu, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, Sigmoid, Tanh,
    },
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    record::Record,
    tensor::{backend::AutodiffBackend, Distribution::Uniform, DistributionSampler, Float, Tensor},
    train::metric::{LossInput, LossMetric},
};
use itertools::Itertools;
use nalgebra::{Vector2, Vector3};
use rand::{
    distr::{
        uniform::{UniformFloat, UniformSampler},
        weighted::WeightedIndex,
        Distribution, StandardUniform,
    },
    rngs::ThreadRng,
    Rng,
};
use tracing::{debug, info};

use crate::{
    tools::UsedInTrait,
    types::{
        action::{BipedalLimbsActivation, GameAction, LimbActivation},
        state::{
            BipedalLimbsReading, GameState, LimbJointsForces, LimbJointsPositions, LimbReading,
            Reward, SensorsReading,
        },
    },
};
#[derive(Debug, Clone, Copy, Default)]
pub struct ValuesHaveWrongSize;

#[derive(Debug, Clone)]
pub struct TakeTensorPartRes<B: Backend, const D: usize> {
    pub value: Tensor<B, D>,
    pub rest: Tensor<B, D>,
}

pub trait TensorConvertible: Sized {
    const VALUES_COUNT: usize;
    fn from_values(values: &[f32]) -> Self;
    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a;

    fn take_from_iter(iter: &mut impl Iterator<Item = f32>) -> Self {
        let vals = iter.take(Self::VALUES_COUNT).collect_vec();
        if vals.len() == Self::VALUES_COUNT {
            Self::from_values(&vals)
        } else {
            panic!()
        }
    }

    fn try_from_values(values: &[f32]) -> Result<Self, ValuesHaveWrongSize> {
        if values.len() != Self::VALUES_COUNT {
            Err(ValuesHaveWrongSize)
        } else {
            Ok(Self::from_values(values))
        }
    }
    fn to_tensor<B: Backend>(&self, dev: &<B as Backend>::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(self.iterate_values().collect_vec().as_slice(), dev)
    }

    fn from_tensor<B: Backend>(tensor: Tensor<B, 1>) -> Self {
        let tensor_data = tensor.to_data();
        Self::from_values(tensor_data.as_slice().unwrap())
    }

    fn take_tensor_part<B: Backend, const D: usize>(
        tensor: Tensor<B, D>,
        dim: usize,
    ) -> TakeTensorPartRes<B, D> {
        let mut take_self_ranges = [None; D];
        take_self_ranges[dim] = Some((0_i64, Self::VALUES_COUNT as i64));
        let self_tensor = tensor.clone().slice(take_self_ranges);

        let rest_len = tensor.dims()[dim];
        let mut rest_ranges = [None; D];
        rest_ranges[dim] = Some((Self::VALUES_COUNT as i64, rest_len as i64));
        let rest_tensor = tensor.slice(rest_ranges);

        TakeTensorPartRes {
            value: self_tensor,
            rest: rest_tensor,
        }
    }
    fn many_from_tensor<B: Backend>(tensor: Tensor<B, 2>) -> Vec<Self> {
        tensor
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .chunks(Self::VALUES_COUNT)
            .into_iter()
            .map(|c| Self::from_values(&c.collect_vec()))
            .collect_vec()
    }
}
impl TensorConvertible for f32{
    const VALUES_COUNT: usize = 1;

    fn from_values(values: &[f32]) -> Self {
        values[0]
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        std::iter::once(*self)
    }
}

impl TensorConvertible for GameAction {
    const VALUES_COUNT: usize = 6;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        let left = &self.limbs_activation.left;
        let right = &self.limbs_activation.right;
        [
            left.shoulder_activation,
            left.thigh_activation,
            left.shin_activation,
            right.shoulder_activation,
            right.thigh_activation,
            right.shin_activation,
        ]
        .into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        Self {
            limbs_activation: BipedalLimbsActivation {
                left: LimbActivation {
                    shoulder_activation: values[0],
                    thigh_activation: values[1],
                    shin_activation: values[2],
                },
                right: LimbActivation {
                    shoulder_activation: values[3],
                    thigh_activation: values[4],
                    shin_activation: values[5],
                },
            },
        }
    }
}

pub trait TensorConvertibleIterExts {
    fn many_to_tensor<B: Backend>(self, dev: &<B as Backend>::Device) -> Tensor<B, 2>;
}
impl<'a, Item, Iter> TensorConvertibleIterExts for Iter
where
    Item: TensorConvertible + 'a,
    Iter: Iterator<Item = &'a Item>,
{
    fn many_to_tensor<B: Backend>(self, dev: &<B as Backend>::Device) -> Tensor<B, 2> {
        self.map(|i| i.to_tensor(dev).unsqueeze_dim(0))
            .collect_vec()
            .used_in(|els| Tensor::cat(els, 0))
    }
}

impl TensorConvertible for GameState {
    const VALUES_COUNT: usize = SensorsReading::VALUES_COUNT + BipedalLimbsReading::VALUES_COUNT;
    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.sensors_reading
            .iterate_values()
            .chain(self.limbs_readings.iterate_values())
    }

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        Self {
            sensors_reading: SensorsReading::take_from_iter(&mut values_iter),
            limbs_readings: BipedalLimbsReading::take_from_iter(&mut values_iter),
        }
    }
}
impl TensorConvertible for BipedalLimbsReading {
    const VALUES_COUNT: usize = 2 * <LimbReading as TensorConvertible>::VALUES_COUNT;

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        Self {
            left: LimbReading::take_from_iter(&mut values_iter),
            right: LimbReading::take_from_iter(&mut values_iter),
        }
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.left
            .iterate_values()
            .chain(self.right.iterate_values())
    }
}
impl TensorConvertible for LimbReading {
    const VALUES_COUNT: usize = 1
        + LimbJointsPositions::VALUES_COUNT
        + LimbJointsForces::VALUES_COUNT
        + Vector3::VALUES_COUNT;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [if self.is_foot_touching_floor {
            1.0
        } else {
            0.0
        }]
        .into_iter()
        .chain(self.positions.iterate_values())
        .chain(self.forces.iterate_values())
        .chain(self.force_applied_by_floor.iterate_values())
    }

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().map(|x| *x);

        Self {
            is_foot_touching_floor: values_iter.next().unwrap() > 0.5,
            positions: TensorConvertible::take_from_iter(&mut values_iter),
            forces: TensorConvertible::take_from_iter(&mut values_iter),
            force_applied_by_floor: TensorConvertible::take_from_iter(&mut values_iter),
        }
    }
}

impl TensorConvertible for LimbJointsForces {
    const VALUES_COUNT: usize = 3;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [self.shoulder, self.thigh, self.shin].into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        Self {
            shoulder: values[0],
            thigh: values[1],
            shin: values[2],
        }
    }
}

impl TensorConvertible for LimbJointsPositions {
    const VALUES_COUNT: usize = 3;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [self.shoulder, self.thigh, self.shin].into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        Self {
            shoulder: values[0],
            thigh: values[1],
            shin: values[2],
        }
    }
}
impl TensorConvertible for SensorsReading {
    const VALUES_COUNT: usize = 16;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [
            self.floor_distance,
            self.target_distance,
            self.up_orientation[0],
            self.up_orientation[1],
            self.up_orientation[2],
            self.speed[0],
            self.speed[1],
            self.speed[2],
            self.target_direction[0],
            self.target_direction[1],
            self.linear_acceleration[0],
            self.linear_acceleration[0],
            self.linear_acceleration[0],
            self.angular_acceleration[0],
            self.angular_acceleration[0],
            self.angular_acceleration[0],
        ]
        .into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        Self {
            floor_distance: values_iter.next().unwrap(),
            target_distance: values_iter.next().unwrap(),
            up_orientation: Vector3::take_from_iter(&mut values_iter),
            speed: Vector3::take_from_iter(&mut values_iter),
            target_direction: Vector2::take_from_iter(&mut values_iter),
            linear_acceleration: Vector3::take_from_iter(&mut values_iter),
            angular_acceleration: Vector3::take_from_iter(&mut values_iter),
        }
    }
}
impl SensorsReading {}

impl TensorConvertible for Vector2<f32> {
    const VALUES_COUNT: usize = 2;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [self.x, self.y].into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        Vector2::new(values[0], values[1])
    }
}
impl TensorConvertible for Vector3<f32> {
    const VALUES_COUNT: usize = 3;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [self.x, self.y, self.z].into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        Vector3::new(values[0], values[1], values[2])
    }
}
