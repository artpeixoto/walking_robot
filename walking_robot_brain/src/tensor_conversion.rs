use std::sync::LazyLock;

use burn::{
    module::Module, prelude::Backend, serde::de::value, tensor::Tensor
};
use itertools::Itertools;
use nalgebra::{Quaternion, Transform, Vector2, Vector3};

use crate::{
    tools::UsedInTrait,
    types::{
        action::{BipedalLimbsActivation, GameAction, LimbActivation},
        state::{
            AccelerometerReading, BipedalLimbsReading, Force, GameState, LimbReading, LinkReading, MotorReading, SensorsReading, TransformReading
        },
    },
};
pub const FORCES_COUNT:usize = 20;

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
        let mut values = Vec::with_capacity(Self::VALUES_COUNT);
        values.extend(self.iterate_values().take(Self::VALUES_COUNT)); //weird behaviour 
        Tensor::<B, 1>::from_floats(values.as_slice(), dev) 
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
impl TensorConvertible for Quaternion<f32>{
    const VALUES_COUNT: usize = 4;

    fn from_values(values: &[f32]) -> Self {
        Quaternion::new(values[0], values[1],values[2], values[3])
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [
            self.w,
            self.i,
            self.j,
            self.k
        ]
        .into_iter()
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

impl TensorConvertible for SensorsReading{
    const VALUES_COUNT: usize = 
    	Vector3::VALUES_COUNT + 
        1 +
        AccelerometerReading::VALUES_COUNT +
        FORCES_COUNT * Force::VALUES_COUNT

        ;

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        SensorsReading{
            target_pos      : Vector3::take_from_iter(&mut values_iter),
            floor_distance  : values_iter.next().unwrap(),
            acc_reading     : AccelerometerReading::take_from_iter(&mut values_iter),
            forces          : {
                let default = Force::default();
                let mut forces = Vec::new();
                for _ in 0..(FORCES_COUNT){
                    let force = Force::take_from_iter(&mut values_iter);
                    if force != default{
                        forces.push(force);
                    }
                }
                forces
            }
        }
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        let forces_iterator = 
            std::iter::repeat_n(
                0f32, 
                (FORCES_COUNT - self.forces.len()) * Force::VALUES_COUNT 
            )
            .chain( 
                self.forces.iter()
                .map(|f| f.iterate_values())
                .flatten()
            );

        self.target_pos.iterate_values()
        .chain(self.floor_distance.iterate_values()) 
        .chain(self.acc_reading.iterate_values())
        .chain(forces_iterator)
    }
}
impl TensorConvertible for AccelerometerReading {
    const VALUES_COUNT: usize = 5 * 3;

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        Self{
            up              : Vector3::take_from_iter(&mut values_iter),
            linear_speed    : Vector3::take_from_iter(&mut values_iter),
            linear_acc      : Vector3::take_from_iter(&mut values_iter),
            angular_speed   : Vector3::take_from_iter(&mut values_iter),
            angular_acc     : Vector3::take_from_iter(&mut values_iter),
        }
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.up.iterate_values()
        .chain(self.linear_speed.iterate_values())
        .chain(self.linear_acc.iterate_values())
        .chain(self.angular_speed.iterate_values())
        .chain(self.angular_acc.iterate_values())
    }
}
impl TensorConvertible for Force{
    const VALUES_COUNT: usize = 6;

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        Force{
            pos     : Vector3::take_from_iter(&mut values_iter),
            force   : Vector3::take_from_iter(&mut values_iter),
        }
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.pos.iterate_values().chain(self.force.iterate_values())
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
    const VALUES_COUNT: usize = 3 * LinkReading::VALUES_COUNT + TransformReading::VALUES_COUNT;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.shoulder.iterate_values().chain(self.thigh.iterate_values()).chain(self.shin.iterate_values()).chain(self.foot.iterate_values())
    }

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        Self {
            shoulder: LinkReading::take_from_iter(&mut values_iter),
            thigh   : LinkReading::take_from_iter(&mut values_iter), 
            shin: LinkReading::take_from_iter(&mut values_iter),
            foot: TransformReading::take_from_iter(&mut values_iter),
        }
    }
}

impl TensorConvertible for LinkReading{
    const VALUES_COUNT: usize = MotorReading::VALUES_COUNT + TransformReading::VALUES_COUNT;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.motor.iterate_values().chain(self.transform.iterate_values())
    }

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned(); 
        LinkReading{
            motor     : MotorReading::take_from_iter(&mut values_iter),
            transform: TransformReading::take_from_iter(&mut values_iter),
        } 
    }
}

impl TensorConvertible for MotorReading{
    const VALUES_COUNT: usize = 4;

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        [
            self.pos,
            self.speed,
            self.acc,
            self.torque
        ]
        .into_iter()
    }

    fn from_values(values: &[f32]) -> Self {
        MotorReading{
            pos     : values[0],
            speed   : values[1],
            acc     : values[2],
            torque  : values[3],
        } 
    }
}
impl TensorConvertible for TransformReading{
    const VALUES_COUNT: usize = 5 * 3 + 4;

    fn from_values(values: &[f32]) -> Self {
        let mut values_iter = values.iter().cloned();
        TransformReading{
            linear_pos      : Vector3::take_from_iter(&mut values_iter),
            linear_speed    : Vector3::take_from_iter(&mut values_iter), 
            linear_acc      : Vector3::take_from_iter(&mut values_iter), 

            angular_pos     : Quaternion::take_from_iter(&mut values_iter),
            angular_speed   : Vector3::take_from_iter(&mut values_iter), 
            angular_acc     : Vector3::take_from_iter(&mut values_iter),  
        }
    }

    fn iterate_values<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.linear_pos             .iterate_values()
        .chain(self.linear_speed    .iterate_values())
        .chain(self.linear_acc      .iterate_values())
        .chain(self.angular_pos     .iterate_values())
        .chain(self.angular_speed   .iterate_values())
        .chain(self.angular_acc     .iterate_values())
    }
}

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
