use super::{action::GameAction, state::{GameState, Reward}};

pub type History = Vec<HistoryStep>;
pub type HistoryStep = (GameState, GameAction, Reward);