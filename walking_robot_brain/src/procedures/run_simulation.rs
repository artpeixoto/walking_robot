use std::time::Duration;

use tokio::time::timeout;
use tracing::{debug, info, trace};

use crate::{comm::SimulationEndpoint, types::{action::GameAction, history::History, state::{GameState, GameUpdate}}};

impl SimulationEndpoint{
	pub async fn run_episode<'this, 'gay_state>(&'this mut self, mut policy: impl FnMut(GameState) -> GameAction + 'this) -> History{
        let mut history = Vec::new();

        let (mut previous_state, _previous_reward) = 'WAIT_FIRST_STATE: loop  {match self.recv_sim_update().await {
            GameUpdate::GameStarted => {continue 'WAIT_FIRST_STATE;},
            GameUpdate::GameStep  {state: previous_state, reward}  => break 'WAIT_FIRST_STATE (previous_state, reward) ,
        }};

	    debug!("picking action");
        let mut previous_action = Default::default();

	    debug!("sending action");
        self.send_action(&previous_action).await;

        'SIMULATION_LOOP: loop{
            debug!("waiting for update");
            let (state, reward) = match timeout(Duration::from_secs_f32(1.0), self.recv_sim_update()).await {
                Ok(GameUpdate::GameStarted) => {
                   break 'SIMULATION_LOOP;
                },
                Ok(GameUpdate::GameStep  {state: previous_state, reward})  => (previous_state, reward) ,
                Err(_) => {
                    self.send_action(&Default::default()).await;
                    continue 'SIMULATION_LOOP
                },
            };

            history.push( (previous_state, previous_action, reward));

            debug!("picking action");
            let action = policy(state.clone());
            trace!("action is: {action:?}");

	        debug!("sending action");
            self.send_action(&action).await;

            previous_action = action;
            previous_state = state;
            // simulation_loop += 1;
      
		}
		history
	}
}