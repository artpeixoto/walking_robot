use std::time::Duration;

use tokio::time::timeout;
use tracing::{debug, info, trace};

use crate::{comm::SimulationEndpoint, types::{history::History, policy::Policy, state::GameUpdate}};

impl SimulationEndpoint{
	pub async fn run_episode<'this, 'state>(&'this mut self,  policy: &mut impl Policy) -> History{
        info!("Running episode");
        let mut history = History::default();

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

            history.states.push( previous_state);
            history.actions.push( previous_action);
            history.rewards.push( reward);

            debug!("picking action");
            let action = policy.select_action(&state);
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