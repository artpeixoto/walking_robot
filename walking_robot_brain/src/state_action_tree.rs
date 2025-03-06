use std::{any::Any, collections::{BTreeMap, BTreeSet}, iter::zip, ops::Not, sync::{Arc, RwLock, Weak}};
use burn::{prelude::Backend, tensor::{cast::ToElement, Tensor}};
use either::Either::{self, Left, Right};
use fix_float::ff32;
use itertools::Itertools;
use rand::{rng, Rng, RngCore};
use sorted_list::SortedList;
use tracing::trace;
use crate::{models::{a_selector::ASelector, rs_estimator::RsEstimator}, tensor_conversion::TensorConvertible, types::{action::GameAction, state::{GameState, Reward, G}}};

pub struct RandomActionsStateExpander<'a, B: Backend>{
	rs_estimator	: &'a RsEstimator<B>,
	a_selector		: &'a ASelector<B>,
	actions_count	: usize,
	dev				: &'a <B as Backend>::Device,
}

impl<'a, B: Backend> RandomActionsStateExpander<'a, B> {
	pub fn new(rs_estimator: &'a RsEstimator<B>, a_selector: &'a ASelector<B>, actions_count: usize, dev: &'a <B as Backend>::Device) -> Self {
		Self { rs_estimator, a_selector, actions_count, dev }
	}

}
pub struct ExpandTensorResult<B: Backend>{
	pub actions_tensor: Tensor<B,2>,
	pub rewards_tensor: Tensor<B,1>,
	pub next_states_tensor: Tensor<B, 2>
}


impl<'a, B: Backend> StateExpander<B> for RandomActionsStateExpander<'a, B>{
	fn expand(&mut self, game_state: &GameState) -> Vec<(GameAction, Reward, GameState)>{
		let ExpandTensorResult { actions_tensor, rewards_tensor, next_states_tensor } = self.expand_tensor(&game_state.to_tensor(self.dev));

		let actions_iter 		= GameAction::many_from_tensor(actions_tensor).into_iter();
		let rewards_iter 		= f32::many_from_tensor(rewards_tensor.unsqueeze()).into_iter();
		let next_states_iter 	= GameState::many_from_tensor(next_states_tensor).into_iter();

		zip(actions_iter, zip(rewards_iter, next_states_iter)).map(|(a, (r, s))| (a, r, s)).collect_vec()
	}
	fn expand_tensor(&mut self, game_state_tensor: &Tensor<B,1>) -> ExpandTensorResult<B>{
		trace!("expanding tensor");
		let base_action_tensor = self.a_selector.forward(&game_state_tensor.clone().unsqueeze()).repeat_dim(0, self.actions_count);
		let action_variations_tensor = Tensor::random([self.actions_count, GameAction::VALUES_COUNT], burn::tensor::Distribution::Uniform(-0.6f64, 0.6f64), self.dev);
		let actions_tensor = base_action_tensor + action_variations_tensor.clamp(-1.0, 1.0);
		let states_tensor = game_state_tensor.clone().unsqueeze_dim(0).repeat_dim( 0, self.actions_count,);
		let (rewards_tensor, next_states_tensor) = self.rs_estimator.forward(&states_tensor, &actions_tensor);

		ExpandTensorResult{
			actions_tensor,
			rewards_tensor,
			next_states_tensor
		}
	}
}

pub trait StateExpander<B: Backend>{
	
	fn expand(&mut self, game_state: &GameState) -> Vec<(GameAction, Reward, GameState)>;
	fn expand_tensor(&mut self, game_state_tensor: &Tensor<B, 1>) -> ExpandTensorResult<B>;
}


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Id{
	Root,
	Node(NodeId)
}
pub type NodeId = u64;

impl Id{
	pub fn is_root(&self) -> bool{
		matches!(self, Id::Root)
	}

	//this will panic if it is not a node

	#[inline]
	pub fn get_node_id(&self) -> Option<NodeId>{
		match self{
			Id::Root => None,
			Id::Node(inner) => Some(*inner) 
		}
	}
}
#[derive(Clone, )]
pub struct StateActionNodeWalker<'t> 
where 
{
	id: Id,
	tree: &'t StateActionTree
}

impl<'t> StateActionNodeWalker<'t>{
	pub fn is_root(&self) -> bool{
		self.id.is_root()
	}
	pub fn id(&self) -> Id{
		self.id.clone()
	}
	pub fn state(&self) -> &'t GameState{
		self.tree.get_state(self.id)
	}
	pub fn node(&self) -> Option<&'t StateActionNode>{
	 	Some(self.tree.get_node(self.id.get_node_id()?))
	}
	pub fn children(&self) -> Option<impl Iterator<Item = StateActionNodeWalker<'t>> + Sized>{
		Some(self.tree.children.get(&self.id)?.iter().map(|c| self.tree.start_walking(*c)))
	}
	pub fn parent(&self) -> Option<StateActionNodeWalker<'t>>{
		let node_id  = self.id.get_node_id()?;	
		Some(StateActionNodeWalker{
			id: self.tree.parents[&node_id],
			tree: self.tree
		})
	}
}

pub struct StateActionTree 
where 
{
	frontier		: BTreeSet<Id>,
	nodes         	: BTreeMap<u64, StateActionNode>,
	parents 		: BTreeMap<u64, Id>,
	children 		: BTreeMap<Id, BTreeSet<Id>>,
	root			: GameState,
}

impl StateActionTree
{
	pub fn new( root: GameState) -> Self {
		let mut frontier = BTreeSet::new();
		frontier.insert(Id::Root);
		Self { 
			frontier,
			nodes: BTreeMap::default(),
			parents: BTreeMap::default(),
			children: BTreeMap::new(),
			root: root, 
		}
	}
	pub fn start_walking<'t>(&'t self, id: Id) -> StateActionNodeWalker<'t>{
		StateActionNodeWalker{id, tree: self}
	}

	pub fn get_state(&self, id: Id) -> &GameState{
		match id{
			Id::Root 					=> &self.root,
			Id::Node(not_root_id) 	=> &self.nodes.get(&not_root_id).unwrap().state,
		}
	}

	pub fn get(&self, id: Id) -> Either<&GameState, &StateActionNode>{
		match id{
			Id::Root 			=> Left(&self.root),
			Id::Node(node_id) 	=> Right(self.get_node(node_id)),
		}
	}
	pub fn get_node(&self, node_id: NodeId) -> &StateActionNode{
		self.nodes.get(&node_id).unwrap()
	}
	pub fn get_children(&self, node_id: Id) -> Option<&BTreeSet<Id>>{
		self.children.get(&node_id)
	}

	pub fn expand_once<'this, B: Backend>(&'this mut self, id: Id, expander: &mut impl StateExpander<B>) {
		let id = self.frontier.take(&id).unwrap();
		let mut rng = rng();
		let state = self.get_state(id);
		let mut children_set = BTreeSet::new();

		for (a, r, s) in expander.expand(&state){
			let child_node_id = rng.next_u64();
			let child_id = Id::Node(child_node_id);
			let new_node = StateActionNode{
				action: a,
				state: s,
				reward: r
			};
			self.nodes.insert(child_node_id, new_node); 
			self.parents.insert(child_node_id, id);
			self.frontier.insert(child_id);
			children_set.insert(child_id);
		}
		self.children.insert(id, children_set);
	}
}



#[derive( Debug, PartialEq)]
pub struct StateActionNode{
	pub state : GameState,
	pub action: GameAction,
	pub reward: Reward,
}
