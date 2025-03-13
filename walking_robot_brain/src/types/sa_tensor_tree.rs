use std::{collections::{BTreeMap, BTreeSet}, iter::zip};

use burn::prelude::Backend;
use either::Either::{self, Left, Right};
use rand::{rng, RngCore};

use crate::tensor_conversion::TensorConvertible;

use super::tensor_types::{GameActionTensor, GameActionsTensor, GameRewardTensors, GameStateTensor, GameStatesTensor};



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
pub struct StateActionNodeView<'t, B: Backend> 
where 
{
	id: Id,
	value: Either<&'t GameStateTensor<B>, &'t StateActionNode<B>>,
	tree: &'t SaTensorTree<B>
}

impl<'t, B: Backend> StateActionNodeView<'t, B>{
	pub fn is_root(&self) -> bool{
		self.id.is_root()
	}
	pub fn id(&self) -> Id{
		self.id.clone()
	}
	pub fn state(&self) -> &'t GameStateTensor<B>{
		self.tree.get_state(self.id)
	}
	pub fn node(&self) -> Option<&'t StateActionNode<B>>{
		Some(self.tree.get_node(self.id.get_node_id()?))
	}
	pub fn children(&self) -> Option<impl Iterator<Item = StateActionNodeView<'t, B>> + Sized>{
		Some(self.tree.children.get(&self.id)?.iter().map(|c| self.tree.get(*c)))
	}
	pub fn parent(&self) -> Option<StateActionNodeView<'t, B>>{
		let node_id  = self.id.get_node_id()?;	
		let parent_id = self.tree.parents[&node_id];
		Some(self.tree.get(parent_id))
	}

	pub fn acc_reward(&self) -> f32{
		match self.value{
			Left(_) => 0.0,
			Right(node) => node.acc_reward,
		}
	}
	pub fn depth(&self) -> i32{
		match self.value{
			Left(_) => 0,
			Right(node) => node.depth,
		}
	}

}

pub struct SaTensorTree<B: Backend>
{
	nodes         	: BTreeMap<u64, StateActionNode<B>>,
	parents 		: BTreeMap<u64, Id>,
	children 		: BTreeMap<Id, BTreeSet<Id>>,
	root			: GameStateTensor<B>,
	alpha			: f32,
	dev				: <B as Backend>::Device
}

impl<B: Backend> SaTensorTree<B>
{
	pub fn new( root: GameStateTensor<B>, alpha: f32, dev: &<B as Backend>::Device) -> Self {
		Self { 
			nodes		: BTreeMap::default(),
			parents		: BTreeMap::default(),
			children	: BTreeMap::new(),
			root		: root, 
			alpha,
			dev			: dev.clone()
		}
	}
	pub fn get<'t>(&'t self, id: Id) -> StateActionNodeView<'t, B>{
		let value = match id{
			Id::Root => Left(&self.root),
			Id::Node(node_id) => Right(self.get_node(node_id)),
		};
		StateActionNodeView{id, value: value, tree: self}
	}

	pub fn get_state(&self, id: Id) -> &GameStateTensor<B>{
		match id{
			Id::Root 					=> &self.root,
			Id::Node(not_root_id) 	=> &self.nodes.get(&not_root_id).unwrap().state,
		}
	}

	pub fn get_node(&self, node_id: NodeId) -> &StateActionNode<B>{
		self.nodes.get(&node_id).unwrap()
	}
	pub fn get_children(&self, node_id: Id) -> Option<&BTreeSet<Id>>{
		self.children.get(&node_id)
	}

	pub fn add_children(&mut self, parent_id: Id, (actions, acc_rewards, states): (GameActionsTensor<B>, GameRewardTensors<B>, GameStatesTensor<B>)) -> Vec<Id>{
		let mut rng = rng();
		let parent = self.get(parent_id);	
		let parent_depth = parent.depth();

		let children_set = 
			match self.children.get_mut(&parent_id){
				Some(c) => c,
				None => {
					self.children.insert(parent_id, Default::default());
					self.children.get_mut(&parent_id).unwrap()
				},
			};

		let rewards = f32::many_from_tensor(acc_rewards.unsqueeze_dim(1));
		let mut ids = Vec::with_capacity(rewards.len());

		for (a, (r, s)) in zip(actions.iter_dim(0) , zip(rewards.into_iter(), states.iter_dim(0))) {
			let a = a.squeeze(0);
			let s = s.squeeze(0);

			let child_node_id = rng.next_u64();
			let child_id = Id::Node(child_node_id);
			let new_node = StateActionNode{
				action: a,
				state: s,
				acc_reward: r,
				depth: parent_depth + 1,
			};
			self.nodes.insert(child_node_id, new_node); 
			self.parents.insert(child_node_id, parent_id);
			children_set.insert(child_id);
			ids.push(child_id);
		}
		ids
	}
}


#[derive( Debug)]
pub struct StateActionNode<B: Backend>{
	pub state : GameStateTensor<B>,
	pub action: GameActionTensor<B>,
	pub acc_reward: f32,
	pub depth : i32 
}
