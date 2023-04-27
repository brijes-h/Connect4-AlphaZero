import torch
from game import get_init_board, place_piece, get_valid_moves, is_board_full, is_win
from view_board import render
import numpy as np
import random
import math

num_simulations = 100

board = np.array(
        [[ 0,-1,-1,-1, 1, 0,-1],
         [ 0, 1,-1, 1, 1, 0, 1],
         [-1, 1,-1, 1, 1, 0,-1],
         [ 1,-1, 1,-1,-1, 0,-1],
         [-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1,-1, 1,-1, 1]] 
         )

def ucb_score(parent, child): # upper confidence bound
	prior_score = child.prior * math.sqrt(parent.visits) / (child.visits + 1)
	value_score = child.value / child.visits
	return value_score + prior_score
	# this helps to find the balance the exploration-exploitation trade off

# test mcts using dummy model predictions
def dummy_model_predict(board):
	value_head = 0.5
	policy_head = [0.5, 0, 0, 0, 0, 0.5, 0]
	return value_head, policy_head
	# policy_head returns action probabilites
	# value_head tells how strong the board is positioned for a player

class Node:
	def __init__(self, prior, turn, state):
		self.prior = prior
		self.turn = turn
		self.state = state
		self.children = {}
		self.value = 0
		self.visits = 0

	# We try and expand the root node during the MCTS
	def expand(self, action_probs): 
		# [0.5, 0, 0, 0, 0, 0.5, 0] -> these are the output values
		#  (columns that are full have 0s)
		for action, prob in enumerate(action_probs):
			if prob > 0:
				next_state = place_piece(board=self.state, player=self.turn, action=action)
				self.children[action] = Node(prior=prob, turn=self.turn*-1, state=next_state)

	def select_child(self):
		max_score = -99
		for action, child in self.children.items():
			score = ucb_score(self, child)
			if score > max_score:
				selected_action = action
				selected_child = child
				max_score = score

		return selected_action, selected_child


# initialize root
root = Node(prior=0, turn=1, state=board)

# expand the root
value, action_probs = dummy_model_predict(root.state)
root.expand(action_probs=action_probs)

# Iterate through simulations
for _ in range(num_simulations):
	node = root
	serach_path = [node]
	# select next child unitl we reach an unexpanded node
	while len(node.children) > 0:
		action, node = node.select_child()
		serach_path.append(node)

	value = None

	if is_board_full(board=node.state):
		value = 0
	if is_win(board=node.state,player=1):
		value = 1
	if is_win(board=node.state,player=-1):
		value = -1
	
	if value is None:
		value, action_probs = dummy_model_predict(node.state)
		node.expand(action_probs)

	# backing up value
	for node in serach_path:
		node.value += value
		node.visits += 1

# running smaller simulations
print(root.children[0].state) 
print(root.children[0].value)

print(root.children[5].state) 
print(root.children[5].value)