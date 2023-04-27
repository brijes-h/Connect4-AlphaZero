import torch
from game import get_init_board, place_piece, get_valid_moves, is_board_full, is_win
from view_board import render
import numpy as np

board = np.array(
        [[ 0,-1,-1,-1, 1, 0,-1],
         [ 0, 1,-1, 1, 1, 0, 1],
         [-1, 1,-1, 1, 1, 0,-1],
         [ 1,-1, 1,-1,-1, 0,-1],
         [-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1,-1, 1,-1, 1]] 
         )

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

	# We try and expand the root node during the MCTS
	def expand(self, action_probs): 
		# [0.5, 0, 0, 0, 0, 0.5, 0] -> these are the output values
		#  (columns that are full have 0s)
		for action, prob in enumerate(action_probs):
			if prob > 0:
				next_state = place_piece(board=self.state, player=self.turn, action=action)
				self.children[action] = Node(prior=prob, turn=self.turn*-1, state=next_state)

# initialize root
root = Node(prior=0, turn=1, state=board)

# expand the root
value, action_probs = dummy_model_predict(root.state)
root.expand(action_probs=action_probs)

# run simulations 
