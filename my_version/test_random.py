import gym

from gym_connect_four import RandomPlayer, ConnectFourEnv
import numpy as np
import pandas as pd
from NeuralModel import *
import torch
from Agent import *
import random

env: ConnectFourEnv = gym.make("ConnectFour-v0")
model_file = "model-temp.pth"
compare_model = "model.pth"
model3_file = "model3.pth"
model3_old_file = "model3-temp.pth"
model2_file = "model2.pth"
model2_old_file = "model2-old.pth"
model4_file = "model4.pth"

def play_game(player1, player2, show = False):
	result, state_hist, act_hist, player_turn = env.run(player1,player2, render=show)
	reward = result.value
	if reward == 0:
		reward = 0.5
	return reward, state_hist, act_hist, player_turn



lr = 1.0
gamma = 0.67

#Q is Q table and r is rewards for all moves we've made
def Q_table(actions, reward):
	q = np.zeros(len(actions))
	r = np.zeros(len(actions))
	if abs(reward) == 0:
		r[-1] = 0.5
		r[-2] = 0.5
	else:
		r[-1] = abs(reward)
		r[-2] = -abs(reward)
	Q(q,r, len(q))
	Q(q,r, len(q)-1)
	table = np.zeros((len(actions), 7))
	for i, a in enumerate(actions):
		table[i][a] = q[i]
	return table

def Q(q, r, i):
	if i < 1:
		return 0
	else:
		q[-i] = q[-i] + lr*(r[-i] + gamma*Q(q, r, i-2) + q[-i])
		return q[-i]

def state_data(states):
	data = []
	states.pop()
	for state in states:
		data.append(state.flatten())

	data = np.array(data)
	return data



def action_data(actions):
	action_data = np.zeros((len(actions), 7))
	for i, action in enumerate(actions):
		action_data[i][action] = 1
	return action_data

def state_reward_model(player1, player2, show = False):
	#SmartPlayer2(env, model = model, show = False, epsilon = 0.7, smart_search = False)
	#FilterPlayer(env, show = False)
	reward, state_hist, _, player_turn = play_game(player1, player2, show)
	if reward == 0:
		reward = 0.5
	state_array = []
	state_hist.popleft()
	n = [(i-i%2)/2 for i in range(len(state_hist))]
	n.reverse()
	q = np.zeros(len(state_hist))
	gamma = 0.6
	for i, state in enumerate(state_hist):
		state_array.append(state.flatten())
		q[i] = gamma**n[i]*reward
	state_array = np.array(state_array)
	return state_array, q, player1

def play_n_games(n):
	all_states = []
	q = []
	win_rate = []
	for i in range(n):
		reward, state_hist, act_hist, player_turn = play_game(player1, player2)
		I_turn = np.diag(player_turn)
		for s in state_hist:
			print(s)
		game_states = state_data(state_hist)
		game_states = I_turn @ game_states
		game_q_table = Q_table(act_hist, reward)
		if reward != 1 and reward != -1:
			print(state_hist[-2])
			print(state_hist[-1])
			print(act_hist[-2])
			print(act_hist[-1])
		if reward != 1:
			all_states = all_states + game_states.tolist()
			q = q + game_q_table.tolist()
		elif random.random() > -0.1:
			all_states = all_states + game_states.tolist()
			q = q + game_q_table.tolist()
		win_rate.append(reward)
	
	for s in win_rate:
		if s != 1 and s != -1:
			print(s)

	all_states = np.array(all_states)
	print('Wins : ', win_rate.count(1))
	print('Loss : ', win_rate.count(-1))
	print('Draw : ', win_rate.count(0.5))
	print('Draw : ', win_rate.count(0))
	q = np.array(q)
	return all_states, q


#state_reward_model()

#data, q = play_n_games(1)
#print(len(q))
#df = pd.DataFrame(data)
#df.to_csv('data.csv', index = False)

#df_q = pd.DataFrame(q)
#df_q.to_csv('Q_data.csv', index = False)