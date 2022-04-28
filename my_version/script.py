import test_random as f
import ModelFunctions as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from numpy import loadtxt
import random
from Agent import *#SmartPlayer, SmartPlayer2, FilterPlayer, SmartPlayer3, ProbabilityPlayer
from NeuralModel import *


def data_n_games(n, player1, player2):
	states = []
	q = []
	win_rate = []
	nr_saved = 0
	size = []
	win_state = []
	for i in range(n):
		game_states, game_q, player1 = f.state_reward_model(player1, player2)
		if len(game_states) == 7:
			if random.random() > 0.0:
				states = states + game_states.tolist()
				q = q + game_q.tolist()
				nr_saved += 1
				size.append(len(game_states))
		elif random.random() > 0.0:
			states = states + game_states.tolist()
			q = q + game_q.tolist()
			nr_saved += 1
			size.append(len(game_states))
		win_state.append(game_states[-1].tolist())
		win_rate.append(game_q[-1])
		#size.append(len(game_states))
	print('Saved games', nr_saved)
	print('Wins : ', win_rate.count(1))
	print('Loss : ', win_rate.count(-1))
	print('Draw : ', win_rate.count(0.5))
	print('Draw : ', win_rate.count(0))
	q = np.array(q)
	states = np.array(states)
	win_state = np.array(win_state)
	df = pd.DataFrame(states)
	df.to_csv('game_states.csv', index = False)
	df_q = pd.DataFrame(q)
	df_q.to_csv('game_q.csv', index = False)
	df_win = pd.DataFrame(win_state)
	df_win.to_csv('win_states.csv', index = False)

def n_games(n, player1, player2, gamma):
	states = []
	q = []
	results = []
	for i in range(n):
		reward, state_hist, _, player_turn = f.play_game(player1, player2)
		state_hist.popleft()
		game = np.array(state_hist)
		p1_states, p1_rewards = player1.get_data(reward, player2.moves[-1], player2.traverse_all(player2.positions[-1]))
		p2_states, p2_rewards = player2.get_data(reward if reward == 0.5 else -reward, player1.moves[-1], player1.traverse_all(player1.positions[-1]))
		states = states + p1_states + p2_states
		q = q + p1_rewards + p2_rewards
		results.append(reward)
		player1.new_game()
		player2.new_game()
	return states, q, results

def count_tuples(arr,elem):
	result = map(lambda x: x[0] == elem, arr)
	return sum(list(result))

def create_q(gamma, reward, player_turn):
	n = len(player_turn)
	p = reward
	k = -1
	q = []
	for i in range(n):
		q.append(p)
		if player_turn[k] == 1:
			p = p*gamma
		k -= 1
	q.reverse()
	return np.array(q)

def reverse_game(game_queue, player_turn, reward, gamma):
	game = np.array(game_queue)
	arr = []
	for i, state in enumerate(game):
		player_turn[i] = -player_turn[i]
		opposing_player_state = (state*player_turn[i]).tolist()
		arr.append(opposing_player_state)
	q = create_q(gamma, reward, player_turn)
	return np.array(arr), q

def datasets(boards, q):
	X = torch.Tensor(np.array([boards]))
	X = X.squeeze().unsqueeze(1)
	Y = torch.Tensor(q)
	Y = Y.unsqueeze(-1)
	return X, Y

def update_players(player1, player2, model, epsilon):
	player1.set_model(model)
	player1.set_epsilon(epsilon)
	player2.set_model(model)
	player2.set_epsilon(epsilon)
	return player1, player2

#here epsilon is the epsilon you want at the end of the training session
def reinforcement_training(players, model, n, epsilon, end_epsilon,epochs=1, load = False, file = "my_model.pth", lr = 0.0005, loss_function = nn.HuberLoss()):
	env = f.env
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9)
	loss_fn = loss_function
	if load:
		checkpoint = torch.load(file)
		model.load_state_dict(checkpoint['model_state'])
		optimizer.load_state_dict(checkpoint['optimizer_state'])
		optimizer.param_groups[0]['lr'] = lr
		print(optimizer)
	player1 = players[0]
	player2 = players[1]
	randomplayer = f.RandomPlayer(env, 'Dexter-Bot2')
	filterplayer = FilterPlayer(env, 'Dexter-Bot2')
	print('lr',optimizer.param_groups[0]['lr'])
	train_loss = []
	val_loss = []
	stats = []
	decay = -np.log(1-0.99)/n
	delta = end_epsilon-epsilon
	gamma = 1.0
	batch_size =32
	NR_ROUNDS = 3
	NR_GAMES = int(n/NR_ROUNDS)
	training_games = []
	training_qs = []
	for j in range(NR_ROUNDS):
		eps = round(epsilon + delta*(1-np.exp(-len(stats)*decay)),3)
		print('epsilon: ', eps)
		player1, player2 = update_players(player1, player2, model, eps)
		boards, q, reward =  n_games(NR_GAMES, player1, player2, gamma)
		stats += reward
		training_games += boards
		training_qs += q
		print(len(training_qs))
		print(len(player1.moves))
		X, Y = datasets(training_games, training_qs)
		#optimizer.param_groups[0]['lr'] = lr/2
		model, optimizer, train_l,val_l, improvement = mf.train_batch(X, Y, model,loss_fn, optimizer, epochs, batch_size, True)
		train_loss += train_l
		val_loss += val_l
		if improvement:
			training_games = []
			training_qs = []
		else:
			all_idx = list(range(len(training_games)))
			keep_idx = random.sample(all_idx, int(len(training_games)/4))
			training_games = [training_games[i] for i in keep_idx]
			training_qs = [training_qs[i] for i in keep_idx]
			epochs += 2
			print("new epochs ", epochs)
	torch.save({'model_state': model.state_dict(),
		'optimizer_state': optimizer.state_dict()},
		'test-model.pth')
	show_result(stats, train_loss, val_loss)

def show_result(stats, train_loss, val_loss):
	print("Done!")
	plt.plot(train_loss, label='Training loss')
	plt.plot(val_loss, label='Validation loss')
	plt.legend()
	plt.show()
	window = 50
	win_rate = []
	wins = 0
	for i, r in enumerate(stats):
		if r == 1:
			wins +=1
		win_rate.append(wins/(i+1))
	plt.plot(win_rate[50:], label='win rate')
	plt.show()

def transfer_model(model, load_file, save_file):
	optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
	checkpoint = torch.load(load_file)
	model.load_state_dict(checkpoint['model_state'])
	optimizer.load_state_dict(checkpoint['optimizer_state'])
	torch.save({'model_state': model.state_dict(),
		'optimizer_state': optimizer.state_dict()},
		save_file)



def train_model(state_file, q_file, saved_model, new_model, optimizer_file):
	file = open(state_file, 'rb')
	data = loadtxt(file,delimiter = ",")
	data = data[1:]
	file = open(q_file, 'rb')

	Q_data = loadtxt(file)
	Q_data = Q_data[1:]
	q_max = np.amax(Q_data)
	q_min = np.amin(Q_data)
	newQ = (Q_data - q_min)/(q_max-q_min)
	print(len(Q_data), len(data), 'SIZE')
	print(data[0:3])
	print(Q_data[0:3])

	X = torch.Tensor(data)
	Y = torch.Tensor(Q_data)
	epochs = 23
	train_loss, val_loss, model, optimizer = mf.single_training(8,0.1, X, Y, epochs, 32, saved_model)#mf.sampler(5, X, Y, epochs, 32, saved_model)#
	torch.save(model.state_dict(),new_model)
	torch.save(optimizer.state_dict(), optimizer_file)
	torch.save({'model_state': model.state_dict(),
		'optimizer_state': optimizer.state_dict()},
		'model.pth')
	print("Done!")
	plt.plot(train_loss, label='Training loss')
	plt.plot(val_loss, label='Validation loss')
	plt.legend()
	plt.show()


#train_model('game_states.csv', 'game_q.csv', 'model.pth', 'model4-temp.pth', 'optimizer.pth')
env = f.env
model = CNN6Model()
checkpoint = torch.load('cnn_model6.pth')
model.load_state_dict(checkpoint['model_state'])
value_model = CNN6Model()
checkpoint = torch.load('cnn_model6.pth')
value_model.load_state_dict(checkpoint['model_state'])
policy_model = PolicyModel3()
checkpoint = torch.load('prob-model3.pth')
policy_model.load_state_dict(checkpoint['model_state'])

randomplayer = f.RandomPlayer(env, 'Dexter-Bot2')
filterplayer = FilterPlayer(env, show = False)
probabilityplayer = ProbabilityPlayer(env, show = False)

#Smartplayer3 uses the value network
smart = SmartPlayer3(env, model = value_model, show = False, epsilon = 1.0, smart_search = True)

#SmartPlayer4 uses the policynetwork
player1 = SmartPlayer4(env, model = policy_model, show = False, epsilon = 1.0, smart_search = True)
player2 = SmartPlayer4(env, model = policy_model, show = True, epsilon = 1.0, smart_search = True)


mctsplayer = MCTSPlayer(env,policy_network = policy_model, value_network = value_model, show = True)


#play a game and show the states 
reward, state_hist, act_hist, player_turn, =f.play_game( player2,filterplayer, show = True)
#s,r = player2.get_data(reward, filterplayer.moves[-1],{})
#print(np.array(r))
#print()
#s,r = player1.get_data(-reward, player2.moves[-1], player2.traverse_all(player2.positions[-1]))
#print(np.array(r))

#run 500 games between two players and get data
data_n_games(500,  player1, filterplayer)


#Reinforcement training for n games
#players = [SmartPlayer4(env, model = model, show = False, epsilon = 0.1, smart_search = False), SmartPlayer4(env, model = model, show = False, epsilon = 0.1, smart_search = False)]
#lf = nn.CrossEntropyLoss()
#reinforcement_training(players, model, 6000, 0.6, 0.85, load = True, epochs=35, file = 'prob-model3.pth', lr = 1e-03, loss_function = lf)
#transfer_model(model, 'test-model.pth', 'prob-model3.pth')
