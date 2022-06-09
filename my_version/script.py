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
from scipy.stats import multinomial
from scipy.stats import binom
from scipy.stats import beta


def data_n_games(n, player1, player2):
	states = []
	q = []
	win_rate = []
	nr_saved = 0
	size = []
	win_state = []
	for i in range(n):
		print('new game')
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
		#value_data, policy_data = player2.get_data([])
		#r = value_data['exp_reward'][0]
		#b = value_data['boards'][0]
		#if r == 1:
		#	if abs(np.array(b)).sum() %2 == 0:
		#		r = -1
		print(game_q[-1], abs(game_states[-1]).sum())
		if game_q[-1] == 0.5:
			print(np.array(game_states[-1]).reshape(6,7))
		win_state.append(game_states[-1].tolist())
		win_rate.append(game_q[-1])
		
		#size.append(len(game_states))
	print('Saved games', nr_saved)
	print('Wins : ', win_rate.count(1))
	print('Loss : ', win_rate.count(-1))
	print('Draw : ', win_rate.count(0))
	print('Draw : ', win_rate.count(0.5))
	print(win_rate)
	q = np.array(q)
	states = np.array(states)
	win_state = np.array(win_state)
	df = pd.DataFrame(states)
	df.to_csv('game_states.csv', index = False)
	df_q = pd.DataFrame(q)
	df_q.to_csv('game_q.csv', index = False)
	df_win = pd.DataFrame(win_state)
	df_win.to_csv('win_states.csv', index = False)
	return states

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

def play_n_games(n, player1, player2):
	value_states = []
	rewards = []
	policy_states = []
	move_prob = []
	stats = []
	for i in range(n):
		reward, state_hist, _, player_turn = f.play_game(player1, player2)
		stats.append(reward)
		state_hist.popleft()
		game = np.array(state_hist)
		value_data, policy_data = player1.get_data([])
		value_states = value_states + value_data['boards']
		rewards = rewards + value_data['exp_reward']
		policy_states = policy_states + policy_data['boards']
		move_prob = move_prob + policy_data['prob_vector']
		if i % 5 == 0:
			player1.new_game()
			player2.new_game()
		if i %25 == 0:
			print('number of games played %s' %i)
	return value_states, rewards, policy_states, move_prob, stats

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
		value_states, rewards, policy_states, move_prob =  n_games(NR_GAMES, player1, player2, gamma)
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


def reinforcement_training_mcts(players, models, n, epsilon, end_epsilon,epochs=1, load = False, files = ['cnn_model6.pth','prob-model3.pth'], lr = [0.001, 0.0001], loss_functions = [nn.HuberLoss(), nn.CrossEntropyLoss()]):
	for fil in files:
		print('loading from ... %s' %fil)
	env = f.env
	value_model = models[0]
	policy_model = models[1]
	optimizer_v = torch.optim.SGD(value_model.parameters(), lr=lr[0], momentum= 0.9)
	optimizer_p = torch.optim.SGD(policy_model.parameters(), lr=lr[1], momentum= 0.9, weight_decay = 0.01)
	value_loss = loss_functions[0]
	policy_loss = loss_functions[1]
	if load:
		checkpoint = torch.load(files[0])
		value_model.load_state_dict(checkpoint['model_state'])
		optimizer_v.load_state_dict(checkpoint['optimizer_state'])
		optimizer_v.param_groups[0]['lr'] = lr[0]
		checkpoint = torch.load(files[1])
		policy_model.load_state_dict(checkpoint['model_state'])
		#optimizer_p.load_state_dict(checkpoint['optimizer_state'])
		optimizer_p.param_groups[0]['lr'] = lr[1]
		print(optimizer_p)
	player1 = players[0]
	player2 = players[1]
	randomplayer = f.RandomPlayer(env, 'Dexter-Bot2')
	filterplayer = FilterPlayer(env, 'Dexter-Bot2')
	train_loss = []
	val_loss = []
	value_loss_t = []
	value_loss_v = []
	policy_loss_t = []
	policy_loss_v = []
	stats = []
	decay = -np.log(1-0.99)/n
	delta = end_epsilon-epsilon
	gamma = 1.0
	batch_size =16
	NR_ROUNDS = 1
	NR_GAMES = int(n/NR_ROUNDS)
	training_states_value = []
	training_rewards = []
	training_states_policy = []
	training_prob = []
	for j in range(NR_ROUNDS):
		eps = round(epsilon + delta*(1-np.exp(-len(stats)*decay)),3)
		print('epsilon: ', eps)
		#player1, player2 = update_players(player1, player2, model, eps)
		value_states, rewards, policy_states, move_prob, stats =  play_n_games(NR_GAMES, player1, player2)
		stats += stats
		training_states_value += value_states
		training_rewards += rewards
		training_states_policy += policy_states
		training_prob += move_prob
		print(len(training_states_value))
		X, Y = datasets(training_states_value, training_rewards)
		#optimizer.param_groups[0]['lr'] = lr/2
		value_model, optimizer_v, train_l,val_l, improvement = mf.train_batch(X, Y, value_model,value_loss, optimizer_v, epochs, batch_size, True)
		value_loss_t += train_l
		value_loss_v += val_l

		X, Y = datasets(training_states_policy, training_prob)
		policy_model, optimizer_p, train_l,val_l, improvement = mf.train_batch(X, Y, policy_model,policy_loss, optimizer_p, epochs, 2*batch_size, True)
		policy_loss_t += train_l
		policy_loss_v += val_l
		player1.new_game()
		player2.new_game()
	torch.save({'model_state': value_model.state_dict(),
		'optimizer_state': optimizer_v.state_dict()},
		files[0])
	torch.save({'model_state': policy_model.state_dict(),
		'optimizer_state': optimizer_p.state_dict()},
		files[1])
	show_result(stats, value_loss_t, value_loss_v)
	show_result(stats, policy_loss_t, policy_loss_v)


def show_result(stats, train_loss, val_loss):
	print("Done!")
	plt.plot(train_loss, label='Training loss')
	plt.plot(val_loss, label='Validation loss')
	plt.legend()
	plt.show()
	window = 50
	win_rate = []
	results = 0
	for i, r in enumerate(stats):
		results += r if r == 1 else 0.5*(r==0)
		win_rate.append(results/(i+1))
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

def crazy(v):
	a = [0]*len(v)
	inx = np.argsort(v)
	p = 1
	for i in range(1,len(v)):
		j = inx[i]
		k = inx[i-1]
		if v[k] != 0:
			a[k] = p
			p = p*v[j]/v[k]
			a[j] = p
	return a

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

def test_games(n, player1, player2):
	states = []
	q = []
	win_rate = []
	nr_saved = 0
	size = []
	win_state = []
	for i in range(n):
		print('new game')
		game_states, game_q, player1 = f.state_reward_model(player1, player2)
		win_rate.append(game_q[-1])
		player1.new_game()
		player2.new_game()
		#size.append(len(game_states))
	print('Wins : ', win_rate.count(1))
	print('Loss : ', win_rate.count(-1))
	print('Draw : ', win_rate.count(0.5))
	print('Draw : ', win_rate.count(0))
	print(win_rate)

#train_model('game_states.csv', 'game_q.csv', 'model.pth', 'model4-temp.pth', 'optimizer.pth')
files = ['mcts-value.pth','mcts-policy2.pth']
loaders = ['cnn_model6.pth','prob-model3.pth']
env = f.env
old_value = CNN6Model()
checkpoint = torch.load(loaders[0])
old_value.load_state_dict(checkpoint['model_state'])
old_policy = PolicyModel3()
checkpoint = torch.load(loaders[1])
old_policy.load_state_dict(checkpoint['model_state'])

value_model = CNN6Model()
checkpoint = torch.load(files[0])
value_model.load_state_dict(checkpoint['model_state'])
policy_model = PolicyModel3()
checkpoint = torch.load(files[1])
policy_model.load_state_dict(checkpoint['model_state'])

randomplayer = f.RandomPlayer(env, 'Dexter-Bot2')
filterplayer = FilterPlayer(env, show = False)
probabilityplayer = ProbabilityPlayer(env, show = False)

#Smartplayer3 uses the value network
smart = SmartPlayer3(env, model = value_model, show = False, epsilon = 1.0, smart_search = False)

#SmartPlayer4 uses the policynetwork
player1 = SmartPlayer4(env, model = policy_model, show = False, epsilon = 0.0, smart_search = True)
player2 = SmartPlayer4(env, model = policy_model, show = True, epsilon = 0.0, smart_search = True)

alphabeta = AlphaBeta(env)

old_mcts = MCTSPlayer(env,policy_network = old_policy, value_network = old_value, show = False)
mctsplayer = MCTSPlayer(env,policy_network = policy_model, value_network = value_model, show = False)
mctsplayer_train= MCTSPlayer(env,policy_network = policy_model, value_network = value_model, show = False)


#play a game and show the states 
#reward, state_hist, act_hist, player_turn, =f.play_game( mctsplayer_train,alphabeta, show = True)
#s,r = mctsplayer_train.get_data([])
#for i in range(20):
#	reward, state_hist, act_hist, player_turn, =f.play_game( randomplayer,randomplayer, show = False)
#	print(state_hist[-1], reward, act_hist[-1], 'lm')
#	print()
#print(np.array(r))
#print()
#s,r = player1.get_data(-reward, player2.moves[-1], player2.traverse_all(player2.positions[-1]))
#print(np.array(r))

#run 500 games between two players and get data
player_types = ['MCTS', 'AlphaB', 'Filter', 'Policy', 'Value', 'Random']
confusion = [
	[[24, 18,8],[76,2,22], [50,0,0], [81,14,5],[99,0,1], [50,0,0]],
	[[8,37,5],[11,10,29], [49,1,0], [74,15,11],[50,0,0], [50,0,0]],
	[[6,44,0], [2,48,0], [32,18,0],[19,31,0], [24,26,0], [50,0,0]],
	[[30,47,23], [134,66,0], [290,201,9],[165,74,61], [258,39,3], [50,0,0]],
	[[2,47,1], [144,46,10], [476,24,0],[198,1,1], [30,20,0], [498,2,0]],
	[[0,50,0], [0,50,0], [0,50,0],[1,49,0], [4,46,0], [27,23,0]]
	]
wld = [379,107,14]
wld = [290,201,9]
def c_m(players, scores):
	reward  = np.array([1,0,0.5])
	mat = [[round( (np.array(res)*reward).sum()/sum(res),3) for res in games] for games in scores]
	print()
	print()
	print('		WIN RATIO	 ')
	print(end = " \t| ")
	for p in players:
		print(  p, end = '\t')
	print('\n-------------------------------------------------------\n')
	for i in range(len(players)):
		print( players[i], end = "\t| ")
		for s in mat[i]:
			print(str(s) ,end = "\t ")
		print('\n------------------------------------------------------\n')

def Probability(rating1, rating2):
  
    return 1.0 * 1.0 / (1 + 1.0 * np.power(10, 1.0 * (rating1 - rating2) / 400))
  

def comb(mat, n, k, i, tot):
	if tot == k:
		return n
	elif i == len(n)-1:
		res = [x for x in n]
		res[i] = k-tot
		return res
	for j in range(0,k+1):
		if tot + j <k:
			res = [x for x in n]
			res[i] = j
			draw = comb(mat, res, k, i+1, tot+j)
			mat.append(draw)
		elif tot + j ==k:
			res = [x for x in n]
			res[i] = j
			return res
	return n

# Function to calculate Elo rating
# K is a constant.
# d determines whether
# Player A wins or Player B. 
def EloRating(Ra, Rb, K, d):
   
  
    # To calculate the Winning
    # Probability of Player B
    Pb = Probability(Ra, Rb)
  
    # To calculate the Winning
    # Probability of Player A
    Pa = Probability(Rb, Ra)
  
    # Case -1 When Player A wins
    # Updating the Elo Ratings
    if (d == 1) :
        Ra = Ra + K * (1 - Pa)
        Rb = Rb + K * (0 - Pb)
      
  
    # Case -2 When Player B wins
    # Updating the Elo Ratings
    else :
        Ra = Ra + K * (0 - Pa)
        Rb = Rb + K * (1 - Pb)
    return Ra, Rb
    print("Updated Ratings:-")
    print("Ra =", round(Ra, 6)," Rb =", round(Rb, 6))


def rate_all(players, games):
	player_ratings = [1200]*len(players)
	for player1, matches in enumerate(games):
		for player2, match  in enumerate(matches):
			if player1 != player2:
				results = [1]*match[0] + [-1]*match[1]
				#random.shuffle(results)
				for r in results:
					r1, r2 = EloRating(player_ratings[player1], player_ratings[player2], 100, r)
					player_ratings[player1] = r1
					player_ratings[player2] = r2
	return [round(r) for r in player_ratings]




#data_n_games(50,  alphabeta, mctsplayer_train)
#data_n_games(50,  alphabeta, mctsplayer_train)
#data_n_games(50,  alphabeta, mctsplayer_train)
#data_n_games(33,  mctsplayer_train, alphabeta)
#data_n_games(33,  mctsplayer_train, alphabeta)
c_m(player_types, confusion)
#print(Probability(1100,1100))
#elo = rate_all(player_types, confusion)
#prob = [[round(Probability(r2,r1),2) for r2 in elo] for r1 in elo]
#print(elo)
#print(np.array(prob))
#print(mctsplayer.avg_nr_evaluations)
#print(alphabeta.avg_nr_evaluations)
#s,r = mctsplayer.get_data([])
#test_games(20, mctsplayer, old_mcts)
#print(mctsplayer.time_game)

#for i in range(7):
#	for j in range(7):
#		key = "3"+str(i)+str(j)
#		if key in mctsplayer.state_data:
#			print(key)
#			print(mctsplayer.state_data[key]['visits'])
#			print()
players = [mctsplayer_train, mctsplayer_train]
#(players, models, n, epsilon, end_epsilon,epochs=1, load = False, files = ['cnn_model6.pth','prob-model3.pth'], lr = [0.0005, 0.001], loss_functions = [nn.HuberLoss(), nn.CrossEntropyLoss()])
#reinforcement_training_mcts(players, [value_model, policy_model], 500, 0.6, 0.85, load = True, epochs=150, files= files)
#play_n_games(20, mctsplayer, mctsplayer)
#print(mctsplayer.time_game, 'times elapsed')

policy_improvement = []
#value_data, policy_data = mctsplayer.get_data(np.array(states[-1]).reshape((6,7)))

def simprob(p):
	probs = np.array([i for i in p])
	left = np.array([True for i in p])
	while left.sum() > 1:
		new = left*np.array([p > random.random() for p in probs])
		if new.sum() == 1:
			return new
		elif new.sum() == 0:
			return left
		left = new
	return left

def simprob2(n,k,prior, p):
	w = prior*k
	tot = k
	for i in range(n):
		s = binom.rvs(k,p)
		w+=s
		tot+=k
		p = w/tot
	return p

def n_sim(n, probs):
	wins = [0]*len(probs)
	for i in range(n):
		result = simprob(probs)
		for j, r in enumerate(result):
			if r:
				wins[j] += 1
	return np.array(wins)/np.array(wins).sum()

def crazy_sim(n,sample, probs, eps):
	pseudo = 10
	min_p = np.array(probs).mean()
	wins = np.array([1.0]*len(probs)) + np.array(probs)*pseudo
	alpha = wins/wins.sum()
	ps = []
	e = 0.1
	for i in range(n):
		block = np.array([0]*len(probs))
		for j in range(sample):
			block += simprob(probs)
		n = block.sum()
		rv = multinomial(n, alpha)
		p = rv.pmf(block)+eps
		frac = -1/np.log(p)
		wins += block*p
		e += (-e+ abs(block*p).sum()/wins.sum())/(1+i)
		alpha = wins/wins.sum()
		if e < 1e-05:
			break
	return alpha

def comb(n,k,x):
	v = [1]*len(x)
	for i in range(len(x)-1):
		for j in range(i, len(x)):
			res = crazy_sim(n,k, [x[i], x[j]], 1/60)
			if res[0] > res[1]:
				v[i] = v[i]*(res[0]/res[1])
			else:
				v[j] = v[j]*(res[1]/res[0])
	return v

def n_sim2(samples,n, probs,b1, b2, lr):
	wins = np.array([0]*len(probs)).astype(float)
	mt = np.array([0]*len(probs)).astype(float)
	vt = np.array([0]*len(probs)).astype(float)
	prob_est = np.array([1/len(probs)]*len(probs)).astype(float)
	est_win = np.array([0]*len(probs)).astype(float)
	eps = 1e-04
	gt = np.array([0]*len(probs)).astype(float)
	t = 0
	for i in range(30):
		t+=1
		res = np.array([1 if w else 0 for w in simprob(probs)])
		gt = (wins+res) - (est_win+prob_est)
		mt = b1*mt + (1-b1)*gt
		vt = b2*vt + (1-b2)*np.power(gt,2)
		mhat = mt/(1-b1**t)
		vhat = vt/(1-b2**t)
		prob_est += lr*mhat/(np.sqrt(vhat) + eps)
		est_win += prob_est
		wins += res
	for i in range(samples):
		t+=1
		res = np.array([1 if w else 0 for w in simprob(probs)])
		gt = (wins+res) - (est_win+prob_est)
		mt = b1*mt + (1-b1)*gt
		vt = b2*vt + (1-b2)*np.power(gt,2)
		mhat = mt/(1-b1**t)
		vhat = vt/(1-b2**t)
		prob_est += lr*mhat/(np.sqrt(vhat) + eps)
		est_win += prob_est
		wins += res
	for i in range(n):
		t+=1
		gt = (wins+prob_est) - (est_win+prob_est)
		mt = b1*mt + (1-b1)*gt
		vt = b2*vt + (1-b2)*np.power(gt,2)
		mhat = mt/(1-b1**t)
		vhat = vt/(1-b2**t)
		prob_est += lr*mhat/(np.sqrt(vhat) + eps)
		est_win += prob_est
		wins += prob_est
	return wins/wins.sum()

def winsim(n, wp):
	cond_p = np.array(wp)
	cond_p = cond_p/cond_p.sum()
	res = [0]*len(cond_p)
	for i in range(n):
		r = random.random()
		cum = 0
		for i in range(len(cond_p)):
			cum += cond_p[i]
			if cum > r:
				res[i] += 1
				break
	return res

def maxprob(prob):
	tot = 1
	share = [0]*len(prob)
	idx = np.argsort(prob).tolist()
	idx.reverse()
	for i in range(10):
		for j in idx:
			share[j] += prob[j]*tot
			tot = prob[j]*tot
	return share

def lol(m):
	if all([n == 0 for n in m]):
		return 1
	count = 0
	for i in range(len(m)):
		if m[i] > 0:
			new = [n for n in m]
			new[i] -= 1
			count += lol(new)
	return count
#Reinforcement training for n games
#players = [SmartPlayer4(env, model = model, show = False, epsilon = 0.1, smart_search = False), SmartPlayer4(env, model = model, show = False, epsilon = 0.1, smart_search = False)]
#lf = nn.CrossEntropyLoss()
#reinforcement_training(players, model, 6000, 0.6, 0.85, load = True, epochs=35, file = 'prob-model3.pth', lr = 1e-03, loss_function = lf)
#transfer_model(value_model, 'test-value-model.pth', 'mcts-value.pth')
#transfer_model(policy_model, 'test-prob.pth', 'mcts-policy2.pth')

