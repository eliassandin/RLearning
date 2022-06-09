from abc import ABC, abstractmethod
import numpy as np
import torch
import random
import stats
import copy
from torch import nn
import scipy.stats as ss
import time
import sys
from collections import defaultdict
from scipy.stats import multinomial



diag2 = [[(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 1), (1, 2), (2, 3), (3, 4)], [(0, 2), (1, 3), (2, 4), (3, 5)], [(0, 3), (1, 4), (2, 5), (3, 6)], [(1, 0), (2, 1), (3, 2), (4, 3)], [(1, 1), (2, 2), (3, 3), (4, 4)], [(1, 2), (2, 3), (3, 4), (4, 5)], [(1, 3), (2, 4), (3, 5), (4, 6)], [(2, 0), (3, 1), (4, 2), (5, 3)], [(2, 1), (3, 2), (4, 3), (5, 4)], [(2, 2), (3, 3), (4, 4), (5, 5)], [(2, 3), (3, 4), (4, 5), (5, 6)]]
diag = [[(5, 0), (4, 1), (3, 2), (2, 3)], [(5, 1), (4, 2), (3, 3), (2, 4)], [(5, 2), (4, 3), (3, 4), (2, 5)], [(5, 3), (4, 4), (3, 5), (2, 6)], [(4, 0), (3, 1), (2, 2), (1, 3)], [(4, 1), (3, 2), (2, 3), (1, 4)], [(4, 2), (3, 3), (2, 4), (1, 5)], [(4, 3), (3, 4), (2, 5), (1, 6)], [(3, 0), (2, 1), (1, 2), (0, 3)], [(3, 1), (2, 2), (1, 3), (0, 4)], [(3, 2), (2, 3), (1, 4), (0, 5)], [(3, 3), (2, 4), (1, 5), (0, 6)]]
hor = [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 1), (0, 2), (0, 3), (0, 4)], [(0, 2), (0, 3), (0, 4), (0, 5)], [(0, 3), (0, 4), (0, 5), (0, 6)], [(1, 0), (1, 1), (1, 2), (1, 3)], [(1, 1), (1, 2), (1, 3), (1, 4)], [(1, 2), (1, 3), (1, 4), (1, 5)], [(1, 3), (1, 4), (1, 5), (1, 6)], [(2, 0), (2, 1), (2, 2), (2, 3)], [(2, 1), (2, 2), (2, 3), (2, 4)], [(2, 2), (2, 3), (2, 4), (2, 5)], [(2, 3), (2, 4), (2, 5), (2, 6)], [(3, 0), (3, 1), (3, 2), (3, 3)], [(3, 1), (3, 2), (3, 3), (3, 4)], [(3, 2), (3, 3), (3, 4), (3, 5)], [(3, 3), (3, 4), (3, 5), (3, 6)], [(4, 0), (4, 1), (4, 2), (4, 3)], [(4, 1), (4, 2), (4, 3), (4, 4)], [(4, 2), (4, 3), (4, 4), (4, 5)], [(4, 3), (4, 4), (4, 5), (4, 6)], [(5, 0), (5, 1), (5, 2), (5, 3)], [(5, 1), (5, 2), (5, 3), (5, 4)], [(5, 2), (5, 3), (5, 4), (5, 5)], [(5, 3), (5, 4), (5, 5), (5, 6)]]
vert = [[(0, 0), (1, 0), (2, 0), (3, 0)], [(1, 0), (2, 0), (3, 0), (4, 0)], [(2, 0), (3, 0), (4, 0), (5, 0)], [(0, 1), (1, 1), (2, 1), (3, 1)], [(1, 1), (2, 1), (3, 1), (4, 1)], [(2, 1), (3, 1), (4, 1), (5, 1)], [(0, 2), (1, 2), (2, 2), (3, 2)], [(1, 2), (2, 2), (3, 2), (4, 2)], [(2, 2), (3, 2), (4, 2), (5, 2)], [(0, 3), (1, 3), (2, 3), (3, 3)], [(1, 3), (2, 3), (3, 3), (4, 3)], [(2, 3), (3, 3), (4, 3), (5, 3)], [(0, 4), (1, 4), (2, 4), (3, 4)], [(1, 4), (2, 4), (3, 4), (4, 4)], [(2, 4), (3, 4), (4, 4), (5, 4)], [(0, 5), (1, 5), (2, 5), (3, 5)], [(1, 5), (2, 5), (3, 5), (4, 5)], [(2, 5), (3, 5), (4, 5), (5, 5)], [(0, 6), (1, 6), (2, 6), (3, 6)], [(1, 6), (2, 6), (3, 6), (4, 6)], [(2, 6), (3, 6), (4, 6), (5, 6)]]
patterns = []
patterns = patterns + diag + diag2 + hor + vert

pattern_map = {}
for y in range(6):
    for x in range(7):
        pos = (y,x)
        ps = []
        for p in patterns:
            if pos in p:
                ps.append(p)
        pattern_map[pos] = ps


class Player(ABC):
    """ Class used for evaluating the game """

    def __init__(self, env: 'ConnectFourEnv', name='Player'):
        self.name = name
        self.env = env

    @abstractmethod
    def get_next_action(self, state: np.ndarray) -> int:
        pass

    def learn(self, state, action: int, state_next, reward: int, done: bool) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        raise NotImplementedError()

    def load_model(self, model_prefix: str = None):
        raise NotImplementedError()

    def reset(self, episode: int = 0, side: int = 1) -> None:
        """
        Allows a player class to reset it's state before each round

            Parameters
            ----------
            episode : which episode we have reached
            side : 1 if the player is starting or -1 if the player is second
        """
        pass




class FilterPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent', show = False):
        super().__init__(env, name)
        self.show = show
        filt_diag = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        filt_diag2 = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        filt_row = np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        filt_row2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]])
        filt_col = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
        filt_col2 = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]])
        self.filters = [filt_diag, filt_diag2, filt_row, filt_row2, filt_col, filt_col2]
        self.moves = []

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        exp_reward = [-float('inf'),-float('inf'),-float('inf'),-float('inf'),-float('inf'),-float('inf'),-float('inf')]
        for action in available_moves:
            board_copy = self.env.next_state(action)
            score = self.filter_all(self.filters, board_copy)
            exp_reward[action] = score
        exp_reward = np.array(exp_reward)
        if self.show:
            print(exp_reward)
        if not available_moves or exp_reward.size == 0:
            print('NO AVAILABLE')
            print(state)
            return 0 
        move = np.argmax(exp_reward)
        if (exp_reward==0).sum() == len(available_moves):
            p = self.probs(available_moves)
            if self.show:
                print(p)
            p_ = 0
            r = random.random()
            valid_moves = list(available_moves)
            for i in range(len(valid_moves)):
                p_ += p[i]
                if p_ > r :
                    move = valid_moves[i]
                    break
        self.moves.append(move)
        if self.show:
            print(move)
        return move
    
    def filter_all(self, filters, b):
        scores = [self.convolve(f, b) for f in filters]
        return np.array(scores).sum()
    
    def convolve(self, filt, b):
        y, x = b.shape
        n,k = filt.shape
        s = [[self.calc_score(filt*b[j: j+n, i:i+k]) for i in range(0, x-n+1) ] for j in range(0, y-n+1)]
        return s

    def calc_score(self, matrix):
        sum_to_score = {0:0, 1:0,2:2, 3:3, 4:1000, -1:0, -2:-2 , -3: -100, -4: -1000}
        return sum_to_score[matrix.sum()]

    def probs(self, available_moves):
        valid_moves = list(available_moves)
        mu = (len(valid_moves)-1)/2
        sigma = 1
        p = [self.distr(mu, sigma, x) for x in range(len(available_moves))]
        return np.array(p)/sum(p)

    def distr(self, mu, sigma, x):
        return np.exp(-1/2*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

    def save_model(self, model_prefix: str = None):
        pass


class SmartPlayer3(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent', model = None, show = False, epsilon = 1.0, smart_search = False):
        super().__init__(env, name)
        self.model = model
        self.show = show
        self.epsilon = epsilon
        self.smart_search = smart_search
        self.abs_mean = [0]*20
        self.variance = [0]*20
        self.mean = [0]*20
        self.m_40 = stats.mm_40
        self.m_60 = stats.mm_60
        self.m_15 = stats.mm_15
        self.m_85 = stats.mm_85
        self.move_nr = 0
        model.eval()
    
    def set_model(self, new_model):
        new_model.eval()
        self.model = new_model

    def set_epsilon(self, eps):
        self.epsilon = eps

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        r = random.random()
        if r > self.epsilon and not self.smart_search:
            return random.choice(list(available_moves))

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        exp_reward = [-20,-20,-20,-20,-20,-20,-20]
        for action in available_moves:
            board_copy = self.env.next_state(action)
            X = torch.Tensor(np.array([board_copy]))
            s = self.model(X)
            score = s.detach().numpy()[0][0]
            exp_reward[action] = score
        exp_reward = np.array(exp_reward)
        exp = exp_reward[list(available_moves)]
        mean = np.mean(abs(exp))
        var = np.var(exp)
        move = self.eps_greedy(exp_reward, available_moves, self.smart_search)
        if self.show:
            print(exp_reward)
        self.move_nr +=1

        if not available_moves or exp_reward.size == 0:
            print('NO AVAILABLE')
            print(state)
            return 0
        if self.show:
            board_copy = self.env.next_state(move)
            X = torch.Tensor(np.array([board_copy]))
            #traceback = np.around(self.model.traceback(X).detach().numpy(), 3)
            #idx = np.argsort(abs(traceback.flatten()))
            #threshold = idx.max() - 15
            #idx = idx > threshold
            #flat = traceback.flatten()*idx
            #traceback = flat.reshape((6,7))
            #print(traceback)
            print(move)
        return move
    
    def reward_variance(self, exp_reward, available_moves):
        exp = exp_reward[list(available_moves)]
        var = np.var(exp)
        mean = np.mean(abs(exp))
        #print(np.mean(exp),self.m_15[self.move_nr],self.m_85[self.move_nr])
        #print(np.mean(exp),self.m_40[self.move_nr],self.m_60[self.move_nr])
        if self.move_nr < 12 and self.move_nr>2:
            if np.mean(exp) > self.m_40[self.move_nr] and np.mean(exp) < self.m_60[self.move_nr]:
                return self.epsilon
            self.variance[self.move_nr] = var
            self.abs_mean[self.move_nr]=mean
            self.mean[self.move_nr]=np.mean(exp)
        if self.move_nr > 1 and self.move_nr < 12 and self.smart_search:
            if np.mean(exp) < self.m_15[self.move_nr]:
                return self.epsilon
        return self.epsilon

#low var and low avg made wrong pred, low variance high avg made right pred, 4
#low/mid var and mid/high avg made right pred
#low var high avg made right pred, once again
#high var and high avg (negative) made half right pred
#low var and mid/high avg made wrong, once again
#low var and low avg made right pred?
#low var negative mean shit pred
#mid var negative mean half good prediction
#low var and high mean made bad pred (player 2)
    def prob_move(self, p, available_moves, exp_reward, explore = False):
        valid_moves = list(available_moves)
        idx_sort = np.argsort(exp_reward)
        large_to_small = np.flip(idx_sort)
        move_probability = [0,0,0,0,0,0,0]
        start = 0
        if explore:
            start = 1
            p = 0.3
        for i in range(start, len(valid_moves)-start):
            idx = large_to_small[i]
            move_probability[i] = p*(1-p)**(i-start)
        move_probability = np.array(move_probability)
        if sum(move_probability) > 0:
            move_probability = move_probability/ sum(move_probability)
        return  move_probability, large_to_small

    def eps_greedy(self, exp_reward, available_moves, smart = False):
        dumb = random.choice(list(available_moves))
        prob = self.reward_variance(exp_reward, available_moves)
        p_adjusted = prob**(len(available_moves)/7)
        
        r = random.random()
        move_probability, idx_sort = self.prob_move(p_adjusted, available_moves, exp_reward, r>self.epsilon)
        #print(idx_sort)
        cum_prob = 0
        for i, p in enumerate(move_probability):
            cum_prob += p
            move = idx_sort[i]
            if cum_prob >= r:
                break
        if move not in available_moves:
            return dumb
        return move
    def get_move(self, exp_reward, available_moves):
        move = -1
        print(exp_reward)
        while move not in available_moves and max(exp_reward) > -20:
            move = np.argmax(exp_reward)
            exp_reward[move] = -20
        return move, exp_reward

    def show_decision(self):
        nrows = 10
        ncols = 10

        data = np.zeros(nrows*ncols)
        data[Cellid] = Cellval
        data = np.ma.array(data.reshape((nrows, ncols)), mask=data==0)

        fig, ax = plt.subplots()
        ax.imshow(data, cmap="Greens", origin="lower", vmin=0)

        # optionally add grid
        ax.set_xticks(np.arange(ncols+1)-0.5, minor=True)
        ax.set_yticks(np.arange(nrows+1)-0.5, minor=True)
        ax.grid(which="minor")
        ax.tick_params(which="minor", size=0)

        plt.show()

    def random_move(self, exp_reward, available_moves, smart = False):
        dumb = random.choice(list(available_moves))
        move, exp_reward = self.get_move(exp_reward, available_moves)
        while random.random() > self.epsilon and max(exp_reward) > -20:
            if not smart:
                print(dumb,'dumb')
                return dumb
            move, exp_reward = self.get_move(exp_reward, available_moves)
        if move not in available_moves:
            move = dumb
        return move

    def save_model(self, model_prefix: str = None):
        pass




class ProbabilityPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent', show = False):
        super().__init__(env, name)
        self.show = show

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        move_probability = [0]*7
        valid_moves = list(available_moves)
        my_nr = sum((state[:, valid_moves]== 1))+1
        my_nr = my_nr * (my_nr < 5)
        his_nr = sum((state[:, valid_moves]== -1))+1
        his_nr = his_nr * (his_nr < 5)
        not_filled = ((his_nr+my_nr >= 6)*(my_nr - his_nr == 0) == False)
        adj = my_nr*his_nr*not_filled
        p = self.probs(available_moves)
        p = adj*p /(adj*p+1e-03).sum()
        p_ = 0
        r = random.random()
        move = random.choice(valid_moves)
        for i, m in enumerate(valid_moves):
            move_probability[m] = p[i]
        for i, m in enumerate(valid_moves):
            p_ += p[i]
            if p_ > r :
                move = m
                break
        if self.show:
            print(move_probability)
            print(adj)
            print(move)
        return move

    def probs(self, available_moves):
        valid_moves = list(available_moves)
        mu = (len(valid_moves)-1)/2
        sigma = 1
        p = [self.distr(mu, sigma, x) for x in range(len(available_moves))]
        return np.array(p)/sum(p)

    def distr(self, mu, sigma, x):
        return np.exp(-1/2*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

    def save_model(self, model_prefix: str = None):
        pass



class MCTSNode():
    def __init__(self, state, parent=None, parent_action=None, priors = [1/7]*7, value = 0, opponent_value = 0, n_visits = 0, current_player = 1, W = 0):
        self.state = state
        self.opponent_value = opponent_value
        self.priors = priors
        self.parent = parent
        self.parent_action = parent_action
        self.current_player = -parent.current_player if parent != None else 1
        self.path = "" if parent == None else parent.path+str(parent_action)
        self.children = []
        self.n_visit = [0]*len(self.available_moves())
        self.value = value 
        self.ys = [6]*7
        if parent != None:
            self.ys = [parent.ys[i] if i!= parent_action else parent.ys[i]-1  for i in range(7)]
        self.Wr = 0
        self.N = 0
        self.Wv = value
        self.Wr = 0
        self.Nv = 0
        self.Nr = 0
        self.satisfied = False
        return

    def is_valid_action(self, action: int) -> bool:
        return self.state[0][action] == 0

    def available_moves(self) -> frozenset:
        return {i for i in range(self.state.shape[1]) if self.is_valid_action(i)}

    def add_child(self, node):
        self.children.append(node)

    def get_child(self, action):
        i = -1
        for i, child in enumerate(self.children):
            if child.parent_action == action:
                return i, child
        return i, None

    def next_state(self,current_player, action: int):

        board_copy = copy.deepcopy(self.state)*current_player
        if not self.is_valid_action(action):
            raise Exception(
                'Unable to determine a valid move! Maybe invoke at the wrong time?'
            )
        #self.untried_actions = self.untried_actions.difference({action})
        # Check and perform action
        for index in list(reversed(range(self.state.shape[0]))):
            if board_copy[index][action] == 0:
                board_copy[index][action] = 1
                break
        return board_copy
def stringboard(x):
    res = ""
    count_zero = 0
    zeros = ""
    for row in list(reversed(range(x.shape[0]))):
        col_str = ""
        for col in list(reversed(range(x.shape[1]))):
            if x[row][col] == 1:
                count_zero = 0
                col_str = 'a' +zeros+ col_str
                zeros = ""
            elif x[row][col] == -1:
                count_zero = 0
                col_str = 'b' +zeros+ col_str
                zeros = ""
            else:
                count_zero += 1
                if count_zero == 7:
                    return res
                zeros = '-' + zeros
        res = col_str + res
    return res
def eq_board(a,b, cp=1):
    num_com = 0
    for row in list(reversed(range(a.shape[0]))):
        count_zero = 0
        for col in range(a.shape[1]):
            num_com += 1
            if a[row][col] != b[row][col]*cp:
                return False
            elif a[row][col] == 0:
                count_zero +=1
                if count_zero == 7:
                    return True
    return True
def next_state(state, current_player, action: int):
    board_copy = copy.deepcopy(state)
    #self.untried_actions = self.untried_actions.difference({action})
    # Check and perform action
    for index in list(reversed(range(state.shape[0]))):
        if board_copy[index][action] == 0:
            board_copy[index][action] = current_player
            break
    return board_copy 

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def p_vec(x):
    return [round(i/sum(x),2) for i in x]

def mk_board(path):
    board = [[0,0,0,0,0,0,0] for i in range(6)]
    height = [5]*7
    cp = 1
    for m in path:
        x = int(m)
        y = height[x]
        board[y][x] = cp
        height[x] -= 1
        cp = -cp
    return np.array(board)

def simprob(p):
    probs = np.array([i for i in p])
    left = np.array([True for i in p])
    i  = 0
    while left.sum() > 1 and i < 12:
        new = left*np.array([p > random.random() for p in probs])
        if new.sum() == 1:
            return new
        elif new.sum() == 0:
            return left
        left = new
        i += 1
    return left

def n_sim(n, probs):
    wins = [0]*len(probs)
    for i in range(n):
        result = simprob(probs)
        for j, r in enumerate(result):
            if r:
                wins[j] += 1
    return wins

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


def txt(v):
    print([round(v[i],2) for i in range(len(v))])

class MCTSPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent',policy_network = None, value_network = None, show = False):
        super().__init__(env, name)
        self.show = show
        policy_network.eval()
        value_network.eval()
        self.policy_network = policy_network
        self.value_network = value_network
        self.current_node = None
        self.tanh = nn.Tanh()
        x = np.arange(0, 7)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU,loc = 3, scale = 2) - ss.norm.cdf(xL,loc = 3, scale = 2)
        self.xprob = prob / prob.sum()
        self.time_game = []
        self.state_data = {}
        self.searchspace = 0
        self.avg_nr_evaluations = []

    def new_game(self):
        del self.current_node
        del self.state_data
        self.current_node = None
        self.state_data = {}
        self.policy_network.eval()
        self.value_network.eval()
        self.avg_nr_evaluations = []
    
    def exhaust(self, node, current_player, depth, max_depth, winner, board_points):
        if depth >= max_depth:
            result = None
            m = node.parent_action
            pos = (node.ys[m], m)
            r = self.check_patterns(pos,node.state)
            result = -r if r != None else None
            return None, node, board_points
        else:
            moves = node.available_moves()
            move_nr = -1
            node_satisfaction = [c.satisfied for c in node.children]
            if True in node_satisfaction:
                move_nr = node_satisfaction.index(True)
                values = [c.value if c.satisfied else -100 for c in node.children]
                idx = np.argmax(values)
                if max(values) == 1:
                    node.satisfied = True
                    node.value = -max(values)
                    child = node.children[idx]
                    pos = (child.ys[child.parent_action], child.parent_action)
                    board_points[pos[0]][pos[1]] = 4
                    child.Nv +=1
                    node.Nv +=1
                    return node.value, node, board_points
            for action in moves:
                j, child = node.get_child(action)
                if child == None:
                    next_state = node.next_state(current_player, action)
                    child = MCTSNode(current_player*next_state, parent=node, parent_action=action, current_player = -current_player)
                    node.add_child(child)
                    move_nr = len(node.children) -1
                    pos = (child.ys[action], action)
                    r = self.check_patterns(pos,next_state)
                    if r != None:
                        node.children[move_nr].satisfied = True
                        node.children[move_nr].value = r
                        node.satisfied = True
                        node.value = -r
                        node.children[move_nr].Nv +=1
                        node.Nv +=1
                        board_points[pos[0]][pos[1]] = 4
                        return -r, node, board_points
            if all([c.satisfied for c in node.children]):
                node.Nv +=1
                node.satisfied = True
                val = max([c.value for c in node.children])
                node.value = -val
                for c in node.children:
                    pos = (c.ys[c.parent_action], c.parent_action)
                return node.value, node, board_points
            if depth == 0:
                for j, c in enumerate(node.children):
                    if not c.satisfied:
                        r, node.children[j], board_points = self.exhaust(c, -current_player, depth+1, max_depth, winner, board_points)
                    if c.satisfied:
                        c.Nv +=1
                if all([c.satisfied for c in node.children]):
                    node.satisfied = True
                    val = max([c.value for c in node.children])
                    node.value = -val
                    for c in node.children:
                        pos = (c.ys[c.parent_action], c.parent_action)
                return node.value, node, board_points
            scores = [-20]*7
            for action in moves:
                j, c = node.get_child(action)
                pos = (c.ys[c.parent_action], c.parent_action)
                if winner == node.current_player:
                    scores[c.parent_action]=c.value + c.Nv/(node.Nv+1) +board_points[pos[0]][pos[1]]
                elif winner == -node.current_player:
                    if not c.satisfied:
                        scores[c.parent_action] = c.value
                    else:
                        scores[c.parent_action] = -1
            bestmove = np.argmax(scores)
            j, c = node.get_child(bestmove)
            if c == None:
                return node.value, node, board_points
            pos = (c.ys[c.parent_action], c.parent_action)
            r = self.check_patterns(pos,current_player*c.state)
            if r != None:
                node.Nv +=1
                c.satisfied = True
                c.value = r
                node.satisfied = True
                node.value = -r
                c.Nv +=1
                if r == 1:
                    board_points[pos[0]][pos[1]] = 4
                return -node.children[j].value, node, board_points
            if max(scores) != -20:
                r, node.children[j], board_points = self.exhaust(c, -current_player, depth, max_depth,winner, board_points)
                if r != None:
                    node.value += (-node.value -r)/(node.Nv +2)
            if node.children[j].satisfied and node.children[j].value == 1:
                node.satisfied = True
                node.value = -1
                node.children[j].Nv +=1
                node.Nv +=1
                return -node.children[j].value, node, board_points
            for action in moves:
                j, c = node.get_child(action)
                pos = (c.ys[c.parent_action], c.parent_action)
                if action != bestmove:
                    r = self.check_patterns(pos,current_player*c.state)
                    if r != None:
                        c.satisfied = True
                        c.value = r
                        node.satisfied = True
                        node.value = -r
                        c.Nv +=1
                        node.Nv +=1
                        board_points[pos[0]][pos[1]] = 4
                        return -node.children[j].value, node, board_points
                    if node.children[j].satisfied and node.children[j].value != -1:
                        node.satisfied = True
                        node.value = -node.children[j].value
                        c.Nv +=1
                        node.Nv +=1
                        return node.value, node, board_points
                    if not node.children[j].satisfied:
                        d = depth+1 if len(moves) > 1 else depth
                        r, node.children[j], board_points = self.exhaust(c, -current_player, d, max_depth, winner, board_points)
                        if r != None:
                            node.value += (-node.value -r)/(node.Nv +2)
                    if r == 1 and node.children[j].satisfied:
                        c.satisfied = True
                        c.value = r
                        node.satisfied = True
                        node.value = -r
                        c.Nv +=1
                        node.Nv +=1
                        board_points[pos[0]][pos[1]] = 4
                        return -node.children[j].value, node, board_points
                    if r != None:
                        node.value += (-node.value -r)/(node.Nv +2)
            if all([c.satisfied for c in node.children]):
                node.satisfied = True
                val = max([c.value for c in node.children])
                node.value = -val
                for c in node.children:
                    pos = (c.ys[c.parent_action], c.parent_action)
                node.Nv +=1
        return node.value, node, board_points
    def softmax_input(self, x):
        if x == 0:
            return -15
        else:
            return np.log(x)

    def subtree_value(self,node):
        s1 = 0
        n = 0
        if node.satisfied:
            return node.value
        for c in node.children:
            if c.satisfied and c.value == 1:
                return -1
            s1 -= c.value*(c.Nv+0.05)
            n += c.Nv+0.05
        s1 = s1/n if n != 0 else s1
        if n == 0:
            print('WARNING')
        lam = 0#1/node.Nv
        return s1
    def get_data(self, state):
        node = self.find_node(self.current_node, state)
        value_data = {
            'boards': [],
            'exp_reward': []
        }
        policy_data = {
            'boards': [],
            'prob_vector': []
        }
        pos = (node.ys[node.parent_action], node.parent_action)
        reward = self.check_patterns(pos,node.state)
        print(reward)
        print()
        boards = []
        p_vectors = []
        end_state = True
        brute = True
        last_moves = 2
        winner = -node.current_player if reward != 0 else 0
        win_rates = []
        stored = node.parent.value
        fin_moves = {'winner': winner, -1: set(), 1: set()}
        board_points = [[0 for i in range(7)] for j in range(6)]
        while node.parent != None:
            prev_action = node.parent_action
            next_state = (-node.current_player*node.state).tolist()
            rew = node.value
            last_moves -= 1
            parent = node.parent
            prev_state = (parent.current_player*parent.state).tolist()
            visits = parent.n_visit if sum(parent.n_visit) > sum([c.Nv for c in parent.children]) else [c.Nv for c in parent.children]
            probs = [0]*7
            for i, nr_v in enumerate(visits):
                action = parent.children[i].parent_action
                probs[action] = nr_v
            pre_wp = [0]*7
            for c in parent.children:
                pre_wp[c.parent_action] = round((c.value+1)/2,4)
            if node.satisfied:
                pos = (node.ys[node.parent_action], node.parent_action)
                for c in parent.children:
                    if c.satisfied and c.value == rew:
                        pos = (c.ys[c.parent_action], c.parent_action)
                        board_points[pos[0]][pos[1]] = rew*4
                brun = 0
            if brute:
                pv = parent.value if not parent.satisfied else sum([-c.value/len(parent.children) for c in parent.children])
                r, parent, fin_moves = self.exhaust(parent, parent.current_player, 0, 4, winner, board_points)
                rew = node.value
                if not parent.satisfied:
                    brute = False
                else:
                    stored = pv
            if not parent.satisfied:
                s1 = 0
                n = 0
                for c in node.children:
                    s1 -= c.value*c.Nv
                    n += c.Nv
                s1 = s1/n if n != 0 else s1
                if node.satisfied:
                    rew = node.value
                    pre_wp[node.parent_action] = (stored+1)/2
                else:
                    rew = s1
                #rew = s1 if not node.satisfied else node.value
                diff = (node.value+1)/2 - (rew+1)/2 if not node.satisfied else (stored+1)/2 - (rew+1)/2
                node.value = rew
                w_r = [0]*7
                sat = 0
                for c in parent.children:
                    win_rate = (c.value+1)/2
                    w_r[c.parent_action] = round(win_rate,4)
                    if c.satisfied and c.value == -1:
                        w_r[c.parent_action] = 0
                        if c.parent_action != node.parent_action:
                            sat += c.Nv
                #estimated nr winners
                #lam = 0.1
                #later_node = node.children[np.argmax([c.Nv for c in node.children])]
                #future_wr = 1-win_rates[-1][later_node.parent_action]
                #w_r[node.parent_action] = round((1-lam)*future_wr + lam*w_r[node.parent_action],2)
                fake = [0]*7
                for c in parent.children:
                    if len(c.children) == 0:
                        fake[c.parent_action] = round((c.value+1)/2,4)
                    elif c.satisfied or c.parent_action == node.parent_action:
                        fake[c.parent_action] = round((c.value+1)/2,4)
                    else:
                        new_v = self.subtree_value(c)
                        c.value = new_v
                        fake[c.parent_action] = round((new_v+1)/2,4)
                w_r = fake
                pre_w = crazy_sim(2,50, pre_wp, 1/100)
                post_w = crazy_sim(2,50, w_r, 1/100)
                fac = [post_w[i]/pre_w[i] if pre_w[i] != 0 else post_w[i] for i in range(len(post_w))]
                diff = 1-post_w[node.parent_action]/pre_w[node.parent_action] if pre_w[node.parent_action] != 0 else post_w[node.parent_action]
                if node.satisfied and node.value == -1:
                    if max(w_r) >= 0.4:
                        diff = 1
                        w_r = np.array(w_r)
                        w_r = w_r/w_r.sum()
                        new = (probs[node.parent_action]+sat)*w_r
                        old = [v for v in probs]
                        for c in parent.children:
                            if c.value == -1:
                                old[c.parent_action] = 0
                        old[node.parent_action] = 0
                        probs = (np.array(old)+ new).tolist()
                        w_r = w_r.tolist()
                if parent.current_player != winner and winner!=0:
                    v = np.array(probs)
                    filt = [0]*7
                    dirch = [1]*7
                    for c in parent.children:
                        if not c.satisfied and c.parent_action != node.parent_action:
                            if c.Nv-max(probs) == 0:
                                dirch[c.parent_action] = 1
                            else:
                                dirch[c.parent_action] = abs(c.Nv-max(probs))
                            filt[c.parent_action] = diff#/(c.Nv+1)
                    wins = probs[node.parent_action]*diff*(diff>0)
                    wins += sat
                    noise = np.random.dirichlet(dirch)
                    win_p = np.array(post_w)*np.array([0 if v == 0 else 1 for v in probs])#+ noise*np.array(filt)
                    prob_m_given_win = win_p/win_p.sum() if win_p.sum() != 0 else win_p
                    new_observation = wins*prob_m_given_win
                    v = v+new_observation
                    v = v.tolist()
                    if max(w_r) > 0.23:
                        probs = [round(p,2) for p in v]
                elif parent.current_player == winner:
                    if w_r[node.parent_action] >0.7:
                        probs = [round(v,2) for v in np.array(probs)*fac]
            else:
                w_r = [0]*7
                w_r[node.parent_action] = (rew + 1)/2
                for c in parent.children:
                    win_rate = (c.value+1)/2
                    w_r[c.parent_action] = win_rate
                    if c.value > rew:
                        probs[c.parent_action] += node.Nv*(1-w_r[node.parent_action])
                    if rew == 1:
                        if c.value < rew:
                            probs[c.parent_action] = 0
                win_rates.append(w_r)
            value_data['boards'].append(next_state)
            value_data['exp_reward'].append(rew)
            for c in parent.children:
                if c.satisfied and c.parent_action != node.parent_action:
                    next_state = (-node.current_player*c.state).tolist()
                    value_data['boards'].append(next_state)
                    value_data['exp_reward'].append(c.value)
            policy_data['boards'].append(prev_state)
            policy_data['prob_vector'].append(probs)
            p_vectors.append(probs)
            node = node.parent
        for i, prob in enumerate(policy_data['prob_vector']):
            new = [self.softmax_input(p) for p in prob]
            policy_data['prob_vector'][i] = new
        node = node.parent
        return value_data, policy_data
    def backtrack(self):
        print(self.current_node.value*(-self.current_node.current_player), 'result')
        new_node = self.find_node(self.current_node, [])
        last_node = True
        score = abs(self.check_if_done(1, new_node.state, new_node.path))
        score = -1 if score == 0 else score
        last_moves = 2
        while new_node.parent != None:
            new_node = new_node.parent
        self.current_node = new_node
        self.current_node = self.clean_tree()
        t = time.time()
        self.time_game.append(round(t-self.start, 2))
        print(t-self.start)
        print()
        self.start = t
    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')
        if self.current_node == None or abs(state).sum() == 0:
            self.new_game()
            self.start = time.time()
            cp = 1 if state.sum() == 0 else -1
            self.current_node = MCTSNode(cp*state)
            self.current_node.current_player = cp
        else:
            cp = 1 if state.sum() == 0 else -1
            new_node = self.find_node(self.current_node, cp*state)
            if new_node == None:
                if abs(state).sum() == 0:
                    self.backtrack()
                    print('hej')
                    print(self.current_node.state)
                    new_node = self.find_node(self.current_node, cp*state)
                    #new_node = MCTSNode(cp*state)
                else:
                    print('none found')
                    print(cp*state)
                    print(cp)
                    print(self.current_node.state, 'current node state')
                    new_node = MCTSNode(cp*state)
                    new_node.current_player = cp
                    print(new_node.state,'new node')
            self.current_node = new_node
        self.current_node = self.n_mcts(100, 4)
        scores = self.score(self.current_node)
        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        idx = np.argmax(self.current_node.n_visit)
        t = np.log(abs(state).sum()+2)
        #p = np.power([c.Nv*(len(c.children) == len(c.available_moves())) for c in self.current_node.children],t)
        #p = p/p.sum() if p.sum() != 0 else p
        #r = random.random()
        #cdf = 0
        #for i, x in enumerate(p):
        #    cdf += x
        #    if cdf >= r:
        #        idx = i
        #        break
        best_node = self.current_node.children[idx]
        #print([self.state_data[c.path]['value']//0.001/1000 for c in self.current_node.children])
        #print([c.value//0.001/1000 for c in self.current_node.children])
        #print([s//0.001/1000 for s in scores])
        #print([c.Nv for c in self.current_node.children])
        move = best_node.parent_action
        n_moves = abs(state).sum()
        if self.show:
            print([c.Nv for c in self.current_node.children])
            print([c.value for c in self.current_node.children])
            print([c.satisfied for c in self.current_node.children])
            print(self.current_node.n_visit)
        #self.current_node.children = [best_node]
        self.current_node = best_node
        #self.traverse(self.current_node)
        return move
    def traverse(self, node, depth = 0):
        children = node.children
        if len(children) == 0:
            if depth == 0:
                print('empty tree')
            return -1
        print('depth', depth)
        print('scores', self.score(node))
        print('visits', node.n_visit)
        print( 'move, prior', [(c.parent_action, node.priors[i]) for i,c in enumerate(children)])
        print()
        for i, c in enumerate(children):
            self.traverse(c, depth+1)
        return -1


    def value_function(self, state, path, cp = 1):
        if path in self.state_data:
            if 'value' in self.state_data[path]:
                print('WARNING')
                print(state)
                print(mk_board(path), 'mk')
                print([path.count(str(i)) for i in range(7)])
                print(path)
                print(self.state_data[path]['value'],'val')
                self.state_data[path]['value'] = self.state_data[path]['value']+1
                print(self.state_data[path]['value'],'val after')
                print()
                return self.state_data[path]['value']
        X = torch.Tensor(np.array([state]))
        V = self.value_network(X)
        V = self.tanh(V)
        V = V.detach().numpy()[0][0]
        return V
    def correct_state(self, state):
        for child in self.current_node.children:
            if (child.state == state).sum() == len(state[0])*len(state):
                return child
    def score(self,  node):
        m = -1
        for j in range(len(node.children)-1):
            m = max(m, node.children[j].parent_action)
            if node.children[j].parent_action != m:
                print([c.parent_action for c in node.children])
                break
        scores = [self.eval(node, i) for i in range(len(node.children))]
        return scores
    def n_mcts(self, n, max_depth):
        self.searchspace = 0
        parent_v = self.current_node.Nv
        thresh = 1e-04
        delta = 0
        for i in range(n):
            s = [0]*len(self.current_node.available_moves())
            if len(self.current_node.children) == len(self.current_node.available_moves()):
                s= [c.value for c in self.current_node.children]
            node = self.current_node
            v, r, d, self.current_node = self.mcts(self.current_node, self.current_node.current_player, 0, max_depth)
            s2 = [c.value for c in self.current_node.children]
            mse = sum([(s[i]- s2[i])**2 for i in range(len(s))])/len(s)
            delta = 0.9*delta + 0.1*mse if i > 0 else mse

            val = [c.value for c in self.current_node.children]
            vis = self.current_node.n_visit
            max_val = np.argmax(val)
            max_visit = np.argmax(vis)
            if delta <thresh and vis[max_visit] > len(vis)*2:
                if max_val == max_visit or all([v == val[max_val]]):
                    break
            if vis[max_visit] > len(vis)+10*(len(vis) >1):
                if all([v == val[max_val]]) or val[max_val] == 1:
                    break
            diff = [vis[max_visit] - nv > n-i for nv in vis]
            diff[max_visit] = True
            if all(diff):
                break
            if vis[max_visit] > n/2:
                break
        self.avg_nr_evaluations.append(self.searchspace)
        if self.show:
            print(self.searchspace, 'nr mcts evaluations')
        return self.current_node

    def eval_pos(self, pos, board):
        ps = pattern_map[pos]
        s = 0
        for p in ps:
            s = 0
            #sum([board[y,x] for (y,x) in p])
            for i in range(4):
                s += board[p[i][0],p[i][1]]
        return s

    def rollout(self,current_player, child):
        child.Nv +=1
        state = copy.deepcopy(child.state)
        path = child.path
        pos = (child.ys[child.parent_action], child.parent_action)
        result = self.check_patterns(pos,state)
        #result = self.check_if_done(current_player, state, path)
        slots_left = [0]*7
        for m in list(range(state.shape[1])):
            for row in list(reversed(range(state.shape[0]))):
                if state[row][m] == 0:
                    slots_left[m] = row +1
                    break
        #child.N_visit +=1
        prob = [1/7]*7
        positions = [(slots_left[i]-1, i) for i in range(7) if state[0][i] == 0]
        lolz = [[self.eval_pos((y,x), state) for x in range(7)] for y in range(6)]
        scores = [[abs(lolz[pos[0]][pos[1]])+len(pattern_map[pos])/10, pos] for pos in positions]
        depth = abs(child.state).sum()
        d = depth
        while result == None and depth - d < 2:
            possible_moves = [i for i in range(7) if state[0][i] == 0]
            move = -1
            r = random.random()
            best = max(scores)
            move = best[1][1]
            if r > 0.83:
                best = random.choice(scores)
                move = best[1][1]
            state = next_state(state, current_player, move)
            slots_left[move] -= 1
            #result = self.check_if_done(current_player, state, path)
            result = self.check_patterns(best[1],state)
            patterns = pattern_map[best[1]]
            for p in patterns:
                for yx in p:
                    lolz[yx[0]][yx[1]] += current_player
            positions = [(slots_left[i]-1, i) for i in range(7) if state[0][i] == 0]
            scores = [[abs(lolz[pos[0]][pos[1]])+len(pattern_map[pos])/10, pos] for pos in positions]
            current_player = -current_player
            depth += 1
        if result == None:
            result = 0
        return result, depth
    def clean_data(self):
        keys = list(self.state_data)
        nv = []
        for i in range(5000):
            path = random.choice(keys)
            nv.append(self.state_data[path]['visits'])
        nv.sort()
        benchmark = nv[1000]
        nonos = []
        for path in self.state_data:
            if self.state_data[path]['visits'] <= benchmark and len(path) > 8:
                nonos.append(path)
        for path in nonos:
            del self.state_data[path]
        del keys
        del nonos

    def clean_tree(self):
        next_nodes = self.current_node.children
        self.current_node.n_visit = [0]*len(self.current_node.n_visit)
        del self.state_data
        self.state_data = {}
        queue = []
        if len(self.state_data) > 150000:
            self.clean_data()
        depth = 0
        while len(next_nodes) != 0:
            s = sum([c.Nv for c in next_nodes])
            l = len(next_nodes)
            for child in next_nodes:
                div = child.parent.Nv if child.parent.Nv > 0 else 1
                if child.Nv/div <depth**(3/7)/7 and depth >0:
                    child.children = []
                    child.n_visit = [0]*len(child.n_visit)
                if depth <10:
                    queue += child.children
                else:
                    child.children = []
                child.n_visit = [0]*len(child.n_visit)
            next_nodes = queue
            queue = []
            depth += 1
        next_nodes = self.current_node.children
        #self.current_node.Nv = 0
        self.current_node.Wr = 0
        #self.current_node.Nr = 0
        self.current_node.Wv = 0
        num_kids = len(next_nodes)
        depth = 0
        while len(next_nodes) != 0:
            l = len(next_nodes)
            for child in next_nodes:
                #child.Nv = 0
                child.Wr = 0
                #child.Nr = 0
                child.Wv = 0
                queue += child.children
            next_nodes = queue
            num_kids += len(next_nodes)
            queue = []
            depth += 1
        return self.current_node
    def find_node(self, node, state, max_depth = 2):
        if len(state) == 0:
            if self.check_if_done(1, node.state, node.path) != None:
                return node
            else:
                for c in node.children:
                    if self.check_if_done(1, c.state, c.path) != None:
                        return c
                return None
        if eq_board(node.state, state):
            return node
        next_nodes = self.current_node.children
        queue = []
        depth = 0
        while len(next_nodes) != 0 and depth != max_depth:
            for child in self.current_node.children:
                if eq_board(child.state, state):
                    return child
                else:
                    queue += child.children
            next_nodes = queue
            depth += 1
        return None

    def eval(self, node, c):
        lam = 0.5
        Nv = node.children[c].Nv
        Nr = node.children[c].Nr
        v = node.children[c].value
        Wv = node.children[c].Wv
        Wr = node.children[c].Wr
        if len(node.children) != len(node.priors):
            print('WARNING EVAL', len(node.children), len(node.priors))
            print(node.state)
            print('lolz')
            print(node.priors)
            print(mk_board(node.path), 'mk')
            print(node.path)
            print([node.path.count(str(i)) for i in range(7)])
            print(node.n_visit)
            print(c, 'child move')
            print([c.parent_action for c in node.children])
        if c >= len(node.priors):
            print('WARNING EVAL 2')
            print(node.state)
            print('lolz')
            print(node.priors)
            print(mk_board(node.path),'mk')
            print(node.path)
            print([node.path.count(str(i)) for i in range(7)])
            print(node.n_visit)
            print(c, 'child move')
            print([c.parent_action for c in node.children])
        Pv = sum([c.Nv for c in node.children])#node.Nv 
        fac = np.log(Pv) if Pv > 3 else 1.4
        return v + fac*node.priors[c]/(Nv+1)
    def get_prob(self, current_player, state):
        possible_moves = [i for i in range(7) if state[0][i] == 0]
        X = torch.Tensor(np.array([current_player*state]))
        prob = self.policy_network(X)
        prob = prob.detach().softmax(dim=1).numpy()[0].tolist()
        prob = np.array([prob[m] for m in possible_moves])
        noise = np.random.dirichlet([0.05]*len(possible_moves))
        prob = (1-0.15)*prob + 0.15*noise
        prob = prob/prob.sum()
        prob = prob.tolist()
        return prob
    def check_patterns(self, pos, board):
        ps = pattern_map[pos]
        cp = board[pos[0], pos[1]]
        for p in ps:
            s = 0
            #sum([board[y,x] for (y,x) in p])
            for i in range(4):
                s += cp*board[p[i][0],p[i][1]]
                if s < i +1:
                    break
            if s == 4:
                return cp
        if np.count_nonzero(board[0]) == board.shape[1]:
            return 0
        return None

    def mcts(self, node, current_player, depth, max_depth):
        node.priors = self.get_prob(current_player, node.state)
        node.Nv += 1
        N = 1
        d = depth
        if node.parent:
            i, _ = node.parent.get_child(node.parent_action)
            N = node.parent.n_visit[i]
        if depth == max_depth:
            result = None
            value = 0
            v = node.value
            action = -1
            idx = -1
            if result == None:
                if 0 in node.n_visit:
                    idx = node.n_visit.index(0)
                    action = list(node.available_moves())[idx]
                else:
                    possible_moves = list(node.available_moves())
                    action = possible_moves[np.random.randint(len(possible_moves))]
                    score = self.score(node)
                    idx = np.argmax(score)
                    action =possible_moves[idx]
                    idx, child = node.get_child(action)
                _ , child = node.get_child(action)
                if child == None:
                    idx = node.n_visit.index(0)
                    child_state = node.next_state(current_player, action)
                    child = MCTSNode(current_player*child_state, parent=node, parent_action=action, current_player = -current_player)
                    V =self.value_function(child_state, child.path, current_player)
                    prob = self.get_prob(-current_player, child_state)
                    child.prob = prob
                    child.value = V
                    node.add_child(child)
                node.n_visit[idx]+=1
                result, d = self.rollout(-current_player, child)
                result = current_player*result
                child.Wr += result
                child.Nr +=1
                #child.Nv += 1
                #print()
                lam = 1/(len(node.available_moves())+1)/(d-depth)
                
                value = child.value
                child.value = (1-lam)*child.value + lam*result
                if len(node.available_moves()) <= 1:
                    child.satisfied = True
                    node.satisfied = True
                    child.value = result
                    node.value = -child.value
                node.children[idx] = child
            else:
                print('HÄNDER ALDRIg')
                result = current_player*result
                v = result
                node.value = result
                node.W += result
            return -value, -result, d, node
        else:
            moves = node.available_moves()
            move_nr = -1
            if len(node.children) == len(moves):
                if len(node.children) != len(node.priors):
                    print('nu händer det')
                score = self.score(node)
                move_nr = np.argmax(score)
                node.n_visit[move_nr] +=1
            else:
                ns = []
                #print('new')
                for action in moves:
                    j, child = node.get_child(action)
                    if child == None:
                        self.searchspace += 1
                        next_state = node.next_state(current_player, action)
                        ns.append(next_state)
                        child = MCTSNode(current_player*next_state, parent=node, parent_action=action, current_player = -current_player)
                        V = self.value_function(next_state, child.path, current_player)
                    #P = self.policy_network(next_X)
                    #P = P.detach().softmax(dim=1).numpy()[0].tolist()
                        child.value = V
                        done = self.check_patterns((child.ys[action],action),next_state)
                        if done != None:
                            child.value = done
                            child.satisfied = True
                            node.satisfied = True
                            node.value = -done
                        node.add_child(child)
                score = self.score(node)
                move_nr = np.argmax(score)
                node.n_visit[move_nr] +=1
            next_state = node.children[move_nr].state
            #r = self.check_if_done(current_player, next_state, node.children[move_nr].path)
            m = node.children[move_nr].parent_action
            pos = (node.children[move_nr].ys[m], m)
            r = self.check_patterns(pos,next_state)
            #if r2 != r:
            #    print(next_state)
            #    print(r)
            #    print(r2)
            #    print(pos)
            #    print()
            r = current_player*r if r!= None else None
            v = None
            if r == None:
                self.searchspace += 1
                v,r, d, node.children[move_nr] = self.mcts(node.children[move_nr], -current_player, depth+1, max_depth)
                if not node.children[move_nr].satisfied:
                    edge = node.children[move_nr]
                    known_moves = sum([c.satisfied for c in edge.children])
                    lam = (known_moves+1)*len(edge.available_moves())*(d+1-depth)
                    if lam == 0:
                        print((known_moves+1), len(edge.available_moves()), d+1-depth)
                    #lam2= 1/(d+1-depth)
                    lam = 1/(lam+1)
                    #s = ((1-lam)*edge.Wv + lam*edge.Wr+ (1-lam*lam2)*v + lam*lam2*r)/edge.Nv
                    s = (1-lam)*v + lam*r
                    edge.Wr = edge.Wr + r
                    edge.Wv = edge.Wv + v if v != r else edge.Wv+ r
                    #s = (1-lam)*edge.Wv/edge.Nv + lam*edge.Wr/edge.Nv
                    edge.value = edge.value + 1/(edge.Nv+1)*(s - edge.value)
                else:
                    v = None
                    if node.children[move_nr].value == 1:
                        node.satisfied = True
                        node.value = -1
                        d = depth
            else:
                node.children[move_nr].value = r
                node.children[move_nr].Nv +=1
                node.children[move_nr].satisfied = True
                node.satisfied = True
                node.value = -r
                d = depth
        if all([c.satisfied for c in node.children]):
            node.satisfied = True
            val = max([c.value for c in node.children])
            node.value = -val
            v = val
            r = val
            d = depth
        if v == None:
            v = r
        return -v, -r, d, node
    def full_expansion(self, state):
        if self.MCTree == None:
            prior = policy_network(state)
            pred = self.model(X)
            prob = pred.detach().numpy()[0]
            node = MCTSNode(state, parent=None, parent_action=None, priors = prob, value = 0)
            self.currentNode = node
            self.MCTree = node
        return -1
    def check_if_done(self,current_player, board, path):
        if path in self.state_data:
            if 'result' in self.state_data[path]:
                self.state_data[path]['visits'] = self.state_data[path]['visits']+1
                return self.state_data[path]['result']
        result = None
        if np.count_nonzero(board[0]) == board.shape[1]:
            result = 0
        else:
            # Check win condition
            if self.is_win_state(board):
                result = 1 if len(path)%2 == 1 else -1
        if path in self.state_data:
            self.state_data[path]['result'] = result
            self.state_data[path]['visits'] = self.state_data[path]['visits'] +1
        else:
            self.state_data[path] = {'result': result, 'visits': 1}
        return result
    def is_win_state(self, board) -> bool:
        # Test rows
        board_shape = board.shape
        for i in range(board_shape[0]):
            for j in range(board_shape[1] - 3):
                value = sum(board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*board)]
        for i in range(board_shape[1]):
            for j in range(board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(board_shape[0] - 3):
            for j in range(board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(board)
        # Test reverse diagonal
        for i in range(board_shape[0] - 3):
            for j in range(board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def save_model(self, model_prefix: str = None):
        pass

class AlphaBeta(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent', show = False):
        super().__init__(env, name)
        self.show = show
        self.searchspace = 0
        self.avg_nr_evaluations = []

    def new_game(self):
        self.avg_nr_evaluations = []
        self.searchspace = 0

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')
        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        move = self.search(state, 1, 4)
        if self.show:
            print(move)
        if not available_moves:
            print('NO AVAILABLE')
            print(state)
            return 0
        return move
    def search(self, state, current_player, depth):
        moves = self.env.available_moves()

        scores = []
        alpha = -10
        evaluations = []
        for move in moves:
            pos, next_state = self.next_state(state, current_player, move)
            res = self.check_patterns(pos, next_state)
            self.searchspace += 1
            if abs(res) == 1:
                return move
            val = -self.alpha_beta(-10, -alpha, next_state,-current_player, depth-1)
            scores.append((val,move))
            evaluations.append(len(pattern_map[pos]))
        evaluations = [e/sum(evaluations)//0.001/1000 for e in evaluations]
        scores = [(scores[i][0]+ evaluations[i]//5/2, scores[i][1]) for i in range(len(scores))]
        random.shuffle(scores)
        if self.show:
            print(self.searchspace, 'nr states evaluated')
        self.avg_nr_evaluations.append(self.searchspace)
        self.searchspace = 0
        scores.sort(key = lambda x: -x[0])
        return scores[0][1]

    def alpha_beta(self, alpha, beta, state, current_player, depth):
        if depth == 0:
            return 0
        moves = [x for x in range(len(state[0])) if state[0][x] == 0]
        for move in moves:
            pos, next_state = self.next_state(state, current_player, move)
            self.searchspace += 1
            res = self.check_patterns(pos, next_state)
            if abs(res) == 1:
                return 1 + 0.01*depth
            val = -self.alpha_beta(-beta, -alpha,next_state, -current_player, depth-1)
            if val >= alpha:
                alpha = val
            if val >= beta:
                return val
        return alpha

    def check_patterns(self, pos, board):
        ps = pattern_map[pos]
        cp = board[pos[0], pos[1]]
        for p in ps:
            s = 0
            #sum([board[y,x] for (y,x) in p])
            for i in range(4):
                s += cp*board[p[i][0],p[i][1]]
                if s < i +1:
                    break
            if s == 4:
                return cp
        if np.count_nonzero(board[0]) == board.shape[1]:
            return 0.5
        return 0

    def next_state(self,state, current_player, action: int):

        board_copy = copy.deepcopy(state)
        if not state[0][action] == 0:
            raise Exception(
                'Unable to determine a valid move! Maybe invoke at the wrong time?'
            )
        #self.untried_actions = self.untried_actions.difference({action})
        # Check and perform action
        pos = (-1,action)
        for index in list(reversed(range(state.shape[0]))):
            if board_copy[index][action] == 0:
                board_copy[index][action] = current_player
                pos = (index, action)
                break
        return pos, board_copy

class SmartPlayer4(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent', model = None, show = False, epsilon = 1.0, smart_search = False):
        super().__init__(env, name)
        self.model = model
        self.show = show
        self.epsilon = epsilon
        self.smart_search = smart_search
        self.move_nr = 0
        model.eval()
        self.predictions = []
        self.states = []
        self.moves = []
        self.rewards = []
        self.graph = {}
        self.positions = []
        self.available_positions = []
    
    def q(self, reward, discount):
        res = [reward*discount**i for i in range(len(predictions))]
        res.reverse()
        return res
    def get_data(self, reward, last_move, opponent_path = {}):
        path = {}
        if reward == 1:
            path = self.traverse_all(self.positions[-1])
        if reward == -1:
            reward *= 3
            for i, pos in enumerate(self.positions):
                av_pos = self.available_positions[i]
                self.rewards[i][pos[1]] = reward
                for av in av_pos:
                    if av in opponent_path and pos[1] != av[1]:
                        discount = round(0.8**(len(self.positions)-i),3)
                        self.rewards[i][av[1]] = opponent_path[av]*discount
            self.rewards[-1][last_move] = 5
            self.rewards[-1][self.moves[-1]] = -5
        else:
            for i, pos in enumerate(self.positions):
                if pos in path:
                    self.rewards[i][pos[1]]  = path[pos]
                else:
                    self.rewards[i][pos[1]] = reward
            if reward == 1:
                self.rewards[-1][self.moves[-1]] = 5
        return self.states, self.rewards


    def new_game(self):
        self.states = []
        self.moves = []
        self.rewards = []
        self.positions = []
        self.available_positions = []
        self.graph = {}
    
    def set_model(self, new_model):
        new_model.eval()
        self.model = new_model

    def set_epsilon(self, eps):
        self.epsilon = eps

    def traverse_all(self, pos):
        for direction in self.graph[pos]:
            path = self.traverse(pos, direction)
            if len(path) == 4:
                for n in path:
                    path[n] = 2
                #len(self.graph[n][direction].keys() & self.graph.keys())
                return path
        return {}
    def traverse(self, pos, direction):
        next_nodes = self.graph[pos][direction]
        visited = {pos: 0}
        while len(next_nodes) > 0 and len(visited) < 4:
            layer2 = []
            for n in next_nodes:
                if n in self.graph and n not in visited:
                    layer2+=self.graph[n][direction]
                    visited[n] = 0
            next_nodes = layer2
        return visited
    def get_pos(self, move, board):
        y = -1
        for index in list(reversed(range(board.shape[0]))):
            if board[index][move] == 0:
                y = index
                break
        return (y, move)
    def pos_node(self, move, board):
        (y, x) = self.get_pos(move, board)
        diag_1 = [(y-1, x-1), (y+1, x+1)]
        diag_2 = [(y-1, x+1), (y+1, x-1)]
        hor = [(y, x-1), (y, x+1)]
        vert = [(y-1, x), (y+1, x)]
        neighbours = {'diag1': diag_1, 'diag2': diag_2, 'hor': hor, 'vert': vert}
        return (y,x), neighbours

    def get_next_action(self, state: np.ndarray) -> int:
        if abs(state).sum() <2:
            self.new_game()
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')
        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        X = torch.Tensor(np.array([state]))
        pred = self.model(X)
        prob = pred.detach().numpy()[0]
        exp = prob[list(available_moves)]
        move = self.eps_greedy(prob, available_moves, self.smart_search)
        self.moves.append(move)
        self.states.append(state.tolist())
        action = [0 if i in available_moves else -5 for i in range(7)]
        self.rewards.append(action)
        pos, neighbours = self.pos_node(move, state)
        self.graph[pos] = neighbours
        self.positions.append(pos)
        self.available_positions.append({self.get_pos(a, state) for a in list(available_moves)})
        if self.show:
            print(move)
        if not available_moves or prob.size == 0:
            print('NO AVAILABLE')
            print(state)
            return 0
        return move

    def eps_greedy(self, prob, available_moves, smart = False):
        move = random.choice(list(available_moves))
        f = [i in available_moves for i in range(7)]
        valid_moves = (torch.Tensor(prob).softmax(dim=0)*torch.Tensor(f)).numpy()
        temperature = 20 - 19.7*0.75**len(self.moves) #if len(self.moves) > 0 else 1.2
        valid_moves = np.power(valid_moves,temperature)
        valid_moves = valid_moves/valid_moves.sum()
        if self.show:
            print('MOVE PROBABILITY DETERMINED BY POLICY NET\n',[round(p,3) for p in valid_moves])
        r = random.random()
        temperature =1 #if len(self.moves) <8 else 0.62
        if r < self.epsilon:
            return np.argmax(valid_moves)
        idx = np.argsort(valid_moves).tolist()
        idx.reverse()
        cum_prob = 0
        r = random.random()
        for i in idx:
            cum_prob += valid_moves[i]
            if cum_prob >= r:
                move = i
                break
        return move
