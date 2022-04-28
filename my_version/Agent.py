from abc import ABC, abstractmethod
import numpy as np
import torch
import random
import stats
import copy
from torch import nn

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


class SmartPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='Agent', model = None, show = False, epsilon = 1.0):
        super().__init__(env, name)
        self.model = model
        self.show = show
        self.epsilon = epsilon
        model.eval()

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        X = torch.Tensor(np.array([state.flatten()]))
        exp_reward = self.model(X)
        exp_reward = exp_reward.detach().numpy()[0]
        if self.show:
            print('exp_reward')
        if not available_moves or exp_reward.size == 0:
            print('NO AVAILABLE')
            print(state)
            return 0 
        move = self.random_move(exp_reward, available_moves, True)
        if self.show:
            print(move)
        return move

    def get_move(self, exp_reward, available_moves):
        move = -1
        print(exp_reward)
        while move not in available_moves and max(exp_reward) > -20:
            move = np.argmax(exp_reward)
            exp_reward[move] = -20
        print(move)
        return move, exp_reward

    def random_move(self, exp_reward, available_moves, smart = False):
        dumb = random.choice(list(available_moves))
        print('get move')
        move, exp_reward = self.get_move(exp_reward, available_moves)
        print('random move')
        while random.random() > self.epsilon and max(exp_reward) > -20:
            if not smart:
                return dumb
            move, exp_reward = self.get_move(exp_reward, available_moves)
        print()
        if move not in available_moves:
            move = dumb
        return move

    def save_model(self, model_prefix: str = None):
        pass


class SmartPlayer2(Player):
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

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed

        exp_reward = [-20,-20,-20,-20,-20,-20,-20]
        for action in available_moves:
            board_copy = self.env.next_state(action)
            X = torch.Tensor(np.array([board_copy.flatten()]))
            score = self.model(X).detach().numpy()[0][0]
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
    def prob_move(self, p, available_moves, exp_reward):
        valid_moves = list(available_moves)
        idx_sort = np.argsort(exp_reward)
        large_to_small = np.flip(idx_sort)
        move_probability = [0,0,0,0,0,0,0]
        for i, m in enumerate(valid_moves):
            idx = large_to_small[i]
            move_probability[i] = p*(1-p)**i
        return np.array(move_probability) / sum(move_probability), large_to_small

    def eps_greedy(self, exp_reward, available_moves, smart = False):
        dumb = random.choice(list(available_moves))
        prob = self.reward_variance(exp_reward, available_moves)
        p_adjusted = prob**(len(available_moves)/7)
        move_probability, idx_sort = self.prob_move(p_adjusted, available_moves, exp_reward)
        r = random.random()
        #print(idx_sort)
        cum_prob = 0
        if r > self.epsilon and not smart:
            return dumb
        for i, p in enumerate(move_probability):
            cum_prob += p
            move = idx_sort[i]
            if cum_prob >= r:
                break
        return move
    def get_move(self, exp_reward, available_moves):
        move = -1
        print(exp_reward)
        while move not in available_moves and max(exp_reward) > -20:
            move = np.argmax(exp_reward)
            exp_reward[move] = -20
        return move, exp_reward

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
        self.current_player = current_player
        self.opponent_value = opponent_value
        self.priors = priors
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.n_visit = [0]*len(self.available_moves())
        self.value = value
        self.W = 0
        self.N = 0
        return

    def is_valid_action(self, action: int) -> bool:
        return self.state[0][action] == 0

    def available_moves(self) -> frozenset:
        return {i for i in range(self.state.shape[1]) if self.is_valid_action(i)}

    def add_child(self, node):
        self.children.append(node)

    def get_child(self, action):
        for i, child in enumerate(self.children):
            if child.parent_action == action:
                return i, child
        return None

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

def next_state(state, current_player, action: int):
    board_copy = copy.deepcopy(state)
    #self.untried_actions = self.untried_actions.difference({action})
    # Check and perform action
    for index in list(reversed(range(state.shape[0]))):
        if board_copy[index][action] == 0:
            board_copy[index][action] = current_player
            break
    return board_copy 

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

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')
        if self.current_node == None:
            self.current_node = MCTSNode(state)
        else:
            new_node = self.correct_state(state)
            if new_node == None:
                new_node = MCTSNode(state)
            self.current_node.children = [new_node]
            self.current_node = new_node
        self.current_node = self.n_mcts(5, 3)
        print(self.current_node.n_visit)
        print([c.N for c in self.current_node.children], 'N visit')
        print([c.W for c in self.current_node.children], 'WINs ')
        print(self.current_node.children[np.argmax(self.current_node.n_visit)].n_visit)
        for c in self.current_node.children[np.argmax(self.current_node.n_visit)].children:
            print(c.value)
        print(self.current_node.priors)
        scores = self.score(self.current_node)
        print(scores)
        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        idx = np.argmax(scores)
        best_node = self.current_node.children[idx]
        move = best_node.parent_action
        print(move)
        self.current_node.children = [best_node]
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


    def value_function(self, state):
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
        scores = [self.eval(node, i) for i in range(len(node.children))]
        return scores
    def n_mcts(self, n, max_depth):
        for i in range(n):
            v, r, self.current_node = self.mcts(self.current_node, 1, 0, max_depth)
        return self.current_node
    def rollout(self,current_player, child):
        state = copy.deepcopy(child.state)
        result = self.check_if_done(current_player, state)
        #child.N_visit +=1
        while result == 0:
            possible_moves = [i for i in range(7) if state[0][i] == 0]
            move = possible_moves[np.random.randint(len(possible_moves))]
            for m in possible_moves:
                state = next_state(state, current_player, move)
                result = self.check_if_done(current_player, state)
                if result != 0:
                    move = m
                    break
            current_player = -current_player
        return result
    def eval(self, node, c):
        v = node.children[c].value
        return v + node.priors[c]/(node.n_visit[c]+0.1)
    def mcts(self, node, current_player, depth, max_depth):
        node.N += 1
        N = 1
        lam = 0.1
        if node.parent:
            i, _ = node.parent.get_child(node.parent_action)
            N = node.parent.n_visit[i]
        if depth == max_depth or self.check_if_done(current_player, node.state) != 0:
            result = self.check_if_done(current_player, node.state)
            v = node.value
            if result == 0:
                if 0 in node.n_visit:
                    idx = node.n_visit.index(0)
                    action = list(node.available_moves())[idx]
                    child_state = node.next_state(current_player, action)
                    V =self.value_function(child_state)
                    child = MCTSNode(current_player*child_state, parent=node, parent_action=action, current_player = -current_player)
                    node.n_visit[idx]+=1
                    result = self.rollout(-current_player, child)
                    result = current_player*result if result != 0.5 else 0.5
                    child.W += result
                    print('childw', child.W)
                    child.value = (1-lam)*V + lam*result
                    v = child.value
                    node.add_child(child)
                else:
                    possible_moves = list(node.available_moves())
                    action = possible_moves[np.random.randint(len(possible_moves))]
                    idx, child = node.get_child(action)
                    node.n_visit[idx]+=1
                    result = self.rollout(-current_player, child)
                    result = current_player*result if result != 0.5 else 0.5
                    child.W += result
                    child.value = (1-lam)*child.value+ lam*result
                    v = child.value
                node.children[idx] = child
            else:
                v = result
                node.value = result
                node.W += result
                for c in node.parent.children:
                    c.value = result
                print(result)
            return -v,result, node
        else:
            moves = node.available_moves()
            if len(node.children) == len(moves):
                score = self.score(node)
                move_nr = np.argmax(score)
                node.n_visit[move_nr] +=1
                v, r, node.children[move_nr] = self.mcts(node.children[move_nr], -current_player, depth+1, max_depth)
                node.children[move_nr].value = (1-lam)*node.children[move_nr].value+v*lam
                node.W += v
            else:
                X = torch.Tensor(np.array([current_player*node.state]))
                prob = self.policy_network(X)
                prob = prob.detach().softmax(dim=1).numpy()[0].tolist()
                node.priors = prob
                ns = []
                for action in moves:
                    child = node.get_child(action)
                    if child == None:
                        next_state = node.next_state(current_player, action)
                        ns.append(next_state)
                        V = self.value_function(next_state)
                    #P = self.policy_network(next_X)
                    #P = P.detach().softmax(dim=1).numpy()[0].tolist()
                        child = MCTSNode(current_player*next_state, parent=node, parent_action=action, value = V, current_player = -current_player)
                        node.add_child(child)
                score = self.score(node)
                move_nr = np.argmax(score)
                node.n_visit[move_nr] +=1

                v,r, node.children[move_nr] = self.mcts(node.children[move_nr], -current_player, depth+1, max_depth)
                node.children[move_nr].value = (1-lam)*node.children[move_nr].value+v*lam
                node.W +=r
        print('childw', node.W)
        return -v, r, node
    def full_expansion(self, state):
        if self.MCTree == None:
            prior = policy_network(state)
            pred = self.model(X)
            prob = pred.detach().numpy()[0]
            node = MCTSNode(state, parent=None, parent_action=None, priors = prob, value = 0)
            self.currentNode = node
            self.MCTree = node
        return -1
    def check_if_done(self,current_player, board):
        result = 0
        if np.count_nonzero(board[0]) == board.shape[1]:
            result = 0.5
        else:
            # Check win condition
            if self.is_win_state(board):
                result = 1 if current_player == 1 else -1
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
        valid_moves = valid_moves/valid_moves.sum()
        if self.show:
            print('MOVE PROBABILITY DETERMINED BY POLICY NET\n',[round(p,3) for p in valid_moves])
        r = random.random()
        temperature =1 #if len(self.moves) <8 else 0.62
        if r < self.epsilon*temperature:
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
