import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims,fc2_dims,n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims  = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 =  nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self,observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent(object):
    def __init__(self,gamma, epsilon, lr, input_dims, batch_size, n_actions,
                            max_mem_size = 30000, eps_end = 0.01, eps_dec = 5e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(lr,n_actions = self.n_actions, input_dims = input_dims,fc1_dims = 256, fc2_dims = 256)
        self.state_memory = np.zeros((self.mem_size,*input_dims))
        self.new_state_memory = np.zeros((self.mem_size,*input_dims))
        self.action_memory = np.zeros((self.mem_size,self.n_actions),
                                            dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.uint8)

    def store_transition(self,state,action,reward,new_state,terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - terminal
        self.new_state_memory[index] = new_state
        self.mem_cntr += 1
    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.Q_eval.forward(observation)
            action = T.argmax(actions).item()
        return action

    def experience_replay(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                        else self.mem_size
            batch = np.random.choice(max_mem,self.batch_size)

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values  = np.array(self.action_space, dtype=np.int32)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)
            batch_index = np.arange(self.batch_size, dtype = np.int32)
            q_target[batch_index, action_indices] = reward_batch + \
                        self.gamma*T.max(q_next,dim = 1)[0]*terminal_batch
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                            self.eps_min else self.eps_min
            loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.Q_eval.state_dict(), 'trained_model.pt')
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.Q_eval.load_state_dict(T.load('trained_model.pt', map_location='cpu'))
