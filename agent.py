from actor import Actor
from critic import Critic
import numpy as np
from numpy.random import choice
import random
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
    
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self, batch_size=32):
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
class Agent:
    def __init__(self, state_size, batch_size, final = False):
        self.state_size = state_size
        self.action_size = 3
        self.buffer_size = 1000000
        self.gamma = 0.99
        self.tau = 0.001
        self.batch_size = batch_size
        self.buying_data = []
        self.final = final
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)    

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights()) 
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
    def act(self, state):
        options = self.actor_local.model.predict(state)
        self.final_state = state
        if not self.final:
            return choice(range(3), p = options[0])
        return np.argmax(options[0])
    
    def step(self, action, reward, next_state, done):
        self.memory.add(self.final_state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            self.final_state = next_state

    def learn(self, experiences):               
        states = np.vstack([exp.state for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,self.state_size)
        actions = np.vstack([exp.action for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([exp.reward for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([exp.done for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([exp.next_state for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,self.state_size)

        actions_next_state = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next_state = self.critic_target.model.predict_on_batch([next_states, actions_next_state])
        
        Q_targets = rewards + self.gamma * Q_targets_next_state * (1 - dones)
        self.critic_local.model.train_on_batch(x = [states, actions], y=Q_targets)
        
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),(-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])
        self.soft_update(self.critic_local.model, self.critic_target.model)  
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weight_values = np.array(local_model.get_weights())
        target_weight_values = np.array(target_model.get_weights())
        assert len(local_weight_values) == len(target_weight_values)
        new_weight_values = self.tau * local_weight_values + (1 - self.tau) * target_weight_values
        target_model.set_weights(new_weight_values)
