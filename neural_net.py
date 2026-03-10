import torch
import torch.nn as nn 
import torch.nn.functional as F

import random
from collections import deque

class TrafficAgent(nn.Module):
    def __init__(self, state_dimensions, action_dimensions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dimensions, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dimensions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, size = 50000):
        self.buffer = deque(maxlen = size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states).float(),
            torch.tensor(actions),
            torch.tensor(rewards).float(),
            torch.tensor(next_states).float(),
            torch.tensor(dones).float()
        )

class Agent:
    def __init__(self, model, target_model, optimizer, buffer):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.buffer = buffer

        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.9998
        self.eps_min = 0.1

        self.batch_size = 64

        self.target_update = 1000
        self.train_steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        state = torch.tensor(state).float()

        with torch.no_grad():
            q = self.model(state)

        return torch.argmax(q).item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        
        target = rewards + self.gamma * next_q * (1 - dones)
        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

            print(f"Steps: {self.train_steps}, epsilon: {self.epsilon:.4f}, buffer: {len(self.buffer)}, Loss: {loss.item():.4f}")

        if self.train_steps % 100000 == 0:
            torch.save(self.model.state_dict(), "traffic_model_pressure.pt")

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

state_dim = 13
action_dim = 2

model = TrafficAgent(state_dim, action_dim)
target_model = TrafficAgent(state_dim, action_dim)
target_model.load_state_dict(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
buffer = ReplayBuffer()