import torch
from torch import nn
from DQN import DQN
import tetris_final as tetris
from collections import deque
from random import random, randint, sample
import numpy as np

width = 10
height = 20
block_size = 20
batch_size = 30
lr = 0.001
gamma = 0.99
initial_epsilon = 1
final_epsilon = 0
num_decay_epochs = 2000
num_epochs = 2500
save_interval = 500
replay_memory_size = 30000
saved_path = "model"


def train():
    env = tetris()
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    state = env.reset()
    replay_memory = deque(maxlen=replay_memory_size)
    epoch = 0
    while epoch < num_epochs:
        next_step = env.get_next_states()
        epsilon = final_epsilon + (max(num_decay_epochs - epoch, 0)
                                   * (initial_epsilon - final_epsilon) / num_decay_epochs)
        u = random()
        random_act = u <= epsilon
        next_action, next_states = zip(*next_step)
        next_states = torch.stack(next_states)
        model.eval()
        with torch.no_grad():
            predict = model(next_states)[:, 0]
        model.train()
        if random_act:
            index = randint(0, len(next_states) - 1)
        else:
            index = torch.argmax(predict).item()
        next_state = next_states[index, :]
        action = next_action[index]
        reward, done = env.step(action)
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
        else:
            state = next_state
            continue
        if len(replay_memory) < replay_memory_size / 10:
            continue
        epoch += 1

        batch = sample(replay_memory, min(len(replay_memory), batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        # 축 1개 추가
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(tuple(reward if done else reward + gamma * predict for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()
