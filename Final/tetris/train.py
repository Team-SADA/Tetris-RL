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
final_epsilon = 1/500
num_decay_epochs = 2000
num_epochs = 5000
save_interval = 250
replay_memory_size = 30000
saved_path = "model"


def train():
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    state = tetris.reset()
    replay_memory = deque(maxlen=replay_memory_size)
    epoch = 0
    while epoch < num_epochs:
        next_step = tetris.get_next_states()
        epsilon = final_epsilon + (max(num_decay_epochs - epoch, 0)
                                   * (initial_epsilon - final_epsilon) / num_decay_epochs)
        u = random()
        random_act = u <= epsilon
        next_action, next_states = zip(*next_step.items())
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
        action = list(next_action[index])
        reward, done = tetris.step(action)
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = tetris.score
            final_tetrominoes = tetris.tetromino
            final_cleared_lines = tetris.f.clearCnt
            state = tetris.reset()
        else:
            state = next_state
            continue
        if len(replay_memory) < replay_memory_size / 10:
            continue
        epoch += 1

        batch = sample(replay_memory, min(len(replay_memory), batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        # 축 1개 추가
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + gamma * prediction
                  for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}/{num_epochs}, Score: {final_score}, "
              f"Tetrominoes {final_tetrominoes}, Cleared lines: {final_cleared_lines}")

        if epoch > 0 and epoch % save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(saved_path, epoch))
            with open(saved_path+"info", "a") as f:
                f.write(f"Epoch: {epoch}, Cleared Lines: {final_cleared_lines}, Total Score: {final_score}\n")


if __name__ == "__main__":
    saved_path = "model_4"
    train()
