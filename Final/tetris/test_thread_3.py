import torch
import tetris_final as tetris

width = 10
height = 20
block_size = 30
fps = 300


def test():
    model = torch.load("ReVR", map_location=lambda storage, loc: storage)
    model.eval()
    tetris.reset()
    while True:
        next_steps = tetris.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = tetris.step(list(action))

        if done:
            print(tetris.f.clearCnt)
            break


if __name__ == "__main__":
    test()
