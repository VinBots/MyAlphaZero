import torch
import numpy as np

import policy_mod  # neural network


def test_final_positions(buffer):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    game_state1 = np.array([[-1, 1, -1], [0, 1, 0], [0, 0, 0]])

    game_state2 = np.array([[-1, 1, -1], [0, 1, -1], [0, 0, 0]])
    count1 = 0
    count2 = 0
    for state, _, _ in buffer.memory:
        if np.array_equal(state.astype(int), game_state1.astype(int)):
            count1 += 1
        if np.array_equal(state.astype(int), game_state2.astype(int)):
            count2 += 1
    print("Count test1: {}; test2 {}".format(count1, count2))

    frame1 = torch.tensor(game_state1, dtype=torch.float, device=device).unsqueeze(0)
    frame2 = torch.tensor(game_state2, dtype=torch.float, device=device).unsqueeze(0)

    policy_path = "ai_ckp.pth"
    policy = policy_mod.Policy(policy_path)
    policy.load_weights(policy_path)

    new_tensor = torch.stack((frame1, frame2))
    v, p = policy.forward_batch(new_tensor)
    v1 = v.detach().numpy()[0][0]
    v2 = v.detach().numpy()[1][0]

    p1 = p.detach().numpy()[0][7]
    p2 = p.detach().numpy()[1][7]

    print("Probabilities = {}, {}; Values = {}, {}".format(p1, p2, v1, v2))
