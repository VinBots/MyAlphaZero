from utils import DotDict #other utilities

# Game settings
game_settings = DotDict({
    "board_size": (3,3),
    "N": 3
})

# Self-play training settings
game_training_settings = DotDict({
    "generations": 100,
    "self_play_iterations": 50,
    "data_augmentation_times": 1
})
# alpha = 10 / average legal moves 
# https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5 

# Self-play training settings
mcts_settings = DotDict({
    "explore_steps": 50,
    "temp": 1.0,
    "dir_enabled": True,
    "dir_eps": 0.25,
    "dir_alpha": 2.0,
})

# neural network settings
nn_training_settings = DotDict({
    "load_policy": False,
    "policy_path": "ai_ckp.pth",
    "ckp_folder":"../ckp",
    "lr": .005, 
    "weight_decay": 1.e-4,
    "buffer_size_target": 1000,
    "n_epochs": 1,
    "batch_size": 50
})
# set compet_freq at 0 for disabling the competition between current and trained network. 
# In this case the trained network replaces the current network at every generation

benchmark_competition_settings = DotDict({
    "compet_freq":0,
    "compet_rounds": 2,
    "net_compet_threshold": 0.0,
    "benchmark_freq": 5,
    "benchmark_rounds": 50,
    "mcts_iterations": 1000,
    "mcts_random_moves":0
})

# play settings
play_settings = DotDict({
    "explore_steps": 50,
    "temperature": 0.01                         
})


