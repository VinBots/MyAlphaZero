import time
import config as config
import policy_mod
import time
import games_mod # Games
import training_mod #neural network training
from replay_buffer_dict import ReplayBuffer #centralized buffer
from log_data import LogData #logging class for monitoring purposes
import cProfile

def main():
    
    log_data = LogData()
    log_data.add_chart("nn_loss", ["nn_loss.csv", ['iter', 'loss', 'value_loss', 'prob_loss']])
    log_data.add_chart("buffer", ["buffer.csv", ['iter', 'wins', 'losses', 'draws']])
    log_data.add_chart("compet", ["compet.csv",['iter', 'scores']])
    log_data.add_chart("bias_action", ["bias_action.csv",['iter', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7', 'pos8', 'pos9']])

    game=games_mod.ConnectN(config.game_settings)

    policy = policy_mod.Policy(config.nn_training_settings.policy_path, 
                            config.nn_training_settings, 
                            log_data)
    policy.save_weights()

    buffer = ReplayBuffer(config.nn_training_settings.buffer_size_target, 
                        config.nn_training_settings.batch_size, 
                        log_data)

    t0 = time.time()
    alpha_0 = training_mod.AlphaZeroTraining(
        config.game_settings, 
        config.game_training_settings,
        config.mcts_settings,
        config.nn_training_settings,
        config.benchmark_competition_settings,
        config.play_settings,
        policy,
        log_data)
    alpha_0.training_pipeline(buffer)
    t1 = time.time()

    print ("Total time (in sec): {}".format(t1 - t0))

if __name__ == "__main__":
    main()

    