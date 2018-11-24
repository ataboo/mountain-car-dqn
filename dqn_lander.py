import gym
from dqn_agent import DQNAgent, DQNConfig
import tensorflow as tf
import numpy as np

def train_lander():
    env = gym.make("LunarLander-v2")

    cfg = DQNConfig()
    cfg.model_savename = "lunar_lander"
    cfg.plot_scores = True
    cfg.validate_completions = True
    cfg.winning_score = 200.0
    cfg.replay_period = 1
    cfg.replay_batch_size = 16
    cfg.max_steps = 1500
    cfg.done_reward_overridden = False
    cfg.epsilon_decay = 0.95

    agent = DQNAgent(env, cfg)
    agent.train()

    scores = agent.test_model(agent.model, True, 10)

def replay_lander_model():
    env = gym.make("LunarLander-v2")

    model = tf.keras.models.load_model('./lander_keeper_800.model')    

    scores = DQNAgent(env, DQNConfig()).test_model(model, True, 10)

    print('================================')
    print(f"Avg Score: {np.mean(scores)}")

if __name__ == "__main__":
    # train_lander()

    replay_lander_model()