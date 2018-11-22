import gym
import numpy as np
from dqn_agent import DQNAgent, DQNConfig


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    cfg = DQNConfig()
    cfg.model_savename = "mountain_car"
    cfg.plot_scores = True
    cfg.validate_completions = False
    cfg.winning_score = -199.0
    cfg.epochs = 1
    cfg.replay_batch_size = 16
    cfg.replay_period = 2

    agent = DQNAgent(env, cfg)
    agent.train()
    
    scores = agent.test_model(agent.model, True, 10)
