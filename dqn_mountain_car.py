import gym
import numpy as np
from dqn_agent import DQNAgent, DQNConfig


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    cfg = DQNConfig()
    cfg.model_savename = "mountain_car"
    cfg.plot_scores = True
    cfg.validate_completions = True

    agent = DQNAgent(env, cfg)
    agent.train()
    
    scores = agent.test_model(agent.model, True, 10)