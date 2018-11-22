import gym
from dqn_agent import DQNAgent, DQNConfig


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    cfg = DQNConfig()
    cfg.model_savename = "lunar_lander"
    cfg.plot_scores = True
    cfg.validate_completions = True

    agent = DQNAgent(env, cfg)
    agent.train()
    
    scores = agent.test_model(agent.model, True, 10)