import gym
from dqn_agent import DQNAgent, DQNConfig


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    cfg = DQNConfig()
    cfg.model_savename = "lunar_lander"
    cfg.plot_scores = True
    cfg.validate_completions = True
    cfg.winning_score = 200.0
    cfg.replay_period = 4
    cfg.replay_batch_size = 64
    cfg.max_steps = 2000
    cfg.done_reward = 0.0

    agent = DQNAgent(env, cfg)
    agent.train()
    
    scores = agent.test_model(agent.model, True, 10)