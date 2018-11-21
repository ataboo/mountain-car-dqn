import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.tau = 1e-3

        self.model = self.create_model()
        self.targ_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(256, input_dim=state_shape[0], activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon *self.epsilon_decay, self.epsilon_min)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample

            q_targets = self.model.predict(state)
            if done:
                q_targets[0][action] = reward
            else:
                next_q_target = self.targ_model.predict(new_state)[0]
                q_targets[0][action] = reward + self.gamma * np.max(next_q_target)
            
            self.model.fit(state, q_targets, epochs=1, verbose=0)

    def update_target_model(self):
        self.targ_model.set_weights(self.model.get_weights())

    def save_model(self, trial):
        model_f = f"storage/trial_{trial}.model"

        self.model.save(model_f)

def test_model(env, model, hm_rounds = 1):
    scores = []
    for round in range(hm_rounds):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = np.argmax(model.predict(state.reshape(1, 2))[0])
            state, reward, done, _ = env.step(action)
            score += reward
        print(f'Test Score: {score}')
        scores.append(score)
    
    return scores

def main():
    env = gym.make("MountainCar-v0")

    trials = 2000
    trial_len = 500

    dqn_agent = DQN(env = env)
    
    for trial in range(trials):
        score = 0
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done else -20

            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            if step % 4 == 0:
                dqn_agent.replay()

            cur_state = new_state

            if done:
                break
            
        dqn_agent.update_target_model()
        print(f'Score: {score}')

        if step >= 199:
            print("Failed to complete trial {}".format(trial+1))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.model.save('storage/chicken_dinner.model')
            test_model(env, dqn_agent.model)

if __name__ == "__main__":
    main()