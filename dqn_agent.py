# GPU seems to underperform, likely because the batch sizes are so small
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import time
from collections import deque


class DQNConfig:
    """Configuration for the DQN agent"""
    
    gamma = 0.99
    # float: starting rate of random actions
    epsilon_max = 1.0
    # float: minimum epsilon value
    epsilon_min = 0.01
    # Rate of decay of epsilon
    epsilon_decay = 0.995
    # LR for both models
    learning_rate = 1e-3
    tau = 1e-3

    # int: replay memories every n steps of a trial
    replay_period = 4
    # int: batch size to replay
    replay_batch_size = 64
    # int: number of memories stored
    memory_size = 100000

    
    # float: any score at or below this is considered a failed trial
    winning_score = 100.0
    # bool: test the model after successful trials
    validate_completions = False
    # int: max steps per trial
    max_steps = 1500

    # bool: override the reward for a done frame
    done_reward_overridden = False
    # float: reward value for done frame (only active if done_reward_overridden)
    done_reward_value = 0
    # str: successful models are saved to storage/{model_savename}_{start_time}.model
    model_savename = 'dqn'

    nodes_per_layer = 256
    epochs = 1

    # bool: plot trial scores while training
    plot_scores = False

class DQNAgent:
    """
    Deep Q-Network designed to complete OpenAI Gym environments
    
    Following the OpenAI:DQN technique described by [Yesh Patel](https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c)
    """
    
    def __init__(self, env, config):
        self.start_time = int(time.time())
        self.config = config

        self.env = env
        self.action_count = self.__get_space_size(env.action_space)
        self.state_count = self.__get_space_size(env.observation_space)

        self.memory = deque(maxlen=config.memory_size)

        self.model = self.__create_model()
        self.targ_model = self.__create_model()
        self.__mean_text = False

    def train(self, trial_count=1000):
        """Train the model for the specified number of trials"""
        self.epsilon = self.config.epsilon_max
        scores = []
        score_means = []
        
        for trial in range(trial_count):
            start_time = time.time()
            score = self.__run_training_trial()

            duration = time.time() - start_time

            if score >= self.config.winning_score:
                print(f"{trial + 1}: {score:.1f} ✓ in {duration:.3f}s")
                self.model.save(f'storage/{self.config.model_savename}_{self.start_time}.model')
                if self.config.validate_completions:
                    self.test_model(self.model, False, 1)
            else:
                print(f"{trial + 1}: {score:.1f} in {duration:.3f}s")
                
            if self.config.plot_scores:
                scores.append(score)
                score_means.append(np.mean(scores[-100:]))

                self.__plot_scores(scores, score_means)
        
        return scores

    def test_model(self, model, render = False, round_count = 1):
        """Test a model in the environment"""

        scores = []
        for _ in range(round_count):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                if render:
                    self.env.render()
                action = np.argmax(model.predict(state.reshape(1, self.state_count))[0])
                state, reward, done, _ = self.env.step(action)
                score += reward
            print(f'Test Score: {score}')
            scores.append(score)
        
        return scores

    def __run_training_trial(self):
        """Run a single trial while training the model and return the score"""

        score = 0
        cur_state = self.env.reset()
        for step in range(self.config.max_steps):
            action = self.__act(cur_state)
            new_state, reward, done, _ = self.env.step(action)
            score += reward
            adj_reward = self.config.done_reward_value if done and self.config.done_reward_overridden else reward

            self.__step(step, cur_state, action, adj_reward, new_state, done)

            cur_state = new_state

            if done:
                break

        return score

    def __create_model(self):
        """Create a neural network to take the env state as input and output actions"""

        model = Sequential()
        model.add(Dense(self.config.nodes_per_layer, input_dim=self.state_count, activation='relu'))
        model.add(Dense(self.config.nodes_per_layer, activation='relu'))
        model.add(Dense(self.action_count))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.config.learning_rate))

        return model

    def __plot_scores(self, scores, score_means):
        """Plot a list of scores without blocking execution"""

        x = np.arange(0.0, len(scores), 1.0)

        plt.plot(x, scores, 'b', x, score_means, 'r--')

        if not self.__mean_text:
            self.__mean_text = plt.xlabel('')
            self.__window_line = plt.axvline(0, alpha=0.3, color='r')

        self.__mean_text.set_text('Mean Score (last 100): {0:0.2f}'.format(score_means[-1]))

        self.__window_line.set_xdata(max(0, len(scores) - 100))
        
        # Plot in interactive mode so it doesn't block
        plt.ion()
        plt.show()
        # GUI only runs when main thread sleeps
        plt.pause(0.01)

    def __get_space_size(self, space):
        """Get the size of a gym Space"""

        if type(space) is gym.spaces.Discrete:
            return space.n

        return space.shape[0]

    def __step(self, step, cur_state, action, reward, new_state, done):
        """Save the step's state and replay memories"""

        self.memory.append([cur_state, action, reward, new_state, done])

        if step % self.config.replay_period == 0:
            # ~.005 CPU, ~.008 sec GPU
            self.__replay_batched()

    def __act(self, state):
        """Determine the action to take for the current state"""

        # Decay epsilon, respecting the lower limit.
        self.epsilon = max(self.epsilon *self.config.epsilon_decay, self.config.epsilon_min)

        # Preform an 'exploratory' action or use one predicted by the NN model
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state.reshape(1, -1))[0])

    def __replay_batched(self):
        """Sample random memories, incorporate Q-learning and train the primary model"""

        if len(self.memory) < self.config.replay_batch_size:
            return
        
        # Take a random sample from the states we have saved
        samples = np.stack(random.sample(self.memory, self.config.replay_batch_size))

        # Group the states by column and reshape them for the NN
        states = np.vstack(samples[:, 0]).reshape(-1, self.state_count)
        new_states = np.vstack(samples[:, 3]).reshape(-1, self.state_count)

        # Predict the current solutions
        q_targets = self.model.predict(states)
        # Predict the future solutions and take the best
        next_q_targs = np.amax(self.targ_model.predict(new_states), 1)

        actions = samples[:, 1].astype(int)
        rewards = samples[:, 2]
        dones = samples[:, 4].astype(float)

        # Add the target Q to rewards not on a 'done' frame
        targets = rewards + (self.config.gamma * next_q_targs * (1 - dones))

        # Set the weighted best predicted target to the corresponding action in the current predictions
        for i in range(self.config.replay_batch_size):
            q_targets[i, actions[i]] = targets[i]
        
        # Fit the model using the adjusted q-targets
        self.model.fit(states, q_targets, epochs=self.config.epochs, verbose=0)

        self.__update_target_model()

    def __update_target_model(self):
        """Update the target model weights"""

        t_weights = np.array(self.targ_model.get_weights())
        m_weights = np.array(self.model.get_weights())

        self.targ_model.set_weights(t_weights * (1.0 - self.config.tau) + self.config.tau * m_weights)

    def __save_model(self, trial):
        """Save the primary model to storage"""

        model_f = f"storage/trial_{trial}.model"

        self.model.save(model_f)