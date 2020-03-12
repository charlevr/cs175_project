import minerl
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (64, 64)
WINDOW_LENGTH = 3

class MCProcessor(Processor):
    def process_observation(self, obs):
        assert observation.ndim == 3  # (height, width, channel)
        obs = obs.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(obs)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def main():
    #logging.basicConfig(level=logging.DEBUG)
    ENV_NAME = "MineRLTreechop-v0"
    env = gym.make(ENV_NAME) # A MineRLTreechop-v0 env
    nb_actions = 9

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape = (64, 64, 3)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=2500, visualize=True, verbose=2)
    print(model.summary())

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    #dqn.test(Monitor(env, '.'), nb_episodes=5, visualize=True)

if __name__ == '__main__':
    main()