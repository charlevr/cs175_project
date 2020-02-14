import gym
import minerl
import logging 

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def main():
    logging.basicConfig(level=logging.DEBUG)
    # Create the environment
    ENV_NAME = "MineRLTreechop-v0"
    env = gym.make(ENV_NAME)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    # Define the model
    model = PPO2(MlpPolicy, env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=25000)

    # After training, watch our agent walk
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()