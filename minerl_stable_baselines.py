import gym
import minerl
import logging 

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2, A2C

def main():
    #logging.basicConfig(level=logging.DEBUG)
    # Create the environment
    ENV_NAME = "MineRLTreechop-v0"
    env = gym.make(ENV_NAME)
    env.action_space = 
    print(check_env(env))


    # Define the model
    model = A2C(CnnPolicy, env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=8000)

if __name__ == '__main__':
    main()