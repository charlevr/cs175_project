import gym
import minerl
import logging 

from tensorforce.agents import Agent, DeepQNetwork, ProximalPolicyOptimization, AdvantageActorCritic
from tensorforce.environments import Environment, OpenAIGym
from tensorforce.execution import Runner

def main():
    #Creates a log for MineRL
    #logging.basicConfig(level=logging.DEBUG)

    # Create the environment
    ENV_NAME = "MineRLTreechop-v0"

    # Pre-defined or custom environment
    env = gym.make(ENV_NAME)

    environment = OpenAIGym(env)
    agent = Agent.create(agent='a2c', 
        environment=environment, 
        max_episode_timesteps = 8000
    )

    print("Created agent")
    runner = Runner(agent = agent, environment = environment)
    print("Created runner")
    runner.run(num_episodes=30)

if __name__ == '__main__':
    main()