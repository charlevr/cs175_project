import gym
import minerl
import logging 

'''
Changed xml for treechop to include reward for touching wood. (find through going to the minerl library files in site-packages)

Regular treechop w/ exploration=.4 has potential with dueling dqn

TODO: maybe add in RewardForStrcuturCopying so that the agent can break the wood

exp=.4, rew touching wood=.3, rew getting wood = 1: about 3.15 total reward
'''

from tensorforce.agents import Agent, DeepQNetwork, ProximalPolicyOptimization, DuelingDQN
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

    agent = Agent.create(agent='dueling_dqn', 
        environment=environment,
        max_episode_timesteps = 8000,
        exploration = .3
    )

    sum_rewards = 0.0
    rewards_by_episode = []
    for _ in range(200):
        states = environment.reset()
        terminal = False
        print("Training episode " + str(_))
        while not terminal:
            actions = agent.act(states=states, evaluation=True)
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
            #print(actions)
        print("Sum reward so far: " + str(sum_rewards))
        rewards_by_episode.append((_, sum_rewards))
        print("Ending episode ", _)
    print(rewards_by_episode)
    print('Mean episode reward:', sum_rewards / 200)

    agent.close()
    environment.close()

if __name__ == '__main__':
    main()