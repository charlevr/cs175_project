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

    agent = Agent.create(agent='ac', 
        environment=environment,
        max_episode_timesteps = 8000,
        exploration = .03,
        critic_optimizer = 'evolutionary'
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