import gym
import minerl
import logging 

from tensorforce.agents import Agent, DeepQNetwork, ProximalPolicyOptimization
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
    agent = Agent.create(agent='dqn', 
        environment=environment, 
        max_episode_timesteps = 8000
    )

    print("Created agent")
    runner = Runner(agent = agent, environment = environment)
    print("Created runner")
    runner.run(num_episodes=30)
    
    '''
    # Train for 300 episodes
    for _ in range(300):

        # Initialize episode
        print("Starting episode " + str(_))
        states = environment.reset()
        terminal = False
        total_reward = 0

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            total_reward += reward

        print(total_reward)
        print("Episode " + str(_) + " ended")
    '''

if __name__ == '__main__':
    main()