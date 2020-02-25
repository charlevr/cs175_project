import gym
import minerl
import logging 

from tensorforce.agents import Agent, DeepQNetwork, ProximalPolicyOptimization
from tensorforce.environments import Environment, OpenAIGym

def main():
    logging.basicConfig(level=logging.DEBUG)
    # Create the environment
    ENV_NAME = "MineRLTreechop-v0"
    # Pre-defined or custom environment
    env = gym.make(ENV_NAME)
    env_states = OpenAIGym.specs_from_gym_space(
        space=env.observation_space, ignore_value_bounds=True 
    )
    env_actions = OpenAIGym.specs_from_gym_space(
        space=env.action_space, ignore_value_bounds=True 
    )

    # Instantiate a Tensorforce agent
    agent = ProximalPolicyOptimization(
        states = env_states,
        actions = env_actions,  
        max_episode_timesteps  = 1000
    )
    agent.initialize()
    # Train for 300 episodes
    for _ in range(300):
        # Initialize episode
        print("Episode " + str(_) + " training ... ")
        states = env.reset()
        terminal = False
        total_reward = 0
        count = 0

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, reward, terminal, _ = env.step(actions)
            agent.observe(terminal=terminal, reward=reward)
            total_reward += reward
            count += 1

        print("Episode " + str(_) + " ended.")
        print("Total reward: " + str(total_reward))
        print("Average reward: " + str( float(total_reward)/float(count)  ) )
        print()

        

    agent.close()
    env.close()

if __name__ == '__main__':
    main()