import gym
import minerl
import logging 

from tensorforce.agents import Agent, DeepQNetwork, ProximalPolicyOptimization
from tensorforce.environments import Environment, OpenAIGym

'''
class MineRL_Env(Environment):
    def __init__(self, minerl_env):
        super().__init__()
        self.minerl_env = minerl_env

    def states(self):
        return self.minerl_env.observation_space.spaces

    def actions(self):
        return self.minerl_env.action_space.spaces

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional
    def close(self):
        self.minerl_env.close()

    def reset(self):
        state = self.minerl_env.reset()
        return state

    def execute(self, actions):
        next_state, reward, terminal, _ = self.minerl_env.step(actions)
        return next_state, terminal, reward
'''
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
        states = env.reset()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, reward, terminal, _ = env.step(actions)
            agent.observe(terminal=terminal, reward=reward)
            print(reward)

    agent.close()
    env.close()

if __name__ == '__main__':
    main()