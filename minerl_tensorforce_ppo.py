import gym
import minerl
import logging 

from tensorforce.agents import Agent, DeepQNetwork, ProximalPolicyOptimization
from tensorforce.environments import Environment, OpenAIGym

def main():
    #Creates a log for MineRL
    #logging.basicConfig(level=logging.DEBUG)

    # Create the environment
    ENV_NAME = "MineRLTreechop-v0"

    # Pre-defined or custom environment
    env = gym.make(ENV_NAME)

    #Change action space and observation spae into something readable by tensorforce
    #These were taken form the OpenAIGym source code. 
    env_states = OpenAIGym.specs_from_gym_space(
        space=env.observation_space, ignore_value_bounds=True 
    )
    env_actions = OpenAIGym.specs_from_gym_space(
        space=env.action_space, ignore_value_bounds=True 
    )

    # Instantiate a Tensorforce agent
    #Can change the type of agent here but different types might have different configurations. 
    agent = ProximalPolicyOptimization(
        states = env_states,
        actions = env_actions,  
        max_episode_timesteps  = 5000,
        memory = 50000
    )

    #Starts the agent
    agent.initialize()

    # Train for 300 episodes
    for _ in range(300):
        # Initialize episode
        print("Episode " + str(_) + " training ... ")

        #Reset the environment so that agent is at starting point
        states = env.reset()

        #boolean that signifies if episode has ended or not
        terminal = False
        
        #keeps running sum of total reward
        total_reward = 0
        count = 0

        #while the episode is not done
        while not terminal:
            # Episode timestep

            #Agent does something based on its given algorithm
            actions = agent.act(states=states)

            #Get resulting state, reward, and status of agent given the action
            states, reward, terminal, _ = env.step(actions)

            #Get a little better based on whether or not the task was finished and how much the reward was
            agent.observe(terminal=terminal, reward=reward)

            #Update reward and count
            total_reward += reward
            count += 1

        #Housekeeping
        print("Episode " + str(_) + " ended.")
        print("Total reward: " + str(total_reward))
        print("Average reward: " + str( float(total_reward)/float(count)  ) )
        print()

        

    agent.close()
    env.close()

if __name__ == '__main__':
    main()