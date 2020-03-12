import gym
import minerl

def main():
    env = gym.make("MineRLTreechop-v0")

    done = False
    env.reset()
    net_reward = 0


    while not done:
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print(action)
        print("Total reward: ", net_reward)
        
if __name__ == '__main__':
    main()