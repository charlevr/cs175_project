import minerl
import tensorflow as tf

#minerl.data.download('./', experiment='MineRLTreechop-v0')

def main():
    data = minerl.data.make(
        'MineRLTreechop-v0',
        data_dir='./')

    # Iterate through a single epoch gathering sequences of at most 32 steps
    for current_state, action, reward, next_state, done \
        in data.sarsd_iter(
            num_epochs=1, max_sequence_len=1):

            # Print the POV @ the first step of the sequence
            print(current_state['pov'].shape)
            print(action["attack"])

            # Print the final reward pf the sequence!
            print(reward[-1])

            # Check if final (next_state) is terminal.
            print(done[-1])

            # ... do something with the data.
            print("At the end of trajectories the length"
                "can be < max_sequence_len", len(reward))
            break

if __name__ == '__main__':
    main()