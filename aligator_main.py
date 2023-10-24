import numpy as np
import matplotlib.pyplot as plt
from aligator_q_learning import step, Environment, GAME_MATRIX
import os

SHOW_EVERY = 500

if not os.path.exists("qtables"):
    os.makedirs("qtables")
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("gif"):
    os.makedirs("gif")
if not os.path.exists("frames"):
    os.makedirs("frames")

def train(env, num_episodes, max_frames):
    aggr_episode_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
    rewards_all_episodes = []
    states = []

    for i in range(num_episodes):
        env.reset()
        env.epsilon = max(env.eps_end, env.epsilon*env.eps_dec)
        # data for plotting
        rewards_current_episode = 0
        # add the starting position
        states.append((env.initial_position))

        while True:
            state, action, reward, Q, done = step(env)

            # data for plotting
            rewards_current_episode += reward
            states.append((state))
            if done or env.current_frame >= max_frames:
                break

        rewards_all_episodes.append(rewards_current_episode)
        if i % SHOW_EVERY == 0:
            np.save(f"qtables/qtable_{i}", Q)
            average_reward = sum(
                rewards_all_episodes[-SHOW_EVERY:])/len(rewards_all_episodes[-SHOW_EVERY:])
            aggr_episode_rewards['ep'].append(i)
            aggr_episode_rewards['avg'].append(average_reward)
            aggr_episode_rewards['min'].append(
                min(rewards_all_episodes[-SHOW_EVERY:]))
            aggr_episode_rewards['max'].append(
                max(rewards_all_episodes[-SHOW_EVERY:]))

    return aggr_episode_rewards, states


def observe(env):
    env.reset()
    while True:
        env.visualize()
        state, action, reward, Q, done = step(env)
        # create a copy of the game matrix with all positions as 0
        if done:
            print("DONE IN: " + str(env.current_frame - 1) + " STEPS" + "\n")
            env.visualize()
            break
    env.render()


if __name__ == '__main__':
    num_episodes = 10000
    max_frames = 100
    env = Environment(GAME_MATRIX, epsilon=1, gamma=0.99, lr=0.1)

    aggr_episode_rewards, states = train(
        env, num_episodes=num_episodes, max_frames=max_frames)
    print("Train done!")

    # Get the positions array
    positions = np.array([(pos['x'], pos['y']) for pos in states])
    # Plot positions as heatmap with viridis color map
    plt.hist2d(positions[:, 0], positions[:, 1], cmap='viridis', range=[
        [-0.5, 6.5], [-0.5, 3.5]], bins=[7, 4])
    # Add a color bar
    plt.colorbar()
    # Add labels
    plt.xlabel('X')
    plt.ylabel('Y')
    # Invert the Y-axis to have 0 at the top and 3 at the bottom
    plt.gca().invert_yaxis()
    # Adjust the axis limits to make sure squares have width and height of 1
    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 3.5)
    plt.gca().invert_yaxis()
    # Show the plot
    plt.show()
    plt.close()

    # Plot the rewards
    plt.plot(aggr_episode_rewards['ep'],
             aggr_episode_rewards['avg'], label="avg")
    plt.plot(aggr_episode_rewards['ep'],
             aggr_episode_rewards['min'], label="min")
    plt.plot(aggr_episode_rewards['ep'],
             aggr_episode_rewards['max'], label="max")
    plt.legend(loc=4)
    plt.ylim(-100, 100)
    plt.show()
    plt.close()

    observe(env)
    print("Observe done!")
