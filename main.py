import numpy as np
import matplotlib.pyplot as plt
from q_learning import step, Environment, GAME_MATRIX
import os
import cv2

SHOW_EVERY = 500
NUM_EPISODES = 10000
MAX_FRAMES = 100

if not os.path.exists("qtables"):
    os.makedirs("qtables")
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("gif"):
    os.makedirs("gif")
if not os.path.exists("frames"):
    os.makedirs("frames")
if not os.path.exists("plots"):
    os.makedirs("plots")


def train(env, num_episodes, max_frames):
    aggr_episode_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
    rewards_all_episodes = []
    states = []

    for episode in range(num_episodes):
        env.reset()
        env.epsilon = max(env.eps_end, env.epsilon*env.eps_dec)
        # data for plotting
        rewards_current_episode = 0
        # add the starting position
        states.append((env.initial_position))

        while True:
            if episode % SHOW_EVERY == 0:
                game_image = env.visualize()
                cv2.imshow("game", np.array(game_image))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    continue

            state, action, reward, Q, done = step(env)

            # data for plotting
            rewards_current_episode += reward
            states.append((state))
            if done or env.current_frame >= max_frames:
                if episode % SHOW_EVERY == 0:
                    game_image = env.visualize()
                    cv2.imshow("game", np.array(game_image))
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                break

        rewards_all_episodes.append(rewards_current_episode)
        if episode % SHOW_EVERY == 0:
            np.save(f"qtables/qtable_{episode}", Q)
            average_reward = sum(
                rewards_all_episodes[-SHOW_EVERY:])/len(rewards_all_episodes[-SHOW_EVERY:])
            aggr_episode_rewards['ep'].append(episode)
            aggr_episode_rewards['avg'].append(average_reward)
            aggr_episode_rewards['min'].append(
                min(rewards_all_episodes[-SHOW_EVERY:]))
            aggr_episode_rewards['max'].append(
                max(rewards_all_episodes[-SHOW_EVERY:]))

    return aggr_episode_rewards, states


def observe(env):
    env.reset()
    game_image = env.visualize()
    game_image.save(f"frames/frame_{env.current_frame}.png")
    while True:

        state, action, reward, Q, done = step(env)
        # create a copy of the game matrix with all positions as 0
        if done:
            print("Observe done in " + str(env.current_frame - 1) + " steps!")
            game_image = env.visualize()
            game_image.save(f"frames/frame_{env.current_frame}.png")
            break
    env.render()


if __name__ == '__main__':
    env = Environment(GAME_MATRIX, epsilon=1, gamma=0.99, lr=0.1)

    aggr_episode_rewards, states = train(
        env, num_episodes=NUM_EPISODES, max_frames=MAX_FRAMES)
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
    plt.savefig('plots/heatmap_positions.png')
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
    plt.savefig('plots/rewards_vs_episodes.png')
    plt.close()

    observe(env)