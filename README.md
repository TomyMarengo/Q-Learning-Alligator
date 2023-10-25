# Q-Learning-Alligator

This is a game about an alligator that wants to collect coins. The goal of the game is to help the alligator collect all the coins before finishing the game. Currently, the alligator wants to collect the large handful of coins (100 of rewards).

## How to Run the Game

To run the game, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project folder.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the game using `python main.py`.

## Parameters

The game has the following parameters:

- `num_episodes`: The number of episodes to run the game for.
- `max_frames`: The maximum number of steps to run the game for.
- `lr (alpha)`: The learning rate for the Q-learning algorithm.
- `gamma`: The discount factor for the Q-learning algorithm.
- `epsilon`: The probability of taking a random action instead of the optimal action.
- `epsilon_dec`: The decay rate for epsilon.
- `epsilon_end`: The minimum value for epsilon.
- `show_every`: The number of episodes to wait before saving q-table and aggregating statistics.

## Project Folders

The project has the following folders:

- `images`: Contains the images used in the game.
- `qtables`: Contains the q-tables generated by the game.
- `gif`: Contains the gif generated by the game.
- `frames`: Contains the frames used to generate the gif.
- `root`: Contains the main files for the game, requirements.txt, etc.
