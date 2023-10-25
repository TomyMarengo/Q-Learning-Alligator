import numpy as np
from enum import Enum
from PIL import Image

IMAGE_SIZE = (50, 50)

EMOJI_HOLE = Image.open('images/hole.png').resize(IMAGE_SIZE)
EMOJI_WIN = Image.open('images/coins.png').resize(IMAGE_SIZE)
EMOJI_COIN = Image.open('images/coin.png').resize(IMAGE_SIZE)
EMOJI_ALLIGATOR = Image.open('images/alligator.png').resize(IMAGE_SIZE)
EMOJI_GRASS = Image.open('images/grass.png').resize(IMAGE_SIZE)


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# 0 = Grass
# 1 = Coin
# -10 = Hole
# 100 = Win
GAME_MATRIX = np.array([
    [0, 100, -10, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0]
])

global Q
Q = np.random.uniform(low=-1, high=1, size=GAME_MATRIX.shape + (len(Action),))


class Environment:
    def __init__(self, game_matrix, gamma, lr, epsilon=1, eps_end=0.05, eps_dec=0.990):
        self.initial_position = {'x': 6, 'y': 3}
        self.aligator_position = self.initial_position.copy()
        self.game_matrix = game_matrix
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.current_frame = 1

    def state(self):
        return self.aligator_position

    def set_state(self, state):
        self.aligator_position = state

    def reset(self):
        self.aligator_position = self.initial_position.copy()
        self.current_frame = 1
        return self.state()

    def visualize(self):
        # Create a new image with the same shape as the game matrix
        game_image = Image.new('RGBA', (GAME_MATRIX.shape[1]*50, GAME_MATRIX.shape[0]*50), (255, 255, 255, 255))

        # Draw the emojis for each value in the game matrix
        for y in range(GAME_MATRIX.shape[0]):
            for x in range(GAME_MATRIX.shape[1]):
                if self.state()['x'] == x and self.state()['y'] == y:
                    game_image.paste(EMOJI_ALLIGATOR, (x*50, y*50), EMOJI_ALLIGATOR)
                elif GAME_MATRIX[y][x] == -10:
                    game_image.paste(EMOJI_HOLE, (x*50, y*50), EMOJI_HOLE)
                elif GAME_MATRIX[y][x] == 100:
                    game_image.paste(EMOJI_WIN, (x*50, y*50), EMOJI_WIN)
                elif GAME_MATRIX[y][x] == 1:
                    game_image.paste(EMOJI_COIN, (x*50, y*50), EMOJI_COIN)
                elif GAME_MATRIX[y][x] == 0:
                    game_image.paste(EMOJI_GRASS, (x*50, y*50), EMOJI_GRASS)
        # Save the image as a PNG file
        return game_image

    def render(self):
        frame_paths = [
            f"frames/frame_{i}.png" for i in range(1, self.current_frame + 1)]
        frames = [Image.open(frame_path) for frame_path in frame_paths]
        frames[0].save('gif/game_animation.gif', save_all=True,
                    append_images=frames[1:], optimize=False, duration=500, loop=0)


def take_action(env, action):
    aligator = env.state().copy()
    if action == Action.UP:
        aligator['y'] -= 1
    elif action == Action.DOWN:
        aligator['y'] += 1
    elif action == Action.LEFT:
        aligator['x'] -= 1
    elif action == Action.RIGHT:
        aligator['x'] += 1
    return aligator


def compute_reward(env, next_state):
    return GAME_MATRIX[next_state['y']][next_state['x']] - 0.1 * env.current_frame


def is_done(state):
    # Grass or just a coin
    return GAME_MATRIX[state['y']][state['x']] not in [0, 1]


def possible_actions(state):
    actions = []
    if state['y'] > 0:
        actions.append(Action.UP)
    if state['y'] < GAME_MATRIX.shape[0] - 1:
        actions.append(Action.DOWN)
    if state['x'] > 0:
        actions.append(Action.LEFT)
    if state['x'] < GAME_MATRIX.shape[1] - 1:
        actions.append(Action.RIGHT)
    return actions


def step(env):
    state = env.state()

    if np.random.rand() < env.epsilon:
        action = np.random.choice(possible_actions(state))
    else:
        possible_act = possible_actions(state)
        q_values = Q[state['y'], state['x'], [a.value for a in possible_act]]
        max_q_value_index = np.argmax(q_values)
        action = possible_act[max_q_value_index]

    next_state = take_action(env, action)
    reward = compute_reward(env, next_state)

    if not is_done(next_state):
        next_possible_act = possible_actions(next_state)
        q_values = Q[next_state['y'], next_state['x'],
                    [a.value for a in next_possible_act]]
        next_max_q_value_index = np.argmax(q_values)
        next_action = next_possible_act[next_max_q_value_index]
        old = (1 - env.lr) * Q[state['y'], state['x'], action.value]
        new = env.lr * (reward + env.gamma *
                        Q[next_state['y'], next_state['x'], next_action.value])
        Q[state['y'], state['x'], action.value] = old + new
    else:
        Q[state['y'], state['x'], action.value] = reward

    env.set_state(next_state)
    env.current_frame += 1

    return next_state, action, reward, Q, is_done(next_state)
