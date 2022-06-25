from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.dqn_agent import DQNAgent
import sys

def format_state(state):
    player_x = round(state[0][0])
    player_y = round(state[0][1])
    invaders_x = [round(x) for x in state[1][0]]
    invaders_y = [round(y) for y in state[1][1]]
    print(f'Player: [{player_x},{player_y}] | Inv: [{invaders_x},{invaders_y}] | Bullet: {state[2]}')

def main(mode):

    game = SpaceInvaders(display=False)
    
    n_episodes = 200
    max_steps = 50
    gamma = 0.9
    alpha = 0.2
    epsilon = 0.8

    if mode == 'keyboard':
        agent = KeyboardController()
    elif mode == 'random':
        agent = RandomAgent(game.na)
    elif mode == 'dqn':
        agent = DQNAgent(game.na)
        agent.learn(n_episodes, max_steps, gamma, alpha, epsilon)
    else:
        print('Unknown agent. Starting as random agent.')
        agent = RandomAgent(game.na)
 
    state = game.reset()
    while True:
        action = agent.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)
        format_state(state)

if __name__ == '__main__' :
    if (len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        main('random')
