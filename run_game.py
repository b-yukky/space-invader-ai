from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.dqn_agent import DQNAgent
from classes import epsilon_profile
from classes import networks
import sys
import torch
import time
import pygame

def format_state(state):
    player_x = round(state[0])
    invaders_x = [round(x) for x in state[1][0]]
    invaders_y = [round(y) for y in state[1][1]]
    print(f'Player: [{player_x}] | Inv: [{invaders_x},{invaders_y}] | Bullet: {state[2]}')

def main(mode):

    game = SpaceInvaders(display=True)
    
    n_episodes = 200
    max_steps = 50
    gamma = 0.9
    alpha = 0.2
    eps_profile = epsilon_profile.EpsilonProfile(0.00, 0.0)
    final_exploration_episode = 480

    #DQN Hyperparameters
    batch_size = 128
    replay_memory_size = 1000
    target_update_frequency = 2
    tau = 1.0
    
    n_inputs = len(game.get_state())
    
    
    if mode == 'keyboard':
        agent = KeyboardController()
    elif mode == 'random':
        agent = RandomAgent(game.na)
    elif mode == 'dqn':
        model = networks.MLP(n_inputs, game.na)
        weights = torch.load("./training/policy_weights_2022-07-03_13h21_best")
        model.load_state_dict(weights)
        agent = DQNAgent(game, model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    else:
        print('Unknown agent. Starting as random agent.')
        agent = RandomAgent(game.na)
 
    state = game.reset()
    score = 0
    step = 0
    game_over = False
        
    # agent.run_tests(game, 10, 20000)
    # state = game.reset()

    while not game_over:
        pygame.event.get()
        action = agent.select_action(state)
        state, reward, is_done = game.step(action)
        score += reward
        step += 1
        time.sleep(0.001)
        print(f"state {state} | action {action} | score {score}")
        print(f"Step {step} | Score {score}") if reward > 0 else None
        game_over = True if is_done else False

if __name__ == '__main__' :
    if (len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        main('dqn')
