from errno import EPIPE
from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.dqn_agent import DQNAgent
import sys
from classes import epsilon_profile
from classes import networks
import torch
import pygame

def format_state(state):
    player_x = round(state[0])
    invaders_x = [round(x) for x in state[1][0]]
    invaders_y = [round(y) for y in state[1][1]]
    print(f'Player: [{player_x}] | Inv: [{invaders_x},{invaders_y}] | Bullet: {state[2]}')

def main(mode):

    game = SpaceInvaders(display=False)
    
    #Basic hyperparameters 
    n_episodes = 5000
    max_steps = 25000
    gamma = 0.9
    alpha = 0.001
    eps_profile = epsilon_profile.EpsilonProfile(1.0, 0.02)
    final_exploration_episode = 4950
    
    #DQN Hyperparameters
    batch_size = 64
    replay_memory_size = 30000
    target_update_frequency = 5
    tau = 1.0
    
    #Neural network instantiation
    n_inputs = len(game.get_state())
    model = networks.MLP(n_inputs, game.na)
    
    # weights = torch.load("trained_0.47623314486883583")
    # model.load_state_dict(weights)
    
    print('--- neural network ---')
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('number of parameters:', num_params)
    print(model)
    
    agent = DQNAgent(game, model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    agent.learn(game, n_episodes, max_steps)

    state = game.reset()
    score = 0
    step = 0
    game_over = False
    while not game_over:
        pygame.event.get()
        action = agent.select_action(state)
        state, reward, is_done = game.step(action)
        score += reward
        step += 1
        print(f"Step {step} | Score {score}") if reward > 0 else None
        game_over = True if is_done else False

if __name__ == '__main__' :
    if (len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        main('dqn')
